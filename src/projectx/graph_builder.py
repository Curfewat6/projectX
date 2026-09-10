import json
import os
from typing import Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from projectx.familiaraisation_chain import (
    build_familiariser,
    read_threatmodel_chain,
)
from projectx.reflection_chain import reflect_threat_model
from projectx.schemas import AnswerQuestion, Reflection, ReflectionDecision, ThreatModel
from projectx.threat_model_chain import generate_threat_model
from projectx.tool_executor import make_read_file_tool, repository_tree_tool

MAX_FILE_READS = 8
MAX_FAMILIARISATION_TURNS = 10
MAX_CHECKLIST_CHARS = 8000
MAX_REVISIONS = 3  # Initial draft is revision 0; permit three improvements.


class ProjectState(TypedDict):
    """CLI inputs, accumulated evidence, and reviews retained across revisions."""

    repository_path: str
    programming_language: str
    repository_tree: NotRequired[str]
    checklist: NotRequired[str]
    familiarisation: NotRequired[str]
    evidence: NotRequired[list[dict]]
    read_errors: NotRequired[list[str]]
    threat_model: NotRequired[ThreatModel]
    revision_count: NotRequired[int]
    reflection: NotRequired[Reflection]
    needs_revision: NotRequired[bool]
    reflection_reason: NotRequired[str]
    history: NotRequired[list[dict]]
    stop_reason: NotRequired[Literal["approved", "max_revisions"] | None]


def fetch_tree(state: ProjectState):
    """Fetch the CLI-selected repository tree without asking an LLM to choose."""
    tree = repository_tree_tool.invoke({"repository_path": state["repository_path"]})
    return {"repository_tree": tree}


def read_threatmodel(state: ProjectState):
    """Turn the actual schema and tree into a checklist, without reading files."""
    schema = json.dumps(ThreatModel.model_json_schema(), indent=2)
    response = read_threatmodel_chain.invoke(
        {
            "language": state["programming_language"],
            "messages": [
                HumanMessage(
                    content=(
                        f"Selected repository: {state['repository_path']}\n\n"
                        f"Threat-model template (JSON Schema):\n{schema}\n\n"
                        "Repository tree (filenames only; contents not read):\n"
                        f"{state['repository_tree']}"
                    )
                )
            ],
        }
    )

    if (
        not isinstance(response, AIMessage)
        or response.tool_calls
        or response.invalid_tool_calls
        or not isinstance(response.content, str)
        or not response.content.strip()
    ):
        raise RuntimeError(
            "Template reading must return a nonempty plain-text checklist."
        )
    checklist = response.content.strip()
    # if len(checklist) > MAX_CHECKLIST_CHARS:
    #     raise RuntimeError(
    #         "The inspection checklist is too long; request a shorter plan."
    #     )
    return {"checklist": checklist}


def _overview_from_response(response: AIMessage) -> str:
    """Validate the terminal schema call; never execute it as a file tool."""

    # AnswerQuestion is an output schema here, not a filesystem tool to execute.
    if (
        not isinstance(response, AIMessage)
        or response.invalid_tool_calls
        or len(response.tool_calls) != 1
        or response.tool_calls[0]["name"] != "AnswerQuestion"
    ):
        raise RuntimeError(
            "Familiarisation must return exactly one AnswerQuestion tool call."
        )
    try:
        overview = AnswerQuestion.model_validate(response.tool_calls[0]["args"])
    except ValidationError as error:
        raise RuntimeError(
            "Familiarisation returned invalid AnswerQuestion arguments."
        ) from error
    if not overview.answer.strip():
        raise RuntimeError("Familiarisation returned an empty answer.")

    return overview.answer


def _prior_review_context(state: ProjectState) -> str:
    """Carry prior work to the next call without replaying every old tool turn."""
    if "threat_model" not in state:
        return ""
    critiques = [
        {
            "revision": item["revision"],
            "reflection": item["reflection"],
            "needs_revision": item["needs_revision"],
            "reason": item["reason"],
        }
        for item in state.get("history", [])
    ]
    latest = state.get("reflection")
    # During fill_threat_model, state['familiarisation'] is already the NEW
    # overview. Retrieve the reviewed overview from its immutable history entry.
    history = state.get("history", [])
    previous_overview = (
        history[-1]["familiarisation"] if history else state.get("familiarisation", "")
    )
    return (
        f"Completed revisions: {state.get('revision_count', 0)} / {MAX_REVISIONS}\n\n"
        "Previous threat model (an assessment, not independent source evidence):\n"
        f"{state['threat_model'].model_dump_json()}\n\n"
        f"Previous familiarisation overview:\n{previous_overview}\n\n"
        "Latest critique to address (may itself contain mistakes):\n"
        f"{latest.model_dump_json() if latest is not None else 'Not reviewed yet.'}\n"
        f"Decision rationale: {state.get('reflection_reason', '')}\n\n"
        "Earlier review history (older feedback may already be resolved):\n"
        f"{json.dumps(critiques, ensure_ascii=False)}"
    )


def familiarise(state: ProjectState):
    """Read toward the checklist/critique while preserving earlier evidence."""
    read_tool = make_read_file_tool(state["repository_path"])
    familiariser = build_familiariser(read_tool)
    execute_reads = ToolNode([read_tool], handle_tool_errors=True)
    messages = [
        HumanMessage(
            content=(
                f"Selected repository: {state['repository_path']}\n\n"
                "Repository tree (filenames only; source evidence is separate):\n"
                f"{state['repository_tree']}"
            )
        )
    ]
    # Copy the lists: return updated state without mutating prior graph snapshots.
    evidence = [dict(item) for item in state.get("evidence", [])]
    read_errors = list(state.get("read_errors", []))
    previous_work = _prior_review_context(state)
    if previous_work:
        messages.append(
            HumanMessage(
                content=(
                    "This is a revision pass. Address the latest critique using the "
                    "existing evidence and targeted additional reads as needed. "
                    "Do not restart discovery or assume earlier claims are facts.\n\n"
                    f"{previous_work}\n\n"
                    "Previously collected source evidence (reuse these source IDs):\n"
                    f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
                    f"Previous failed or empty reads:\n{json.dumps(read_errors)}"
                )
            )
        )
    read_count = 0
    seen_call_ids = set()

    for _ in range(MAX_FAMILIARISATION_TURNS):
        response = familiariser.invoke(
            {
                "language": state["programming_language"],
                "checklist": state["checklist"],
                "max_file_reads": MAX_FILE_READS,
                "messages": messages,
            }
        )
        if (
            not isinstance(response, AIMessage)
            or response.invalid_tool_calls
            or not response.tool_calls
        ):
            raise RuntimeError(
                "Familiarisation must request read_file or finish with AnswerQuestion."
            )

        for call in response.tool_calls:
            call_id = call.get("id")
            if not call_id or call_id in seen_call_ids:
                raise RuntimeError("Every tool call must have a unique nonempty ID.")
            seen_call_ids.add(call_id)

        names = [call["name"] for call in response.tool_calls]
        if "AnswerQuestion" in names:
            overview = _overview_from_response(response)
            if not evidence:
                raise RuntimeError(
                    "Familiarisation finished without successfully reading a nonempty file. "
                    "No source-backed threat model can be produced."
                )
            return {
                "familiarisation": overview,
                "evidence": evidence,
                "read_errors": read_errors,
            }

        if any(name != "read_file" for name in names):
            raise RuntimeError(
                "Only the read_file tool is available during familiarisation."
            )
        if read_count + len(names) > MAX_FILE_READS:
            raise RuntimeError(
                f"Familiarisation exceeded its {MAX_FILE_READS}-call file-read budget."
            )

        read_count += len(names)
        messages.append(response)
        # ToolNode returns ToolMessages linked to the model's tool-call IDs.
        results = execute_reads.invoke({"messages": [response]})["messages"]
        for result in results:
            if not isinstance(result, ToolMessage):
                raise RuntimeError("The file reader did not return a ToolMessage.")
            artifact = result.artifact
            if result.status == "success" and isinstance(artifact, dict):
                if artifact.get("content", "").strip():
                    record = {**artifact, "id": f"source-{len(evidence) + 1}"}
                    evidence.append(record)
                    result = result.model_copy(
                        update={"content": f"[{record['id']}]\n{result.content}"}
                    )
                else:
                    read_errors.append(
                        f"No nonempty source excerpt: {artifact.get('path')}"
                    )
            else:
                read_errors.append(str(result.content))
            messages.append(result)

    raise RuntimeError(
        f"Familiarisation did not finish within {MAX_FAMILIARISATION_TURNS} model turns."
    )


def _evidence_context(state: ProjectState) -> str:
    """The same observed material is supplied to both the writer and reviewer."""
    return (
        f"Selected repository: {state['repository_path']}\n\n"
        "Coverage: tree plus ONLY the source excerpts recorded below. "
        "Unlisted files and omitted lines have not been read. A read is not a "
        "runtime test, security verification, or external advisory search.\n\n"
        f"Repository tree:\n{state['repository_tree']}\n\n"
        f"Inspection checklist (not proof of completed work):\n{state['checklist']}\n\n"
        "Familiarisation overview (interpretation, not independent evidence):\n"
        f"{state['familiarisation']}\n\n"
        "Actual source evidence (each sha256 identifies the file snapshot read):\n"
        f"{json.dumps(state['evidence'], ensure_ascii=False)}\n\n"
        f"Failed or empty reads:\n{json.dumps(state['read_errors'], ensure_ascii=False)}"
    )


def fill_threat_model(state: ProjectState):
    """Write/revise from evidence and critique, counting completed improvements."""
    revision_count = state.get("revision_count", 0)
    is_revision = "threat_model" in state
    if is_revision and revision_count >= MAX_REVISIONS:
        raise RuntimeError(
            f"The {MAX_REVISIONS}-revision limit has already been reached."
        )

    context = _evidence_context(state)
    if is_revision:
        context += (
            "\n\nRevise the previous threat model using the latest critique and "
            "the evidence above. Preserve supported material, correct unsupported "
            "claims, and leave unresolved facts explicitly unknown.\n\n"
            + _prior_review_context(state)
        )
    report = generate_threat_model(context, state["programming_language"])
    return {
        "threat_model": report,
        "revision_count": revision_count + int(is_revision),
    }


def reflection_node(state: ProjectState):
    """Critique the completed report and record the model's improve/end decision."""
    revision_count = state.get("revision_count", 0)
    context = (
        f"Review the current threat model: revision {revision_count}.\n"
        "The initial draft is revision 0. A stop at the revision cap does not "
        "mean this report was approved; honestly record remaining issues.\n\n"
        f"Current threat model:\n{state['threat_model'].model_dump_json()}\n\n"
        f"{_evidence_context(state)}\n\n"
        "Previous critiques (use to check whether feedback was addressed):\n"
        + json.dumps(
            [
                {
                    "revision": item["revision"],
                    "reflection": item["reflection"],
                    "reason": item["reason"],
                }
                for item in state.get("history", [])
            ],
            ensure_ascii=False,
        )
    )
    review = reflect_threat_model(context, state["programming_language"])
    if not isinstance(review, ReflectionDecision):
        raise RuntimeError("Reflection did not return a validated ReflectionDecision.")

    iteration = {
        "revision": revision_count,
        "threat_model": state["threat_model"].model_dump(mode="json"),
        "reflection": review.reflection.model_dump(mode="json"),
        "needs_revision": review.needs_revision,
        "reason": review.reason,
        "familiarisation": state["familiarisation"],
        "evidence_ids": [item["id"] for item in state["evidence"]],
        "read_errors": list(state["read_errors"]),
    }
    stop_reason = None
    if not review.needs_revision:
        stop_reason = "approved"
    elif revision_count >= MAX_REVISIONS:
        stop_reason = "max_revisions"
    return {
        "reflection": review.reflection,
        "needs_revision": review.needs_revision,
        "reflection_reason": review.reason,
        "history": [*state.get("history", []), iteration],
        "stop_reason": stop_reason,
    }


def route_after_reflection(state: ProjectState) -> Literal["familiarise", "__end__"]:
    """The model chooses whether improvement is useful; code enforces the cap."""
    if state["needs_revision"] and state.get("revision_count", 0) < MAX_REVISIONS:
        return "familiarise"
    return END


def build_graph():
    builder = StateGraph(ProjectState)
    builder.add_node("fetch_tree", fetch_tree)
    builder.add_node("read_threatmodel", read_threatmodel)
    builder.add_node("familiarise", familiarise)
    builder.add_node("fill_threat_model", fill_threat_model)
    builder.add_node("reflect", reflection_node)

    builder.add_edge(START, "fetch_tree")
    builder.add_edge("fetch_tree", "read_threatmodel")
    builder.add_edge("read_threatmodel", "familiarise")
    builder.add_edge("familiarise", "fill_threat_model")
    builder.add_edge("fill_threat_model", "reflect")
    builder.add_conditional_edges(
        "reflect", route_after_reflection, {"familiarise": "familiarise", END: END}
    )

    graph = builder.compile()

    print("[+] Here's your graph in mermaid. Don't kys!")
    print(graph.get_graph().draw_mermaid())
    return graph


if __name__ == "__main__":
    print(
        f"[INFO] NOT RUNNING Main.py.\n[Info] Now running {os.path.basename(__file__)}"
    )
