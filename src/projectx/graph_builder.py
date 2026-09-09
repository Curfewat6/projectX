import os
from typing import NotRequired, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from projectx.familiaraisation_chain import familiariser
from projectx.schemas import AnswerQuestion, ThreatModel
from projectx.threat_model_chain import generate_threat_model
from projectx.tool_executor import repository_tree_tool


class ProjectState(TypedDict):
    """CLI inputs and the output produced by each step of the linear graph."""

    repository_path: str
    programming_language: str
    repository_tree: NotRequired[str]
    familiarisation: NotRequired[str]
    threat_model: NotRequired[ThreatModel]


def fetch_tree(state: ProjectState):
    """Fetch the CLI-selected repository tree without asking an LLM to choose."""
    tree = repository_tree_tool.invoke({"repository_path": state["repository_path"]})
    return {"repository_tree": tree}


def familiarise(state: ProjectState):
    """Pass the CLI language and fetched tree into the familiarisation chain."""
    response = familiariser.invoke(
        {
            "language": state["programming_language"],
            "messages": [
                HumanMessage(
                    content=(
                        f"Selected repository: {state['repository_path']}\n\n"
                        "Repository tree (filenames only; contents not read):\n"
                        f"{state['repository_tree']}"
                    )
                )
            ],
        }
    )

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

    return {"familiarisation": overview.answer}


def fill_threat_model(state: ProjectState):
    """Populate the existing schema from the tree and initial overview."""
    context = (
        f"Selected repository: {state['repository_path']}\n\n"
        "Coverage: directory and file names only. README and source contents "
        "have not been read; no security checks or external searches were run.\n\n"
        f"Repository tree:\n{state['repository_tree']}\n\n"
        "Initial familiarisation (may contain unverified inferences):\n"
        f"{state['familiarisation']}"
    )
    report = generate_threat_model(context, state["programming_language"])
    return {"threat_model": report}


def build_graph():
    builder = StateGraph(ProjectState)
    builder.add_node("fetch_tree", fetch_tree)
    builder.add_node("familiarise", familiarise)
    builder.add_node("fill_threat_model", fill_threat_model)

    builder.add_edge(START, "fetch_tree")
    builder.add_edge("fetch_tree", "familiarise")
    builder.add_edge("familiarise", "fill_threat_model")
    builder.add_edge("fill_threat_model", END)

    graph = builder.compile()

    print("[+] Here's your graph in mermaid. Don't kys!")
    print(graph.get_graph().draw_mermaid())
    return graph

if __name__ == "__main__":
    print(
        f"[INFO] NOT RUNNING Main.py.\n[Info] Now running {os.path.basename(__file__)}"
    )
