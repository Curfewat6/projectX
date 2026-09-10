"""Offline coverage for the source-reading repository overview graph.

Run with: .venv/bin/python -m unittest discover -s tests -v
"""

import importlib
import io
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

# Disable callbacks before importing LangChain, and never read local .env files.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

with (
    patch("dotenv.load_dotenv", return_value=False),
    patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test-placeholder"}),
):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_ollama import ChatOllama
    from langchain_openrouter import ChatOpenRouter
    from langgraph.graph import END, START, StateGraph
    from langsmith import get_tracing_context, tracing_context
    from pydantic import ValidationError

    from projectx import graph_builder
    from projectx.schemas import Reflection, ReflectionDecision, ThreatModel

    cli = importlib.import_module("projectx.main")


def minimal_report() -> ThreatModel:
    """A valid report which makes no claims about an existing repository."""
    return ThreatModel.model_validate(
        {
            "status": {
                "target": "synthetic-repository",
                "version_commit_or_build": None,
                "lab_state": "Not assessed",
                "current_decision": "exploring",
            },
            "attacker_model": {
                "actor": "Unknown",
                "access_level": "Unknown",
                "inputs_controlled": [],
                "inputs_not_controlled": [],
                "required_prerequisites": [],
                "explicitly_out_of_scope": [],
            },
            "crown_jewels": [],
            "entry_points": {
                "public_or_low_privilege_entry_points": [],
                "internal_attacker_reachable_transitions": [],
                "configuration_dependent_surfaces": [],
            },
            "trust_boundaries": [],
            "high_risk_operations": {
                "parsing_decoding_or_deserialization": [],
                "authn_authz_session_token_logic": [],
                "file_archive_update_plugin_or_template_handling": [],
                "native_kernel_ipc_sandbox_or_privilege_boundary": [],
                "caching_tenant_isolation_or_cross_user_state": [],
            },
            "invariants": [],
            "first_thin_slices": [],
            "variant_analysis": {
                "seed_advisory_issue_commit_patch_or_invariant": None,
                "sibling_pass": [],
                "incomplete_fix_pass": [],
                "bypass_regression_pass": [],
                "negative_results_and_surfaces_ruled_out": [],
                "variants_promoted_for_verification": [],
            },
            "verifier_plan": {
                "static_proof": None,
                "local_model_or_unit_test": None,
                "integration_or_harness": None,
                "disposable_vm_or_crash_only_step": None,
                "negative_control": None,
            },
            "novelty_gate": {
                "current_advisories_cves_ghsas_checked": [],
                "historical_advisories_and_release_notes_checked": [],
                "vendor_upstream_issues_commits_changelogs_and_patches_checked": [],
                "subsystem_function_names_bug_class_terms_and_expressions_searched": [],
                "search_date": None,
                "sources_not_yet_covered": [],
                "candidate_versus_known_issue": "Not assessed",
                "classification": "unresolved",
                "novelty_confidence_and_remaining_collision_risk": "Not assessed",
            },
            "decision_log": [],
        }
    )


def answer_message(answer: str = "Synthetic overview from src/app.py.") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "AnswerQuestion", "args": {"answer": answer}, "id": "answer-1"}
        ],
    )


def read_message(file_path: str, call_id: str = "read-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"file_path": file_path}, "id": call_id}
        ],
    )


def reflection_decision(needs_revision: bool = False) -> ReflectionDecision:
    return ReflectionDecision(
        reflection=Reflection(
            missing=(
                "Inspect component responsibilities." if needs_revision else "None."
            ),
            superfluous="Remove unsupported claims." if needs_revision else "None.",
        ),
        needs_revision=needs_revision,
        reason=(
            "More source evidence is useful." if needs_revision else "Scope is covered."
        ),
    )


class LinearGraphTests(unittest.TestCase):
    def setUp(self):
        network_patch = patch(
            "socket.socket.connect",
            side_effect=AssertionError("Offline tests must not contact a network."),
        )
        network_patch.start()
        self.addCleanup(network_patch.stop)
        temporary_repository = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_repository.cleanup)
        self.repository = Path(temporary_repository.name).resolve()
        (self.repository / "README.md").write_text(
            "The entry point is app.rs.\n", encoding="utf-8"
        )
        (self.repository / "app.rs").write_text(
            'fn main() { println!("Synthetic application"); }\n', encoding="utf-8"
        )
        self.state = {
            "repository_path": str(self.repository),
            "programming_language": "Rust",
        }
        self.checklist = (
            "Inspect README.md and app.rs to identify application components."
        )

    def familiarise(self, responses, **limits):
        chain = Mock(invoke=Mock(side_effect=responses))
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(graph_builder, "build_familiariser", return_value=chain)
            )
            for name, value in limits.items():
                stack.enter_context(patch.object(graph_builder, name, value))
            # ToolNode needs the runtime context supplied by a compiled graph.
            builder = StateGraph(graph_builder.ProjectState)
            builder.add_node("familiarise", graph_builder.familiarise)
            builder.add_edge(START, "familiarise")
            builder.add_edge("familiarise", END)
            result = builder.compile().invoke(
                {
                    **self.state,
                    "repository_tree": "synthetic/\nREADME.md\napp.rs",
                    "checklist": self.checklist,
                }
            )
        return result, chain

    def test_build_graph_prints_mermaid_without_running_steps(self):
        output = io.StringIO()
        with (
            patch.object(graph_builder, "repository_tree_tool") as tree,
            patch.object(graph_builder, "read_threatmodel_chain") as planner,
            patch.object(graph_builder, "build_familiariser") as familiariser,
            patch.object(graph_builder, "generate_threat_model") as generate,
            patch.object(graph_builder, "reflect_threat_model") as reflect,
            redirect_stdout(output),
        ):
            graph = graph_builder.build_graph()
        self.assertIn(graph.get_graph().draw_mermaid().strip(), output.getvalue())
        tree.invoke.assert_not_called()
        planner.invoke.assert_not_called()
        familiariser.assert_not_called()
        generate.assert_not_called()
        reflect.assert_not_called()

    def test_graph_has_requested_initial_path_and_conditional_reflection_loop(self):
        graph = graph_builder.build_graph().get_graph()
        nodes = [
            "__start__",
            "fetch_tree",
            "read_threatmodel",
            "familiarise",
            "fill_threat_model",
            "reflect",
            "__end__",
        ]
        self.assertEqual(set(graph.nodes), set(nodes))
        self.assertEqual(
            {(edge.source, edge.target) for edge in graph.edges},
            set(zip(nodes, nodes[1:])) | {("reflect", "familiarise")},
        )
        self.assertEqual(
            {(edge.source, edge.target) for edge in graph.edges if edge.conditional},
            {("reflect", "familiarise"), ("reflect", "__end__")},
        )

    def test_graph_carries_checklist_and_actual_read_evidence_in_order(self):
        order = []
        report = minimal_report()

        def plan(arguments):
            order.append("read_threatmodel")
            self.assertEqual(arguments["language"], "Rust")
            content = arguments["messages"][0].content
            self.assertIn("README.md", content)
            self.assertIn("crown_jewels", content)
            self.assertIn("trust_boundaries", content)
            return AIMessage(content=self.checklist)

        def familiarise(arguments):
            self.assertEqual(arguments["language"], "Rust")
            self.assertEqual(arguments["checklist"], self.checklist)
            tool_messages = [
                message
                for message in arguments["messages"]
                if isinstance(message, ToolMessage)
            ]
            order.append("familiarise")
            if not tool_messages:
                return read_message("README.md")
            if len(tool_messages) == 1:
                self.assertEqual(tool_messages[0].tool_call_id, "read-1")
                self.assertIn("app.rs", str(tool_messages[0].content))
                return read_message("app.rs", "read-2")
            self.assertEqual(tool_messages[-1].tool_call_id, "read-2")
            self.assertIn("Synthetic application", str(tool_messages[-1].content))
            return answer_message("A synthetic Rust executable [source-2].")

        def generate(context, language):
            order.append("fill_threat_model")
            self.assertEqual(language, "Rust")
            for expected in (
                self.checklist,
                "README.md",
                "app.rs",
                "Synthetic application",
                "source-1",
                "source-2",
            ):
                self.assertIn(expected, context)
            return report

        with (
            patch.object(
                graph_builder,
                "read_threatmodel_chain",
                Mock(invoke=Mock(side_effect=plan)),
            ),
            patch.object(
                graph_builder,
                "build_familiariser",
                return_value=Mock(invoke=Mock(side_effect=familiarise)),
            ),
            patch.object(graph_builder, "generate_threat_model", side_effect=generate),
            patch.object(
                graph_builder,
                "reflect_threat_model",
                return_value=reflection_decision(),
            ),
        ):
            result = graph_builder.build_graph().invoke(self.state)
        self.assertEqual(
            order,
            [
                "read_threatmodel",
                "familiarise",
                "familiarise",
                "familiarise",
                "fill_threat_model",
            ],
        )
        self.assertEqual(result["checklist"], self.checklist)
        self.assertEqual(
            [item["path"] for item in result["evidence"]], ["README.md", "app.rs"]
        )
        for index, item in enumerate(result["evidence"], 1):
            self.assertEqual(item["id"], f"source-{index}")
            self.assertTrue(
                {"start_line", "end_line", "content", "sha256"} <= item.keys()
            )
        self.assertIs(result["threat_model"], report)

    def test_real_chains_and_parser_work_without_network(self):
        report = minimal_report()
        calls = []

        def generate(model, messages, **kwargs):
            self.assertIn("Rust", messages[0].content)
            names = [tool["function"]["name"] for tool in kwargs.get("tools", [])]
            calls.append(names)
            if not names:
                self.assertEqual(names, [])
                self.assertIn("crown_jewels", messages[-1].content)
                message = AIMessage(content=self.checklist)
            elif set(names) == {"read_file", "AnswerQuestion"}:
                message = (
                    read_message("app.rs") if len(calls) == 2 else answer_message()
                )
                if len(calls) == 3:
                    self.assertTrue(
                        any(isinstance(item, ToolMessage) for item in messages)
                    )
            elif names == ["ThreatModel"]:
                self.assertIn("Synthetic application", messages[-1].content)
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ThreatModel",
                            "args": report.model_dump(mode="json"),
                            "id": "report-1",
                        }
                    ],
                )
            elif names == ["ReflectionDecision"]:
                self.assertIn("synthetic-repository", messages[-1].content)
                self.assertIn("Synthetic application", messages[-1].content)
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ReflectionDecision",
                            "args": reflection_decision().model_dump(mode="json"),
                            "id": "reflection-1",
                        }
                    ],
                )
            else:
                self.fail(f"Unexpected bound tools: {names}")
            return ChatResult(generations=[ChatGeneration(message=message)])

        with (
            patch.object(ChatOllama, "_generate", autospec=True, side_effect=generate),
            patch.object(
                ChatOpenRouter, "_generate", autospec=True, side_effect=generate
            ),
        ):
            result = graph_builder.build_graph().invoke(self.state)
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[-2:], [["ThreatModel"], ["ReflectionDecision"]])
        self.assertEqual(result["threat_model"], report)
        self.assertEqual(result["stop_reason"], "approved")

    def test_real_threat_model_parser_rejects_invalid_report(self):
        invalid = AIMessage(
            content="",
            tool_calls=[
                {"name": "ThreatModel", "args": {"status": {}}, "id": "invalid"}
            ],
        )
        with patch.object(
            ChatOpenRouter,
            "_generate",
            return_value=ChatResult(generations=[ChatGeneration(message=invalid)]),
        ):
            with self.assertRaises(ValidationError):
                graph_builder.generate_threat_model("No evidence.", "Rust")

    def test_familiariser_rejects_invalid_responses_and_unread_completion(self):
        invalid = [
            None,
            HumanMessage(content="Wrong role"),
            AIMessage(content="No tool"),
            answer_message(),
            AIMessage(
                content="",
                tool_calls=[{"name": "unknown", "args": {}, "id": "unknown"}],
            ),
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "read_file",
                        "args": "{",
                        "id": "bad",
                        "error": "Invalid JSON",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=read_message("README.md").tool_calls * 2,
            ),
            AIMessage(
                content="",
                tool_calls=(
                    read_message("README.md").tool_calls + answer_message().tool_calls
                ),
            ),
        ]
        for response in invalid:
            with (
                self.subTest(response=response),
                patch.object(
                    graph_builder,
                    "read_threatmodel_chain",
                    Mock(invoke=Mock(return_value=AIMessage(content=self.checklist))),
                ),
                patch.object(
                    graph_builder,
                    "build_familiariser",
                    return_value=Mock(invoke=Mock(return_value=response)),
                ),
                patch.object(graph_builder, "generate_threat_model") as generate,
            ):
                with self.assertRaises(RuntimeError):
                    graph_builder.build_graph().invoke(self.state)
                generate.assert_not_called()

    def test_familiariser_validates_answer_after_reading(self):
        for answer in ("", "  ", []):
            bad_answer = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "AnswerQuestion",
                        "args": {"answer": answer},
                        "id": "answer",
                    }
                ],
            )
            with self.subTest(answer=answer), self.assertRaises(RuntimeError):
                self.familiarise([read_message("README.md"), bad_answer])

    def test_read_failures_are_returned_to_model_and_preserved(self):
        result, chain = self.familiarise(
            [
                read_message("missing.txt"),
                read_message("README.md", "read-2"),
                answer_message(),
            ]
        )
        self.assertTrue(result["read_errors"])
        self.assertEqual(len(result["evidence"]), 1)
        history = chain.invoke.call_args_list[1].args[0]["messages"]
        self.assertTrue(
            any(
                isinstance(message, ToolMessage) and message.status == "error"
                for message in history
            )
        )

    def test_read_and_turn_budgets_stop_loop(self):
        for limit in ("MAX_FILE_READS", "MAX_FAMILIARISATION_TURNS"):
            with self.subTest(limit=limit), self.assertRaises(RuntimeError):
                self.familiarise(
                    [
                        read_message("README.md"),
                        read_message("app.rs", "read-2"),
                        answer_message(),
                    ],
                    **{limit: 1},
                )

    def test_parallel_file_calls_are_matched_to_their_results(self):
        calls = AIMessage(
            content="",
            tool_calls=(
                read_message("README.md", "read-1").tool_calls
                + read_message("app.rs", "read-2").tool_calls
            ),
        )
        result, chain = self.familiarise([calls, answer_message()])
        self.assertEqual(
            {item["path"] for item in result["evidence"]}, {"README.md", "app.rs"}
        )
        history = chain.invoke.call_args_list[1].args[0]["messages"]
        tool_messages = [
            message for message in history if isinstance(message, ToolMessage)
        ]
        self.assertEqual(
            {message.tool_call_id for message in tool_messages}, {"read-1", "read-2"}
        )

    def test_reader_uses_current_repository_not_fixed_path(self):
        first, _ = self.familiarise([read_message("README.md"), answer_message()])
        with tempfile.TemporaryDirectory() as other:
            repository = Path(other).resolve()
            (repository / "README.md").write_text(
                "Different repository.\n", encoding="utf-8"
            )
            self.state["repository_path"] = str(repository)
            second, _ = self.familiarise([read_message("README.md"), answer_message()])
        self.assertIn("app.rs", first["evidence"][0]["content"])
        self.assertIn("Different repository", second["evidence"][0]["content"])

    def test_invalid_checklist_stops_before_familiarisation(self):
        for response in (
            AIMessage(content=""),
            AIMessage(content="   "),
            answer_message(),
        ):
            with (
                self.subTest(response=response),
                patch.object(
                    graph_builder,
                    "read_threatmodel_chain",
                    Mock(invoke=Mock(return_value=response)),
                ),
                patch.object(graph_builder, "build_familiariser") as familiariser,
                self.assertRaises(RuntimeError),
            ):
                graph_builder.build_graph().invoke(self.state)
            familiariser.assert_not_called()

    def test_fetch_tree_does_not_read_file_contents(self):
        with (
            patch.object(
                Path, "read_text", side_effect=AssertionError("No content reads")
            ),
            patch.object(
                Path, "read_bytes", side_effect=AssertionError("No content reads")
            ),
        ):
            result = graph_builder.fetch_tree(self.state)
        self.assertEqual(set(result), {"repository_tree"})
        self.assertIn("README.md", result["repository_tree"])

    def test_tree_failure_stops_before_model_calls(self):
        with (
            patch.object(
                graph_builder,
                "repository_tree_tool",
                Mock(invoke=Mock(side_effect=OSError("Cannot list tree"))),
            ),
            patch.object(graph_builder, "read_threatmodel_chain") as planner,
            self.assertRaisesRegex(OSError, "Cannot list tree"),
        ):
            graph_builder.build_graph().invoke(self.state)
        planner.invoke.assert_not_called()

    def test_cli_arguments_preserve_language_and_resolve_repository(self):
        args = cli.get_args(["--lang", "Rust", "--repo", str(self.repository)])
        self.assertEqual(args.lang, "Rust")
        self.assertEqual(args.repo, self.repository)
        self.assertFalse(args.preview)
        with (
            patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            cli.get_args(["--lang", "Rust", "--repo", str(self.repository / "missing")])

    def test_cli_preview_disables_tracing_and_skips_graph(self):
        def fetch(arguments):
            self.assertIs(get_tracing_context()["enabled"], False)
            return "synthetic-preview/"

        output = io.StringIO()
        with (
            patch.object(
                cli, "repository_tree_tool", Mock(invoke=Mock(side_effect=fetch))
            ),
            patch.object(cli, "build_graph") as build_graph,
            patch(
                "sys.argv",
                [
                    "projectx",
                    "--lang",
                    "Rust",
                    "--repo",
                    str(self.repository),
                    "--preview",
                ],
            ),
            redirect_stdout(output),
            tracing_context(enabled=True),
        ):
            cli.main()
            self.assertIs(get_tracing_context()["enabled"], True)
        self.assertEqual(output.getvalue(), "synthetic-preview/\n")
        build_graph.assert_not_called()

    def test_cli_passes_selected_repository_to_graph(self):
        graph = Mock()
        with (
            patch.object(cli, "build_graph", return_value=graph),
            patch(
                "sys.argv",
                ["projectx", "--lang", "Rust", "--repo", str(self.repository)],
            ),
            redirect_stdout(io.StringIO()),
        ):
            cli.main()
        graph.invoke.assert_called_once_with(self.state)


if __name__ == "__main__":
    unittest.main()
