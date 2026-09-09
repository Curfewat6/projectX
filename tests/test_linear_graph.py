"""Offline coverage for the three-node repository overview graph.

Run with: .venv/bin/python -m unittest discover -s tests -v
"""

import importlib
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

# Disable callbacks before importing LangChain, and never read local .env files.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

with patch("dotenv.load_dotenv", return_value=False):
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_ollama import ChatOllama
    from langsmith import get_tracing_context, tracing_context
    from pydantic import ValidationError

    from projectx import graph_builder
    from projectx.familiaraisation_chain import familiarisation_prompt_template
    from projectx.schemas import ThreatModel

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


class LinearGraphTests(unittest.TestCase):
    def setUp(self):
        network_patch = patch(
            "socket.socket.connect",
            side_effect=AssertionError("Offline tests must not contact a network."),
        )
        network_patch.start()
        self.addCleanup(network_patch.stop)
        self.temporary_repository = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_repository.cleanup)
        self.repository = Path(self.temporary_repository.name).resolve()
        self.state = {
            "repository_path": str(self.repository),
            "programming_language": "Rust",
        }

    def test_build_graph_prints_mermaid_without_running_tree_or_models(self):
        output = io.StringIO()

        with (
            patch.object(graph_builder, "repository_tree_tool") as tree_tool,
            patch.object(graph_builder, "familiariser") as familiariser,
            patch.object(graph_builder, "generate_threat_model") as generate_model,
            redirect_stdout(output),
        ):
            graph = graph_builder.build_graph()

        self.assertEqual(
            output.getvalue().strip(), graph.get_graph().draw_mermaid().strip()
        )
        tree_tool.invoke.assert_not_called()
        familiariser.invoke.assert_not_called()
        generate_model.assert_not_called()

    def test_graph_has_only_the_requested_linear_nodes_and_edges(self):
        graph = graph_builder.build_graph().get_graph()

        self.assertEqual(
            set(graph.nodes),
            {"__start__", "fetch_tree", "familiarise", "fill_threat_model", "__end__"},
        )
        self.assertEqual(
            {(edge.source, edge.target) for edge in graph.edges},
            {
                ("__start__", "fetch_tree"),
                ("fetch_tree", "familiarise"),
                ("familiarise", "fill_threat_model"),
                ("fill_threat_model", "__end__"),
            },
        )
        self.assertFalse(any(edge.conditional for edge in graph.edges))

    def test_graph_carries_tree_overview_and_language_in_order(self):
        tree = "synthetic-repository/\n└── src/\n    └── app.rs"
        overview = "A synthetic Rust application; source contents remain unknown."
        report = minimal_report()
        order = []

        def fetch(arguments):
            order.append("fetch_tree")
            self.assertEqual(arguments, {"repository_path": str(self.repository)})
            return tree

        def familiarise(arguments):
            order.append("familiarise")
            self.assertEqual(arguments["language"], "Rust")
            self.assertEqual(len(arguments["messages"]), 1)
            self.assertIsInstance(arguments["messages"][0], HumanMessage)
            content = arguments["messages"][0].content
            self.assertIn(tree, content)
            self.assertIn(str(self.repository), content)
            prompt = familiarisation_prompt_template.invoke(arguments).to_messages()
            self.assertIn("Rust", prompt[0].content)
            return answer_message(overview)

        def generate(context, language):
            order.append("fill_threat_model")
            self.assertEqual(language, "Rust")
            self.assertIn(tree, context)
            self.assertIn(overview, context)
            return report

        tree_tool = Mock(invoke=Mock(side_effect=fetch))
        familiariser = Mock(invoke=Mock(side_effect=familiarise))
        with (
            patch.object(graph_builder, "repository_tree_tool", tree_tool),
            patch.object(graph_builder, "familiariser", familiariser),
            patch.object(
                graph_builder, "generate_threat_model", side_effect=generate
            ) as generate_model,
        ):
            result = graph_builder.build_graph().invoke(self.state)

        self.assertEqual(order, ["fetch_tree", "familiarise", "fill_threat_model"])
        tree_tool.invoke.assert_called_once()
        familiariser.invoke.assert_called_once()
        generate_model.assert_called_once()
        self.assertEqual(result["repository_tree"], tree)
        self.assertEqual(result["familiarisation"], overview)
        self.assertIs(result["threat_model"], report)

    def test_real_chains_propagate_language_and_parse_the_report_offline(self):
        (self.repository / "app.rs").touch()
        report = minimal_report()
        overview = "Synthetic Rust entry point app.rs; contents are unknown."
        generations = []

        def generate(model, messages, **kwargs):
            expected_tool = "AnswerQuestion" if not generations else "ThreatModel"
            self.assertEqual(
                [tool["function"]["name"] for tool in kwargs["tools"]],
                [expected_tool],
            )
            self.assertIn("Rust", messages[0].content)
            self.assertIn(str(self.repository), messages[-1].content)
            self.assertIn("app.rs", messages[-1].content)
            generations.append(expected_tool)
            if expected_tool == "AnswerQuestion":
                message = answer_message(overview)
            else:
                self.assertIn(overview, messages[-1].content)
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
            return ChatResult(generations=[ChatGeneration(message=message)])

        with patch.object(
            ChatOllama, "_generate", autospec=True, side_effect=generate
        ) as model_generate:
            result = graph_builder.build_graph().invoke(self.state)

        self.assertEqual(model_generate.call_count, 2)
        self.assertEqual(generations, ["AnswerQuestion", "ThreatModel"])
        self.assertEqual(result["familiarisation"], overview)
        self.assertIsInstance(result["threat_model"], ThreatModel)
        self.assertEqual(result["threat_model"], report)
        self.assertEqual(
            result["threat_model"].model_dump_json(indent=2),
            report.model_dump_json(indent=2),
        )

    def test_real_threat_model_parser_rejects_an_invalid_report(self):
        invalid_report = AIMessage(
            content="",
            tool_calls=[
                {"name": "ThreatModel", "args": {"status": {}}, "id": "invalid-report"}
            ],
        )
        outputs = [
            ChatResult(generations=[ChatGeneration(message=answer_message())]),
            ChatResult(generations=[ChatGeneration(message=invalid_report)]),
        ]

        with patch.object(
            ChatOllama, "_generate", autospec=True, side_effect=outputs
        ) as model_generate:
            with self.assertRaises(ValidationError):
                graph_builder.build_graph().invoke(self.state)

        self.assertEqual(model_generate.call_count, 2)

    def test_fetch_tree_lists_only_a_synthetic_repository_without_reading_contents(
        self,
    ):
        (self.repository / "src").mkdir()
        (self.repository / "src" / "app.py").touch()
        (self.repository / "README.md").touch()

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
        self.assertIn(self.repository.name + "/", result["repository_tree"])
        self.assertIn("src/", result["repository_tree"])
        self.assertIn("app.py", result["repository_tree"])
        self.assertIn("README.md", result["repository_tree"])

    def test_invalid_familiarisation_stops_before_threat_model(self):
        valid_call = answer_message().tool_calls[0]
        responses = {
            "no response": None,
            "wrong message type": HumanMessage(content="Unstructured answer"),
            "no tool call": AIMessage(content="Unstructured answer"),
            "wrong tool": AIMessage(
                content="",
                tool_calls=[
                    {"name": "OtherTool", "args": {"answer": "Text"}, "id": "other"}
                ],
            ),
            "multiple calls": AIMessage(
                content="", tool_calls=[valid_call, {**valid_call, "id": "answer-2"}]
            ),
            "missing answer": AIMessage(
                content="",
                tool_calls=[{"name": "AnswerQuestion", "args": {}, "id": "missing"}],
            ),
            "invalid answer type": AIMessage(
                content="",
                tool_calls=[
                    {"name": "AnswerQuestion", "args": {"answer": []}, "id": "invalid"}
                ],
            ),
            "empty answer": answer_message(""),
            "blank answer": answer_message(" \n\t "),
            "malformed call": AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "AnswerQuestion",
                        "args": "{",
                        "id": "malformed",
                        "error": "Invalid JSON",
                    }
                ],
            ),
            "valid plus malformed call": AIMessage(
                content="",
                tool_calls=[valid_call],
                invalid_tool_calls=[
                    {
                        "name": "AnswerQuestion",
                        "args": "{",
                        "id": "malformed",
                        "error": "Invalid JSON",
                    }
                ],
            ),
        }

        for label, response in responses.items():
            with self.subTest(response=label):
                with (
                    patch.object(
                        graph_builder,
                        "repository_tree_tool",
                        Mock(invoke=Mock(return_value="synthetic/")),
                    ),
                    patch.object(
                        graph_builder,
                        "familiariser",
                        Mock(invoke=Mock(return_value=response)),
                    ),
                    patch.object(
                        graph_builder, "generate_threat_model"
                    ) as generate_model,
                ):
                    with self.assertRaises(RuntimeError):
                        graph_builder.build_graph().invoke(self.state)
                    generate_model.assert_not_called()

    def test_tree_failure_stops_before_model_calls(self):
        with (
            patch.object(
                graph_builder,
                "repository_tree_tool",
                Mock(invoke=Mock(side_effect=OSError("Cannot list tree"))),
            ),
            patch.object(graph_builder, "familiariser") as familiariser,
            patch.object(graph_builder, "generate_threat_model") as generate_model,
        ):
            with self.assertRaisesRegex(OSError, "Cannot list tree"):
                graph_builder.build_graph().invoke(self.state)

        familiariser.invoke.assert_not_called()
        generate_model.assert_not_called()

    def test_cli_arguments_preserve_language_and_resolve_repository(self):
        args = cli.get_args(["--lang", "Rust", "--repo", str(self.repository)])

        self.assertEqual(args.lang, "Rust")
        self.assertEqual(args.repo, self.repository)
        self.assertFalse(args.preview)

        actual_repository = self.repository / "actual-repository"
        actual_repository.mkdir()
        selected_repository = self.repository / "selected-repository"
        selected_repository.symlink_to(actual_repository, target_is_directory=True)
        linked_args = cli.get_args(
            ["--lang", "Rust", "--repo", str(selected_repository)]
        )
        self.assertEqual(linked_args.repo, actual_repository.resolve())

        with (
            redirect_stdout(io.StringIO()),
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            with self.assertRaises(SystemExit):
                cli.get_args(
                    ["--lang", "Rust", "--repo", str(self.repository / "missing")]
                )

    def test_cli_preview_disables_tracing_without_building_graph_or_calling_models(
        self,
    ):
        def fetch_tree(arguments):
            self.assertIs(get_tracing_context()["enabled"], False)
            return "synthetic-preview/"

        tree_tool = Mock(invoke=Mock(side_effect=fetch_tree))
        output = io.StringIO()
        with (
            patch.object(cli, "repository_tree_tool", tree_tool),
            patch.object(cli, "build_graph") as build_graph,
            patch.object(graph_builder, "familiariser") as familiariser,
            patch.object(graph_builder, "generate_threat_model") as generate_model,
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
        tree_tool.invoke.assert_called_once_with(
            {"repository_path": str(self.repository)}
        )
        build_graph.assert_not_called()
        familiariser.invoke.assert_not_called()
        generate_model.assert_not_called()

    def test_cli_invokes_graph_and_prints_overview_and_structured_report(self):
        report = minimal_report()
        graph = Mock()
        graph.invoke.return_value = {
            **self.state,
            "familiarisation": "Synthetic architecture overview.",
            "threat_model": report,
        }
        output = io.StringIO()

        with (
            patch.object(cli, "build_graph", return_value=graph) as build_graph,
            patch(
                "sys.argv",
                ["projectx", "--lang", "Rust", "--repo", str(self.repository)],
            ),
            redirect_stdout(output),
        ):
            cli.main()

        build_graph.assert_called_once_with()
        graph.invoke.assert_called_once_with(self.state)
        self.assertIn("Synthetic architecture overview.", output.getvalue())
        self.assertIn(report.model_dump_json(indent=2), output.getvalue())


if __name__ == "__main__":
    unittest.main()
