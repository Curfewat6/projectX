"""Read-only improvement loop tests using synthetic source files and fake models."""

import copy
import hashlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from test_linear_graph import (
    AIMessage,
    END,
    HumanMessage,
    START,
    StateGraph,
    ToolMessage,
    answer_message,
    graph_builder,
    minimal_report,
    read_message,
    reflection_decision,
)


class ImprovementNodeTests(unittest.TestCase):
    def setUp(self):
        network_patch = patch(
            "socket.socket.connect",
            side_effect=AssertionError("Offline tests must not contact a network."),
        )
        network_patch.start()
        self.addCleanup(network_patch.stop)
        repository = tempfile.TemporaryDirectory()
        self.addCleanup(repository.cleanup)
        self.repository = Path(repository.name).resolve()
        original_text = "Synthetic entry point: app.py.\n"
        (self.repository / "README.md").write_text(original_text, encoding="utf-8")
        (self.repository / "app.py").write_text(
            'COMPONENT = "synthetic application"\n', encoding="utf-8"
        )
        review = reflection_decision(True)
        self.state = {
            "repository_path": str(self.repository),
            "programming_language": "Python",
            "repository_tree": "synthetic/\nREADME.md\napp.py",
            "checklist": "Describe the application's component responsibilities.",
            "familiarisation": "Initial familiarisation remains unchanged.",
            "threat_model": minimal_report(),
            "revision_count": 0,
            "iteration_count": 1,
            "reflection": review.reflection,
            "needs_revision": review.needs_revision,
            "reflection_reason": review.reason,
            "evidence": [
                {
                    "id": "source-1",
                    "path": "README.md",
                    "start_line": 1,
                    "end_line": 1,
                    "content": original_text,
                    "sha256": hashlib.sha256(original_text.encode()).hexdigest(),
                }
            ],
            "read_errors": [],
        }

    def improve(self, responses, **limits):
        inputs = []

        def invoke(arguments):
            inputs.append(copy.deepcopy(arguments))
            return responses[len(inputs) - 1]

        chain = Mock(invoke=Mock(side_effect=invoke))
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    graph_builder, "build_threat_model_improver", return_value=chain
                )
            )
            initial_builder = stack.enter_context(
                patch.object(graph_builder, "build_familiariser")
            )
            writer = stack.enter_context(
                patch.object(graph_builder, "generate_threat_model")
            )
            for name, value in limits.items():
                stack.enter_context(patch.object(graph_builder, name, value))
            # Actual ToolNode execution requires the runtime from a compiled graph.
            builder = StateGraph(graph_builder.ProjectState)
            builder.add_node("improve_threat_model", graph_builder.improve_threat_model)
            builder.add_edge(START, "improve_threat_model")
            builder.add_edge("improve_threat_model", END)
            result = builder.compile().invoke(self.state)
            initial_builder.assert_not_called()
            writer.assert_not_called()
        return result, inputs

    def test_improver_uses_critique_and_prior_evidence_without_overwriting_originals(
        self,
    ):
        before = copy.deepcopy(self.state)
        result, inputs = self.improve(
            [
                read_message("app.py"),
                answer_message("Clarify the app component using source-2."),
            ]
        )
        self.assertEqual(self.state, before)
        self.assertEqual(result["familiarisation"], before["familiarisation"])
        self.assertEqual(result["threat_model"], before["threat_model"])
        self.assertEqual(result["reflection"], before["reflection"])
        self.assertEqual(result["iteration_count"], 1)
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(
            result["improvements"], "Clarify the app component using source-2."
        )
        self.assertEqual(
            [item["id"] for item in result["evidence"]], ["source-1", "source-2"]
        )
        self.assertEqual(result["evidence"][0], before["evidence"][0])
        self.assertEqual(result["evidence"][1]["path"], "app.py")
        self.assertEqual(inputs[0]["language"], "Python")
        self.assertEqual(inputs[0]["checklist"], self.state["checklist"])
        prompt_text = "\n".join(
            str(message.content) for message in inputs[0]["messages"]
        )
        for content in (
            self.state["reflection"].missing,
            self.state["reflection"].superfluous,
            self.state["familiarisation"],
            "source-1",
            "synthetic-repository",
            "app.py",
        ):
            self.assertIn(content, prompt_text)
        tool_messages = [
            item for item in inputs[1]["messages"] if isinstance(item, ToolMessage)
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].tool_call_id, "read-1")
        self.assertIn("source-2", tool_messages[0].content)

    def test_existing_evidence_allows_correction_without_another_file_read(self):
        result, inputs = self.improve(
            [answer_message("Remove unsupported deployment claims.")]
        )
        self.assertEqual(len(inputs), 1)
        self.assertEqual(result["evidence"], self.state["evidence"])
        self.assertEqual(result["familiarisation"], self.state["familiarisation"])
        self.assertEqual(
            result["improvements"], "Remove unsupported deployment claims."
        )

    def test_unread_completion_is_not_accepted(self):
        self.state["evidence"] = []
        with self.assertRaises(RuntimeError):
            self.improve([answer_message()])

    def test_missing_review_or_reached_cap_stops_before_constructing_tools(self):
        invalid_states = [
            {},
            {"threat_model": minimal_report(), "reflection": None},
            {
                "threat_model": minimal_report(),
                "reflection": reflection_decision(True).reflection,
                "iteration_count": graph_builder.MAX_ITERATIONS,
            },
        ]
        for state in invalid_states:
            with (
                self.subTest(state=state),
                patch.object(graph_builder, "make_read_file_tool") as make_tool,
                patch.object(graph_builder, "build_threat_model_improver") as build,
                self.assertRaises(RuntimeError),
            ):
                graph_builder.improve_threat_model(state)
            make_tool.assert_not_called()
            build.assert_not_called()

    def test_invalid_tool_requests_and_invalid_terminal_answers_are_rejected(self):
        invalid = [
            None,
            HumanMessage(content="Wrong role"),
            AIMessage(content="No tool call"),
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
            AIMessage(content="", tool_calls=read_message("app.py").tool_calls * 2),
            AIMessage(
                content="",
                tool_calls=read_message("app.py").tool_calls
                + answer_message().tool_calls,
            ),
            answer_message(""),
            answer_message("  "),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "AnswerQuestion", "args": {"answer": []}, "id": "answer"}
                ],
            ),
        ]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                self.improve([response])

    def test_file_and_model_turn_budgets_also_bound_improvement_calls(self):
        for limit in ("MAX_FILE_READS", "MAX_IMPROVEMENT_TURNS"):
            with self.subTest(limit=limit), self.assertRaises(RuntimeError):
                self.improve(
                    [
                        read_message("README.md"),
                        read_message("app.py", "read-2"),
                        answer_message(),
                    ],
                    **{limit: 1},
                )

    def test_failed_reads_reach_model_and_do_not_become_source_evidence(self):
        result, inputs = self.improve(
            [
                read_message("missing.py"),
                answer_message("Leave missing implementation details unknown."),
            ]
        )
        self.assertEqual(result["evidence"], self.state["evidence"])
        self.assertEqual(len(result["read_errors"]), 1)
        self.assertIn("Could not read", result["read_errors"][0])
        self.assertTrue(
            any(
                isinstance(item, ToolMessage) and item.status == "error"
                for item in inputs[1]["messages"]
            )
        )


if __name__ == "__main__":
    unittest.main()
