"""Exercise bounded reflection using real graph nodes and synthetic file reads."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from test_linear_graph import (
    AIMessage,
    ChatGeneration,
    ChatOllama,
    ChatOpenRouter,
    ChatResult,
    Reflection,
    ReflectionDecision,
    ToolMessage,
    ValidationError,
    answer_message,
    graph_builder,
    minimal_report,
    read_message,
)


def decision(needs_revision: bool, round_number: int) -> ReflectionDecision:
    return ReflectionDecision(
        reflection=Reflection(
            missing=f"Round {round_number}: explain module responsibilities.",
            superfluous=f"Round {round_number}: remove unsupported deployment assumptions.",
        ),
        needs_revision=needs_revision,
        reason=f"Round {round_number} evidence assessment.",
    )


class ReflectionLoopTests(unittest.TestCase):
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
        for index in range(4):
            (self.repository / f"module_{index}.py").write_text(
                f'COMPONENT = "synthetic component {index}"\n', encoding="utf-8"
            )
        self.state = {
            "repository_path": str(self.repository),
            "programming_language": "Python",
        }
        self.checklist = "Describe component responsibilities from module_0.py."

    def run_scenario(self, revisions, responses_by_round=None):
        """Mock LLM outputs, but keep routing, state, ToolNode, and reader real."""
        decisions = [decision(value, index) for index, value in enumerate(revisions)]
        reports = []
        familiarisation_inputs = []
        fill_contexts = []
        reflection_contexts = []
        node_events = []
        tree_tool = Mock(invoke=Mock(wraps=graph_builder.repository_tree_tool.invoke))
        planner = Mock(invoke=Mock(return_value=AIMessage(content=self.checklist)))

        def build_familiariser(read_tool):
            round_number = len(familiarisation_inputs)
            inputs = []
            familiarisation_inputs.append(inputs)
            responses = (
                responses_by_round[round_number]
                if responses_by_round is not None
                else [
                    read_message(f"module_{round_number}.py"),
                    answer_message(f"Overview for revision {round_number}."),
                ]
            )

            def invoke(arguments):
                inputs.append(copy.deepcopy(arguments))
                return responses[len(inputs) - 1]

            return Mock(invoke=Mock(side_effect=invoke))

        def generate(context, language):
            self.assertEqual(language, "Python")
            fill_contexts.append(context)
            report = minimal_report()
            report.status.lab_state = f"Synthetic report {len(reports)}; not assessed."
            reports.append(report)
            return report

        def reflect(context, language):
            self.assertEqual(language, "Python")
            reflection_contexts.append(context)
            return decisions[len(reflection_contexts) - 1]

        with (
            patch.object(graph_builder, "repository_tree_tool", tree_tool),
            patch.object(graph_builder, "read_threatmodel_chain", planner),
            patch.object(
                graph_builder, "build_familiariser", side_effect=build_familiariser
            ),
            patch.object(graph_builder, "generate_threat_model", side_effect=generate),
            patch.object(graph_builder, "reflect_threat_model", side_effect=reflect),
            redirect_stdout(io.StringIO()),
        ):
            result = None
            for mode, output in graph_builder.build_graph().stream(
                self.state, stream_mode=["updates", "values"]
            ):
                if mode == "values":
                    result = output
                else:
                    node_events.extend(output)
        self.assertIsNotNone(result)
        return {
            "result": result,
            "decisions": decisions,
            "reports": reports,
            "familiarisation_inputs": familiarisation_inputs,
            "fill_contexts": fill_contexts,
            "reflection_contexts": reflection_contexts,
            "node_events": node_events,
            "tree": tree_tool,
            "planner": planner,
        }

    def test_initial_report_can_be_accepted_without_any_revisions(self):
        run = self.run_scenario([False])
        result = run["result"]
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(result["stop_reason"], "approved")
        self.assertFalse(result["needs_revision"])
        self.assertEqual(result["reflection"], run["decisions"][0].reflection)
        self.assertEqual(len(run["reports"]), 1)
        self.assertEqual(len(run["reflection_contexts"]), 1)
        self.assertEqual(len(result["history"]), 1)
        run["tree"].invoke.assert_called_once_with(
            {"repository_path": str(self.repository)}
        )
        run["planner"].invoke.assert_called_once()

    def test_revision_receives_prior_report_critique_and_original_evidence(self):
        run = self.run_scenario([True, False])
        result = run["result"]
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["stop_reason"], "approved")
        self.assertEqual(len(run["familiarisation_inputs"]), 2)
        second_input = run["familiarisation_inputs"][1][0]
        self.assertEqual(second_input["checklist"], self.checklist)
        familiarisation_text = "\n".join(
            str(message.content) for message in second_input["messages"]
        )
        for content in (
            run["decisions"][0].reflection.missing,
            run["decisions"][0].reflection.superfluous,
            run["reports"][0].status.lab_state,
            "synthetic component 0",
            "source-1",
        ):
            self.assertIn(content, familiarisation_text)
            self.assertIn(content, run["fill_contexts"][1])
        self.assertEqual(
            [item["id"] for item in result["evidence"]], ["source-1", "source-2"]
        )
        self.assertEqual(
            [item["path"] for item in result["evidence"]],
            ["module_0.py", "module_1.py"],
        )
        self.assertIn("synthetic component 1", run["reflection_contexts"][1])
        run["tree"].invoke.assert_called_once()
        run["planner"].invoke.assert_called_once()

    def test_always_revise_stops_after_three_revisions_not_three_total_reports(self):
        run = self.run_scenario([True] * 4)
        result = run["result"]
        self.assertEqual(graph_builder.MAX_REVISIONS, 3)
        self.assertEqual(result["revision_count"], 3)
        self.assertEqual(result["stop_reason"], "max_revisions")
        # Reaching a limit must not falsely turn a failed review into approval.
        self.assertTrue(result["needs_revision"])
        self.assertEqual(result["reflection_reason"], run["decisions"][-1].reason)
        self.assertEqual(len(run["reports"]), 4)
        self.assertEqual(len(run["reflection_contexts"]), 4)
        self.assertEqual(len(run["familiarisation_inputs"]), 4)
        self.assertIs(result["threat_model"], run["reports"][-1])
        self.assertEqual(len(result["history"]), 4)
        self.assertEqual([item["revision"] for item in result["history"]], [0, 1, 2, 3])
        self.assertEqual(
            [item["id"] for item in result["evidence"]],
            ["source-1", "source-2", "source-3", "source-4"],
        )
        run["tree"].invoke.assert_called_once()
        run["planner"].invoke.assert_called_once()

    def test_review_can_end_early_after_one_or_two_revisions(self):
        for revision_count in (1, 2):
            with self.subTest(revision_count=revision_count):
                run = self.run_scenario([True] * revision_count + [False])
                self.assertEqual(run["result"]["revision_count"], revision_count)
                self.assertEqual(run["result"]["stop_reason"], "approved")
                self.assertEqual(len(run["reports"]), revision_count + 1)

    def test_streamed_node_events_revisit_only_familiarise_fill_and_reflect(self):
        run = self.run_scenario([True, True, False])
        self.assertEqual(
            run["node_events"],
            ["fetch_tree", "read_threatmodel"]
            + ["familiarise", "fill_threat_model", "reflect"] * 3,
        )

    def test_approval_on_third_revision_is_still_approval_not_budget_failure(self):
        run = self.run_scenario([True, True, True, False])
        self.assertEqual(run["result"]["revision_count"], 3)
        self.assertEqual(run["result"]["stop_reason"], "approved")
        self.assertFalse(run["result"]["needs_revision"])

    def test_history_keeps_each_report_critique_and_evidence_snapshot(self):
        run = self.run_scenario([True, True, False])
        history = run["result"]["history"]
        self.assertEqual(len(history), 3)
        for index, entry in enumerate(history):
            self.assertEqual(entry["revision"], index)
            self.assertEqual(
                entry["threat_model"], run["reports"][index].model_dump(mode="json")
            )
            self.assertEqual(
                entry["reflection"],
                run["decisions"][index].reflection.model_dump(mode="json"),
            )
            self.assertEqual(entry["needs_revision"], index < 2)
            self.assertEqual(entry["reason"], run["decisions"][index].reason)
            self.assertEqual(
                entry["familiarisation"], f"Overview for revision {index}."
            )
            self.assertEqual(
                entry["evidence_ids"],
                [f"source-{item + 1}" for item in range(index + 1)],
            )
        last_input = run["familiarisation_inputs"][2][0]
        last_text = "\n".join(
            str(message.content) for message in last_input["messages"]
        )
        self.assertIn(run["decisions"][0].reflection.missing, last_text)
        self.assertIn(run["decisions"][1].reflection.missing, last_text)

    def test_multiple_reads_in_one_pass_do_not_count_as_revisions(self):
        calls = AIMessage(
            content="",
            tool_calls=(
                read_message("module_0.py", "read-0").tool_calls
                + read_message("module_1.py", "read-1").tool_calls
            ),
        )
        run = self.run_scenario([False], [[calls, answer_message()]])
        self.assertEqual(run["result"]["revision_count"], 0)
        self.assertEqual(len(run["result"]["evidence"]), 2)
        self.assertEqual(len(run["reports"]), 1)

    def test_revision_can_correct_report_using_existing_evidence_without_new_read(self):
        run = self.run_scenario(
            [True, False],
            [
                [read_message("module_0.py"), answer_message("Initial notes.")],
                [answer_message("Corrected notes using existing source-1.")],
            ],
        )
        self.assertEqual(run["result"]["revision_count"], 1)
        self.assertEqual(len(run["result"]["evidence"]), 1)
        self.assertEqual(len(run["familiarisation_inputs"][1]), 1)
        self.assertIn("synthetic component 0", run["fill_contexts"][1])

    def test_failed_reads_remain_visible_after_revision(self):
        run = self.run_scenario(
            [True, False],
            [
                [
                    read_message("missing.py", "missing"),
                    read_message("module_0.py", "good-0"),
                    answer_message(),
                ],
                [read_message("module_1.py", "good-1"), answer_message()],
            ],
        )
        self.assertEqual(len(run["result"]["read_errors"]), 1)
        self.assertIn("Could not read", run["result"]["read_errors"][0])
        # Reader errors intentionally hide raw OS paths; preserve their content.
        self.assertIn("Could not read", run["fill_contexts"][1])
        self.assertEqual(
            run["result"]["history"][0]["read_errors"],
            run["result"]["history"][1]["read_errors"],
        )

    def test_actual_reflection_parser_rejects_malformed_decision_before_next_read(self):
        malformed = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ReflectionDecision",
                    "args": {"reflection": {"missing": "Something"}},
                    "id": "invalid-reflection",
                }
            ],
        )
        model_result = ChatResult(generations=[ChatGeneration(message=malformed)])
        familiariser = Mock(
            invoke=Mock(side_effect=[read_message("module_0.py"), answer_message()])
        )
        with (
            patch.object(
                graph_builder,
                "read_threatmodel_chain",
                Mock(invoke=Mock(return_value=AIMessage(content=self.checklist))),
            ),
            patch.object(
                graph_builder, "build_familiariser", return_value=familiariser
            ) as build,
            patch.object(
                graph_builder, "generate_threat_model", return_value=minimal_report()
            ) as fill,
            patch.object(ChatOpenRouter, "_generate", return_value=model_result),
            patch.object(ChatOllama, "_generate", return_value=model_result),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(ValidationError):
                graph_builder.build_graph().invoke(self.state)
        build.assert_called_once()
        fill.assert_called_once()
        self.assertEqual(familiariser.invoke.call_count, 2)

    def test_real_chains_complete_a_revision_with_only_provider_transport_mocked(self):
        calls = []
        reports = []
        reviews = []

        def generate(model, messages, **kwargs):
            names = [tool["function"]["name"] for tool in kwargs.get("tools", [])]
            calls.append((names, copy.deepcopy(messages)))
            if not names:
                self.assertIn("crown_jewels", messages[-1].content)
                message = AIMessage(content=self.checklist)
            elif set(names) == {"read_file", "AnswerQuestion"}:
                round_number = len(reports)
                if round_number == 1:
                    prompt_text = "\n".join(str(item.content) for item in messages)
                    self.assertIn(reviews[0].reflection.missing, prompt_text)
                    self.assertIn("synthetic component 0", prompt_text)
                if any(isinstance(item, ToolMessage) for item in messages):
                    message = answer_message(f"Overview for revision {round_number}.")
                else:
                    message = read_message(f"module_{round_number}.py")
            elif names == ["ThreatModel"]:
                if reports:
                    self.assertIn(reviews[0].reflection.missing, messages[-1].content)
                    self.assertIn("synthetic component 1", messages[-1].content)
                report = minimal_report()
                report.status.lab_state = f"Report {len(reports)}; not assessed."
                reports.append(report)
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ThreatModel",
                            "args": report.model_dump(mode="json"),
                            "id": f"report-{len(reports)}",
                        }
                    ],
                )
            elif names == ["ReflectionDecision"]:
                self.assertIn(reports[-1].status.lab_state, messages[-1].content)
                review = decision(len(reviews) == 0, len(reviews))
                reviews.append(review)
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ReflectionDecision",
                            "args": review.model_dump(mode="json"),
                            "id": f"reflection-{len(reviews)}",
                        }
                    ],
                )
            else:
                self.fail(f"Unexpected tools: {names}")
            return ChatResult(generations=[ChatGeneration(message=message)])

        with (
            patch.object(
                ChatOpenRouter, "_generate", autospec=True, side_effect=generate
            ),
            patch.object(ChatOllama, "_generate", autospec=True, side_effect=generate),
            redirect_stdout(io.StringIO()),
        ):
            result = graph_builder.build_graph().invoke(self.state)
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["stop_reason"], "approved")
        self.assertEqual(result["threat_model"], reports[-1])
        self.assertEqual(result["reflection"], reviews[-1].reflection)
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(len(calls), 9)
        self.assertEqual(sum(not names for names, _ in calls), 1)
        self.assertEqual(
            [item["id"] for item in result["evidence"]], ["source-1", "source-2"]
        )

    def test_reflection_without_required_tool_output_cannot_silently_approve(self):
        no_tool_result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Looks good."))]
        )
        with (
            patch.object(ChatOpenRouter, "_generate", return_value=no_tool_result),
            patch.object(ChatOllama, "_generate", return_value=no_tool_result),
            self.assertRaisesRegex(RuntimeError, "ReflectionDecision"),
        ):
            graph_builder.reflect_threat_model("Synthetic evidence only.", "Python")

    def test_reflection_decision_rejects_coerced_flags_and_empty_reasons(self):
        valid = decision(False, 0).model_dump(mode="json")
        for bad_flag in ("false", "true", 0, 1, None):
            with self.subTest(flag=bad_flag), self.assertRaises(ValidationError):
                ReflectionDecision.model_validate({**valid, "needs_revision": bad_flag})
        for reason in ("", "   "):
            with self.subTest(reason=reason), self.assertRaises(ValidationError):
                ReflectionDecision.model_validate({**valid, "reason": reason})


if __name__ == "__main__":
    unittest.main()
