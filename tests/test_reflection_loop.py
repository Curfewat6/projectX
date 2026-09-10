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
    reflection_chain,
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
        # These scenarios exercise a three-visit cap independently of the user's
        # configured production limit.
        iteration_patch = patch.object(graph_builder, "MAX_ITERATIONS", 3)
        iteration_patch.start()
        self.addCleanup(iteration_patch.stop)
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
        improvement_inputs = []
        fill_contexts = []
        reflection_contexts = []
        node_events = []
        tree_tool = Mock(invoke=Mock(wraps=graph_builder.repository_tree_tool.invoke))
        planner = Mock(invoke=Mock(return_value=AIMessage(content=self.checklist)))

        def build_inspector(round_number, recorded_inputs):
            inputs = []
            recorded_inputs.append(inputs)
            responses = (
                responses_by_round[round_number]
                if responses_by_round is not None
                else [
                    read_message(f"module_{round_number}.py"),
                    answer_message(
                        "Initial overview."
                        if round_number == 0
                        else f"Improvement notes for revision {round_number}."
                    ),
                ]
            )

            def invoke(arguments):
                inputs.append(copy.deepcopy(arguments))
                return responses[len(inputs) - 1]

            return Mock(invoke=Mock(side_effect=invoke))

        def build_familiariser(read_tool):
            self.assertEqual(len(familiarisation_inputs), 0)
            return build_inspector(0, familiarisation_inputs)

        def build_improver(read_tool):
            return build_inspector(len(improvement_inputs) + 1, improvement_inputs)

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
            patch.object(
                graph_builder, "build_threat_model_improver", side_effect=build_improver
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
            "improvement_inputs": improvement_inputs,
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
        self.assertEqual(result["iteration_count"], 1)
        self.assertEqual(result["stop_reason"], "approved")
        self.assertFalse(result["needs_revision"])
        self.assertEqual(result["reflection"], run["decisions"][0].reflection)
        self.assertEqual(len(run["reports"]), 1)
        self.assertEqual(len(run["reflection_contexts"]), 1)
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(len(run["familiarisation_inputs"]), 1)
        self.assertEqual(run["improvement_inputs"], [])
        run["tree"].invoke.assert_called_once_with(
            {"repository_path": str(self.repository)}
        )
        run["planner"].invoke.assert_called_once()

    def test_revision_receives_prior_report_critique_and_original_evidence(self):
        run = self.run_scenario([True, False])
        result = run["result"]
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["iteration_count"], 2)
        self.assertEqual(result["stop_reason"], "approved")
        self.assertEqual(len(run["familiarisation_inputs"]), 1)
        self.assertEqual(len(run["improvement_inputs"]), 1)
        second_input = run["improvement_inputs"][0][0]
        self.assertEqual(second_input["checklist"], self.checklist)
        familiarisation_text = "\n".join(
            str(message.content) for message in second_input["messages"]
        )
        self.assertIn("Upcoming iteration: 2 / 3", familiarisation_text)
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
        self.assertEqual(result["familiarisation"], "Initial overview.")
        self.assertEqual(result["improvements"], "Improvement notes for revision 1.")
        for context in (run["fill_contexts"][1], run["reflection_contexts"][1]):
            self.assertIn("Initial overview.", context)
            self.assertIn(result["improvements"], context)
        self.assertIn("Reflection iteration 1 of 3", run["reflection_contexts"][0])
        self.assertIn("Reflection iteration 2 of 3", run["reflection_contexts"][1])
        run["tree"].invoke.assert_called_once()
        run["planner"].invoke.assert_called_once()

    def test_third_reflection_visit_stops_without_calling_the_critic(self):
        # Supplying only two decisions makes any third critic call fail the test.
        run = self.run_scenario([True, True])
        result = run["result"]
        self.assertEqual(result["revision_count"], 2)
        self.assertEqual(result["iteration_count"], 3)
        self.assertEqual(result["stop_reason"], "max_iterations")
        # The final report was not reviewed: neither approve it nor attach the
        # prior report's critique to it as if a third critique had occurred.
        self.assertIsNone(result["needs_revision"])
        self.assertIsNone(result["reflection"])
        self.assertIsNone(result["reflection_reason"])
        self.assertEqual(len(run["reports"]), 3)
        self.assertEqual(len(run["reflection_contexts"]), 2)
        self.assertEqual(len(run["familiarisation_inputs"]), 1)
        self.assertEqual(len(run["improvement_inputs"]), 2)
        self.assertEqual(run["node_events"].count("reflect"), 3)
        self.assertIs(result["threat_model"], run["reports"][-1])
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual([item["revision"] for item in result["history"]], [0, 1])
        self.assertEqual([item["iteration"] for item in result["history"]], [1, 2])
        self.assertEqual(
            [item["id"] for item in result["evidence"]],
            ["source-1", "source-2", "source-3"],
        )
        run["tree"].invoke.assert_called_once()
        run["planner"].invoke.assert_called_once()

    def test_configured_one_and_five_visit_caps_bound_improvements_and_critics(self):
        for cap in (1, 5):
            with (
                self.subTest(cap=cap),
                patch.object(graph_builder, "MAX_ITERATIONS", cap),
            ):
                responses = [
                    [read_message("module_0.py"), answer_message("Initial overview.")]
                ] + [
                    [answer_message(f"Correction {index} using existing source-1.")]
                    for index in range(1, cap)
                ]
                # No decision is supplied for the final visit; calling the critic
                # there would exhaust the mock's responses and fail this test.
                run = self.run_scenario([True] * (cap - 1), responses)
                result = run["result"]
                self.assertEqual(result["iteration_count"], cap)
                self.assertEqual(result["revision_count"], cap - 1)
                self.assertEqual(result["stop_reason"], "max_iterations")
                self.assertEqual(len(run["reports"]), cap)
                self.assertEqual(run["node_events"].count("reflect"), cap)
                self.assertEqual(len(run["reflection_contexts"]), cap - 1)
                self.assertEqual(len(run["improvement_inputs"]), cap - 1)
                self.assertEqual(
                    run["node_events"].count("improve_threat_model"), cap - 1
                )
                self.assertEqual(len(run["familiarisation_inputs"]), 1)
                self.assertEqual(run["node_events"].count("familiarise"), 1)
                self.assertEqual(len(result.get("history", [])), cap - 1)
                self.assertEqual(result["familiarisation"], "Initial overview.")
                self.assertEqual(len(result["evidence"]), 1)
                self.assertIsNone(result["reflection"])
                self.assertIsNone(result["needs_revision"])
                self.assertIs(result["threat_model"], run["reports"][-1])
                run["tree"].invoke.assert_called_once()
                run["planner"].invoke.assert_called_once()

    def test_review_can_end_early_on_first_or_second_iteration(self):
        for revision_count in (0, 1):
            with self.subTest(revision_count=revision_count):
                run = self.run_scenario([True] * revision_count + [False])
                self.assertEqual(run["result"]["revision_count"], revision_count)
                self.assertEqual(run["result"]["iteration_count"], revision_count + 1)
                self.assertEqual(run["result"]["stop_reason"], "approved")
                self.assertEqual(len(run["reports"]), revision_count + 1)
                self.assertEqual(len(run["reflection_contexts"]), revision_count + 1)

    def test_streamed_events_revisit_only_improve_fill_and_reflect(self):
        run = self.run_scenario([True, True])
        self.assertEqual(
            run["node_events"],
            [
                "fetch_tree",
                "read_threatmodel",
                "familiarise",
                "fill_threat_model",
                "reflect",
            ]
            + ["improve_threat_model", "fill_threat_model", "reflect"] * 2,
        )

    def test_capped_reflection_returns_before_reading_report_or_evidence(self):
        with patch.object(
            graph_builder,
            "reflect_threat_model",
            side_effect=AssertionError("The capped visit must not call the critic."),
        ) as critic:
            # No report, language, or evidence is needed to enforce the cap.
            updates = graph_builder.reflection_node({"iteration_count": 2})
        critic.assert_not_called()
        self.assertEqual(
            updates,
            {
                "iteration_count": 3,
                "stop_reason": "max_iterations",
                "reflection": None,
                "needs_revision": None,
                "reflection_reason": None,
            },
        )

    def test_router_obeys_iteration_cap_before_considering_review_decision(self):
        for needs_revision in (True, False, None):
            with self.subTest(needs_revision=needs_revision):
                self.assertEqual(
                    graph_builder.route_after_reflection(
                        {"iteration_count": 3, "needs_revision": needs_revision}
                    ),
                    graph_builder.END,
                )
        self.assertEqual(
            graph_builder.route_after_reflection({"iteration_count": 3}),
            graph_builder.END,
        )

    def test_fill_guard_does_not_generate_a_fourth_report(self):
        with patch.object(graph_builder, "generate_threat_model") as generate:
            with self.assertRaises(RuntimeError):
                graph_builder.fill_threat_model(
                    {
                        "threat_model": minimal_report(),
                        "revision_count": 2,
                        "iteration_count": 3,
                    }
                )
        generate.assert_not_called()

    def test_history_keeps_only_actual_report_reviews_and_evidence_snapshots(self):
        run = self.run_scenario([True, True])
        history = run["result"]["history"]
        self.assertEqual(len(history), 2)
        for index, entry in enumerate(history):
            self.assertEqual(entry["revision"], index)
            self.assertEqual(entry["iteration"], index + 1)
            self.assertEqual(
                entry["threat_model"], run["reports"][index].model_dump(mode="json")
            )
            self.assertEqual(
                entry["reflection"],
                run["decisions"][index].reflection.model_dump(mode="json"),
            )
            self.assertTrue(entry["needs_revision"])
            self.assertEqual(entry["reason"], run["decisions"][index].reason)
            self.assertEqual(entry["familiarisation"], "Initial overview.")
            self.assertEqual(
                entry["improvements"],
                "" if index == 0 else f"Improvement notes for revision {index}.",
            )
            self.assertEqual(
                entry["evidence_ids"],
                [f"source-{item + 1}" for item in range(index + 1)],
            )
        last_input = run["improvement_inputs"][1][0]
        last_text = "\n".join(
            str(message.content) for message in last_input["messages"]
        )
        self.assertIn(run["decisions"][0].reflection.missing, last_text)
        self.assertIn(run["decisions"][1].reflection.missing, last_text)
        self.assertIn("Upcoming iteration: 3 / 3", last_text)

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
        self.assertEqual(run["result"]["iteration_count"], 1)
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
        self.assertEqual(len(run["familiarisation_inputs"]), 1)
        self.assertEqual(len(run["improvement_inputs"][0]), 1)
        self.assertEqual(run["result"]["familiarisation"], "Initial notes.")
        self.assertEqual(
            run["result"]["improvements"], "Corrected notes using existing source-1."
        )
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
            patch.object(reflection_chain, "USE_SCHEMA_CORRECTION", False),
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
            patch.object(graph_builder, "build_threat_model_improver") as improve,
            patch.object(ChatOpenRouter, "_generate", return_value=model_result),
            patch.object(ChatOllama, "_generate", return_value=model_result),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(ValidationError):
                graph_builder.build_graph().invoke(self.state)
        build.assert_called_once()
        improve.assert_not_called()
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
            patch.object(reflection_chain, "USE_SCHEMA_CORRECTION", False),
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
