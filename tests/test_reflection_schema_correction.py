"""Offline tests for correction of the nested reflection schema."""

import copy
import io
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

# The bootstrap disables tracing and prevents reading the project's .env file.
from test_linear_graph import (
    AIMessage,
    ChatGeneration,
    ChatOllama,
    ChatOpenRouter,
    ChatResult,
    HumanMessage,
    ReflectionDecision,
    ToolMessage,
    ValidationError,
    answer_message,
    graph_builder,
    llm,
    minimal_report,
    read_message,
    reflection_chain,
    reflection_decision,
)


def review_message(*, valid=True, needs_revision=False, call_id="review-1"):
    arguments = reflection_decision(needs_revision).model_dump(mode="json")
    if not valid:
        # Reproduce the user's failure: both nested string fields are objects
        # inside lists. The surrounding routing fields remain valid.
        arguments["reflection"] = {
            "missing": [{"issue": "Describe the component responsibilities."}],
            "superfluous": [{"issue": "Remove unsupported deployment claims."}],
        }
    return AIMessage(
        content="",
        tool_calls=[{"name": "ReflectionDecision", "args": arguments, "id": call_id}],
    )


class ReflectionSchemaCorrectionTests(unittest.TestCase):
    def setUp(self):
        for current_patch in (
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("Offline tests must not contact a network."),
            ),
            patch.object(reflection_chain, "USE_SCHEMA_CORRECTION", True),
        ):
            current_patch.start()
            self.addCleanup(current_patch.stop)
        self.context = (
            "Current report: a synthetic application component.\n"
            "Checklist: explain component responsibilities.\n"
            "source-1 README.md:1: Synthetic source excerpt; no runtime checks.\n"
            "Prior critique: deployment details remain unknown."
        )

    @contextmanager
    def model_responses(self, responses):
        """Use the actual inner agent/validators, mocking only model transport."""
        calls = []

        def generate(model, messages, **kwargs):
            self.assertIs(model, llm)
            calls.append(
                {
                    "messages": copy.deepcopy(messages),
                    "tools": copy.deepcopy(kwargs.get("tools", [])),
                }
            )
            self.assertLessEqual(
                len(calls), len(responses), "Unexpected extra reflection model call"
            )
            response = responses[len(calls) - 1]
            if isinstance(response, Exception):
                raise response
            return ChatResult(generations=[ChatGeneration(message=response)])

        with (
            patch.object(ChatOllama, "_generate", autospec=True, side_effect=generate),
            patch.object(
                ChatOpenRouter, "_generate", autospec=True, side_effect=generate
            ),
        ):
            yield calls

    def test_valid_first_response_returns_decision_without_legacy_chain(self):
        with (
            self.model_responses([review_message()]) as calls,
            patch.object(reflection_chain, "reflection_chain") as legacy_chain,
        ):
            review = reflection_chain.reflect_threat_model(self.context, "Rust")
        self.assertIsInstance(review, ReflectionDecision)
        self.assertEqual(review, reflection_decision(False))
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [tool["function"]["name"] for tool in calls[0]["tools"]],
            ["ReflectionDecision"],
        )
        self.assertIn("Rust", calls[0]["messages"][0].content)
        self.assertEqual(calls[0]["messages"][-1].content, self.context)
        legacy_chain.invoke.assert_not_called()

    def test_both_nested_errors_are_returned_with_original_prompt_and_context(self):
        invalid = review_message(valid=False, call_id="invalid-review")
        with self.model_responses(
            [invalid, review_message(call_id="corrected-review")]
        ) as calls:
            review = reflection_chain.reflect_threat_model(self.context, "Rust")
        self.assertEqual(review, reflection_decision(False))
        self.assertEqual(len(calls), 2)
        first_messages = calls[0]["messages"]
        correction_messages = calls[1]["messages"]
        self.assertEqual(correction_messages[: len(first_messages)], first_messages)
        human_messages = [
            item for item in correction_messages if isinstance(item, HumanMessage)
        ]
        self.assertEqual(len(human_messages), 1)
        self.assertEqual(human_messages[0].content, self.context)
        rejected_messages = [
            item for item in correction_messages if isinstance(item, AIMessage)
        ]
        self.assertEqual(rejected_messages[0].tool_calls, invalid.tool_calls)
        feedback = [
            item for item in correction_messages if isinstance(item, ToolMessage)
        ]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0].tool_call_id, "invalid-review")
        self.assertIn("reflection.missing", feedback[0].content)
        self.assertIn("reflection.superfluous", feedback[0].content)
        self.assertIn("string", feedback[0].content.lower())

    def test_fifth_attempt_can_return_a_valid_decision(self):
        responses = [
            review_message(valid=False, call_id=f"invalid-{index}")
            for index in range(4)
        ] + [review_message(needs_revision=True, call_id="valid-5")]
        with self.model_responses(responses) as calls:
            review = reflection_chain.reflect_threat_model(self.context, "Rust")
        self.assertEqual(review, reflection_decision(True))
        self.assertEqual(len(calls), 5)

    def test_repeated_nested_errors_continue_until_valid_sixth_response(self):
        responses = [
            review_message(valid=False, call_id=f"invalid-{index}")
            for index in range(5)
        ] + [review_message(call_id="valid-6")]
        with self.model_responses(responses) as calls:
            review = reflection_chain.reflect_threat_model(self.context, "Rust")
        self.assertEqual(review, reflection_decision(False))
        self.assertEqual(len(calls), 6)
        feedback = [
            item for item in calls[-1]["messages"] if isinstance(item, ToolMessage)
        ]
        self.assertEqual(len(feedback), 5)
        for message in feedback:
            self.assertIn("reflection.missing", message.content)
            self.assertIn("reflection.superfluous", message.content)

    def test_missing_decision_tool_can_be_corrected_after_more_than_three_calls(self):
        responses = [AIMessage(content="Looks good.") for _ in range(4)] + [
            review_message()
        ]
        with self.model_responses(responses) as calls:
            review = reflection_chain.reflect_threat_model(self.context, "Rust")
        self.assertEqual(review, reflection_decision(False))
        self.assertEqual(len(calls), 5)

    def test_correction_agent_has_no_explicit_model_call_limit_middleware(self):
        with (
            patch.object(
                reflection_chain, "create_agent", wraps=reflection_chain.create_agent
            ) as create_agent,
            self.model_responses([review_message()]) as calls,
        ):
            reflection_chain.reflect_threat_model(self.context, "Rust")
        self.assertEqual(len(calls), 1)
        create_agent.assert_called_once()
        self.assertEqual(create_agent.call_args.kwargs.get("middleware", []), [])

    def test_each_review_has_a_fresh_correction_conversation(self):
        responses = [
            review_message(valid=attempt == 4, call_id=f"review-{review}-{attempt}")
            for review in range(2)
            for attempt in range(5)
        ]
        with self.model_responses(responses) as calls:
            first = reflection_chain.reflect_threat_model(
                "First report evidence.", "Rust"
            )
            second = reflection_chain.reflect_threat_model(
                "Second report evidence.", "Rust"
            )
        self.assertEqual(first, reflection_decision(False))
        self.assertEqual(second, reflection_decision(False))
        self.assertEqual(len(calls), 10)
        second_start = calls[5]["messages"]
        self.assertFalse(
            any(isinstance(item, (AIMessage, ToolMessage)) for item in second_start)
        )
        self.assertEqual(second_start[-1].content, "Second report evidence.")

    def test_disabled_correction_preserves_single_call_validation_error(self):
        with (
            patch.object(reflection_chain, "USE_SCHEMA_CORRECTION", False),
            patch.object(reflection_chain, "create_agent") as create_agent,
            self.model_responses([review_message(valid=False)]) as calls,
            self.assertRaises(ValidationError) as failure,
        ):
            reflection_chain.reflect_threat_model(self.context, "Rust")
        self.assertEqual(len(calls), 1)
        self.assertIn("reflection.missing", str(failure.exception))
        self.assertIn("reflection.superfluous", str(failure.exception))
        create_agent.assert_not_called()

    def test_disabled_correction_still_returns_valid_decision(self):
        with (
            patch.object(reflection_chain, "USE_SCHEMA_CORRECTION", False),
            patch.object(reflection_chain, "create_agent") as create_agent,
            self.model_responses([review_message()]) as calls,
        ):
            review = reflection_chain.reflect_threat_model(self.context, "Rust")
        self.assertEqual(review, reflection_decision(False))
        self.assertEqual(len(calls), 1)
        create_agent.assert_not_called()

    def test_neither_branch_can_return_missing_or_unvalidated_decision(self):
        for enabled in (True, False):
            for decision in (None, reflection_decision(False).model_dump(mode="json")):
                with (
                    self.subTest(enabled=enabled, decision=decision),
                    patch.object(reflection_chain, "USE_SCHEMA_CORRECTION", enabled),
                ):
                    if enabled:
                        backend_patch = patch.object(
                            reflection_chain,
                            "create_agent",
                            return_value=Mock(
                                invoke=Mock(
                                    return_value={"structured_response": decision}
                                )
                            ),
                        )
                    else:
                        backend_patch = patch.object(
                            reflection_chain,
                            "reflection_chain",
                            Mock(
                                invoke=Mock(
                                    return_value={
                                        "parsed": decision,
                                        "parsing_error": None,
                                    }
                                )
                            ),
                        )
                    with (
                        backend_patch,
                        self.assertRaisesRegex(RuntimeError, "ReflectionDecision"),
                    ):
                        reflection_chain.reflect_threat_model(self.context, "Rust")

    def test_repairs_do_not_count_as_outer_iterations_or_record_invalid_decisions(self):
        responses = [
            review_message(
                valid=attempt == 4,
                needs_revision=revision == 0 if attempt == 4 else revision != 0,
                call_id=f"revision-{revision}-attempt-{attempt}",
            )
            for revision in range(2)
            for attempt in range(5)
        ]
        familiariser = Mock(
            invoke=Mock(
                side_effect=[
                    read_message("README.md"),
                    answer_message(),
                ]
            )
        )
        improvements = "Corrected notes using existing source-1."
        improver = Mock(invoke=Mock(return_value=answer_message(improvements)))
        with (
            tempfile.TemporaryDirectory() as temporary_repository,
            patch.object(graph_builder, "MAX_ITERATIONS", 3),
            patch.object(
                graph_builder,
                "read_threatmodel_chain",
                Mock(invoke=Mock(return_value=AIMessage(content="Read README.md."))),
            ),
            patch.object(
                graph_builder, "build_familiariser", return_value=familiariser
            ) as build,
            patch.object(
                graph_builder, "build_threat_model_improver", return_value=improver
            ) as build_improver,
            patch.object(
                graph_builder, "generate_threat_model", return_value=minimal_report()
            ) as fill,
            self.model_responses(responses) as calls,
            redirect_stdout(io.StringIO()),
        ):
            repository = Path(temporary_repository).resolve()
            (repository / "README.md").write_text(
                "Synthetic component.\n", encoding="utf-8"
            )
            graph = graph_builder.build_graph()
            events = []
            result = None
            for mode, output in graph.stream(
                {"repository_path": str(repository), "programming_language": "Rust"},
                stream_mode=["updates", "values"],
            ):
                if mode == "updates":
                    events.extend(output)
                else:
                    result = output
        self.assertEqual(len(calls), 10)
        build.assert_called_once()
        build_improver.assert_called_once()
        self.assertEqual(familiariser.invoke.call_count, 2)
        improver.invoke.assert_called_once()
        self.assertEqual(fill.call_count, 2)
        self.assertEqual(result["iteration_count"], 2)
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["stop_reason"], "approved")
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual([item["iteration"] for item in result["history"]], [1, 2])
        self.assertEqual(
            [item["needs_revision"] for item in result["history"]], [True, False]
        )
        self.assertEqual(result["reflection"], reflection_decision(False).reflection)
        self.assertEqual(
            result["familiarisation"], "Synthetic overview from src/app.py."
        )
        self.assertEqual(result["improvements"], improvements)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("source-1", calls[5]["messages"][-1].content)
        self.assertIn("Reflection iteration 2 of 3", calls[5]["messages"][-1].content)
        self.assertIn(
            "Inspect component responsibilities", calls[5]["messages"][-1].content
        )
        self.assertEqual(
            events,
            [
                "fetch_tree",
                "read_threatmodel",
                "familiarise",
                "fill_threat_model",
                "reflect",
                "improve_threat_model",
                "fill_threat_model",
                "reflect",
            ],
        )
        self.assertNotIn("correction", graph.get_graph().nodes)

    def test_provider_failure_during_correction_stops_outer_graph_before_improvement(
        self,
    ):
        responses = [
            review_message(valid=False, needs_revision=True, call_id=f"invalid-{index}")
            for index in range(4)
        ] + [RuntimeError("Synthetic provider failure")]
        familiariser = Mock(
            invoke=Mock(side_effect=[read_message("README.md"), answer_message()])
        )
        with (
            tempfile.TemporaryDirectory() as temporary_repository,
            patch.object(
                graph_builder,
                "read_threatmodel_chain",
                Mock(invoke=Mock(return_value=AIMessage(content="Read README.md."))),
            ),
            patch.object(
                graph_builder, "build_familiariser", return_value=familiariser
            ) as build,
            patch.object(graph_builder, "build_threat_model_improver") as improve,
            patch.object(
                graph_builder, "generate_threat_model", return_value=minimal_report()
            ) as fill,
            patch.object(
                graph_builder,
                "route_after_reflection",
                wraps=graph_builder.route_after_reflection,
            ) as route,
            self.model_responses(responses) as calls,
            redirect_stdout(io.StringIO()),
        ):
            repository = Path(temporary_repository).resolve()
            (repository / "README.md").write_text(
                "Synthetic component.\n", encoding="utf-8"
            )
            events = []
            with self.assertRaisesRegex(RuntimeError, "Synthetic provider failure"):
                for output in graph_builder.build_graph().stream(
                    {
                        "repository_path": str(repository),
                        "programming_language": "Rust",
                    },
                    stream_mode="updates",
                ):
                    events.extend(output)
        self.assertEqual(len(calls), 5)
        build.assert_called_once()
        improve.assert_not_called()
        fill.assert_called_once()
        route.assert_not_called()
        self.assertEqual(familiariser.invoke.call_count, 2)
        self.assertEqual(
            events,
            ["fetch_tree", "read_threatmodel", "familiarise", "fill_threat_model"],
        )

    def test_third_reflect_visit_skips_agent_creation_and_all_model_attempts(self):
        with (
            patch.object(graph_builder, "MAX_ITERATIONS", 3),
            patch.object(reflection_chain, "create_agent") as create_agent,
            patch.object(reflection_chain, "reflection_chain") as legacy_chain,
            self.model_responses([]) as calls,
        ):
            updates = graph_builder.reflection_node({"iteration_count": 2})
        create_agent.assert_not_called()
        legacy_chain.invoke.assert_not_called()
        self.assertEqual(calls, [])
        self.assertEqual(updates["iteration_count"], 3)
        self.assertEqual(updates["stop_reason"], "max_iterations")
        self.assertIsNone(updates["needs_revision"])
        self.assertIsNone(updates["reflection"])
        self.assertEqual(
            graph_builder.route_after_reflection(updates), graph_builder.END
        )


if __name__ == "__main__":
    unittest.main()
