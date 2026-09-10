"""Offline coverage of reversible ThreatModel correction without an attempt cap."""

import copy
import io
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

# This helper disables tracing and patches dotenv before importing the project.
from test_linear_graph import (
    AIMessage,
    ChatGeneration,
    ChatOllama,
    ChatOpenRouter,
    ChatResult,
    HumanMessage,
    ToolMessage,
    ValidationError,
    answer_message,
    graph_builder,
    llm,
    minimal_report,
    read_message,
    reflection_decision,
    threat_model_chain,
)


def report_message(*, valid=True, call_id="report-1"):
    arguments = minimal_report().model_dump(mode="json")
    if not valid:
        arguments["attacker_model"]["inputs_controlled"] = "Must be a list instead."
    return AIMessage(
        content="",
        tool_calls=[{"name": "ThreatModel", "args": arguments, "id": call_id}],
    )


class SchemaCorrectionTests(unittest.TestCase):
    def setUp(self):
        for current_patch in (
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("Offline tests must not contact a network."),
            ),
            patch.object(threat_model_chain, "USE_SCHEMA_CORRECTION", True),
        ):
            current_patch.start()
            self.addCleanup(current_patch.stop)
        self.context = (
            "source-1 README.md:1: Synthetic source excerpt; no runtime checks."
        )

    @contextmanager
    def model_responses(self, responses):
        """Keep the real agent and validators, replacing only model transport."""
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
                len(calls), len(responses), "Unexpected extra model call"
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

    def test_valid_first_response_returns_validated_report_without_legacy_chain(self):
        with (
            self.model_responses([report_message()]) as calls,
            patch.object(threat_model_chain, "threat_model_chain") as legacy_chain,
        ):
            report = threat_model_chain.generate_threat_model(self.context, "Rust")
        self.assertEqual(report, minimal_report())
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [tool["function"]["name"] for tool in calls[0]["tools"]], ["ThreatModel"]
        )
        self.assertIn("Rust", calls[0]["messages"][0].content)
        self.assertIn(self.context, calls[0]["messages"][-1].content)
        legacy_chain.invoke.assert_not_called()

    def test_schema_error_is_returned_to_model_with_original_prompt_and_evidence(self):
        invalid = report_message(valid=False, call_id="invalid-report")
        with self.model_responses(
            [invalid, report_message(call_id="fixed-report")]
        ) as calls:
            report = threat_model_chain.generate_threat_model(self.context, "Rust")
        self.assertEqual(report, minimal_report())
        self.assertEqual(len(calls), 2)
        first_messages = calls[0]["messages"]
        correction_messages = calls[1]["messages"]
        self.assertEqual(correction_messages[: len(first_messages)], first_messages)
        self.assertEqual(
            len(
                [
                    message
                    for message in correction_messages
                    if isinstance(message, HumanMessage)
                ]
            ),
            1,
        )
        prior_output = [
            message for message in correction_messages if isinstance(message, AIMessage)
        ]
        self.assertEqual(prior_output[0].tool_calls, invalid.tool_calls)
        feedback = [
            message
            for message in correction_messages
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0].tool_call_id, "invalid-report")
        self.assertIn("inputs_controlled", feedback[0].content)
        self.assertIn("list", feedback[0].content.lower())
        self.assertIn(self.context, correction_messages[1].content)

    def test_invalid_output_can_continue_past_three_attempts_and_succeed(self):
        responses = [
            report_message(valid=attempt == 4, call_id=f"report-attempt-{attempt}")
            for attempt in range(5)
        ]
        with self.model_responses(responses) as calls:
            report = threat_model_chain.generate_threat_model(self.context, "Rust")
        self.assertEqual(report, minimal_report())
        self.assertEqual(len(calls), 5)
        feedback = [
            message
            for message in calls[-1]["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(len(feedback), 4)
        self.assertTrue(
            all("inputs_controlled" in message.content for message in feedback)
        )
        self.assertIn(self.context, calls[-1]["messages"][1].content)

    def test_provider_failure_propagates_without_becoming_schema_feedback(self):
        provider_error = RuntimeError("Synthetic provider stream failed.")
        responses = [
            report_message(valid=False, call_id=f"invalid-{n}") for n in range(4)
        ] + [provider_error]
        with (
            self.model_responses(responses) as calls,
            patch.object(threat_model_chain, "threat_model_chain") as legacy_chain,
        ):
            with self.assertRaises(RuntimeError) as failure:
                threat_model_chain.generate_threat_model(self.context, "Rust")
        self.assertEqual(len(calls), 5)
        self.assertIs(failure.exception, provider_error)
        legacy_chain.invoke.assert_not_called()

    def test_missing_schema_tool_can_continue_past_three_attempts_and_succeed(self):
        responses = [
            AIMessage(content="No structured report was provided.") for _ in range(4)
        ] + [report_message()]
        with self.model_responses(responses) as calls:
            report = threat_model_chain.generate_threat_model(self.context, "Rust")
        self.assertEqual(report, minimal_report())
        self.assertEqual(len(calls), 5)
        self.assertIn(self.context, calls[-1]["messages"][1].content)

    def test_correction_agent_has_no_model_call_limit_middleware(self):
        with (
            patch.object(
                threat_model_chain,
                "create_agent",
                wraps=threat_model_chain.create_agent,
            ) as create_agent,
            self.model_responses([report_message()]) as calls,
        ):
            report = threat_model_chain.generate_threat_model(self.context, "Rust")
        self.assertEqual(report, minimal_report())
        self.assertEqual(len(calls), 1)
        create_agent.assert_called_once()
        self.assertNotIn("middleware", create_agent.call_args.kwargs)

    def test_each_report_generation_has_a_fresh_correction_conversation(self):
        responses = [
            report_message(
                valid=attempt == 4, call_id=f"report-{report}-attempt-{attempt}"
            )
            for report in range(2)
            for attempt in range(5)
        ]
        with self.model_responses(responses) as calls:
            first = threat_model_chain.generate_threat_model(
                "First report evidence.", "Rust"
            )
            second = threat_model_chain.generate_threat_model(
                "Second report evidence.", "Rust"
            )
        self.assertEqual(first, minimal_report())
        self.assertEqual(second, minimal_report())
        self.assertEqual(len(calls), 10)
        second_start = calls[5]["messages"]
        self.assertFalse(
            any(
                isinstance(message, (AIMessage, ToolMessage))
                for message in second_start
            )
        )
        self.assertIn("Second report evidence.", second_start[-1].content)
        self.assertNotIn("First report evidence.", second_start[-1].content)

    def test_disabled_correction_uses_original_single_call_validation(self):
        with (
            patch.object(threat_model_chain, "USE_SCHEMA_CORRECTION", False),
            patch.object(threat_model_chain, "create_agent") as create_agent,
            self.model_responses([report_message(valid=False)]) as calls,
            self.assertRaises(ValidationError),
        ):
            threat_model_chain.generate_threat_model(self.context, "Rust")
        self.assertEqual(len(calls), 1)
        create_agent.assert_not_called()

    def test_disabled_correction_still_returns_valid_reports(self):
        with (
            patch.object(threat_model_chain, "USE_SCHEMA_CORRECTION", False),
            patch.object(threat_model_chain, "create_agent") as create_agent,
            self.model_responses([report_message()]) as calls,
        ):
            report = threat_model_chain.generate_threat_model(self.context, "Rust")
        self.assertEqual(report, minimal_report())
        self.assertEqual(len(calls), 1)
        create_agent.assert_not_called()

    def test_invalid_output_on_both_branches_never_returns_unvalidated_dict(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                with patch.object(threat_model_chain, "USE_SCHEMA_CORRECTION", enabled):
                    if enabled:
                        backend_patch = patch.object(
                            threat_model_chain,
                            "create_agent",
                            return_value=Mock(
                                invoke=Mock(
                                    return_value={"structured_response": {"status": {}}}
                                )
                            ),
                        )
                    else:
                        backend_patch = patch.object(
                            threat_model_chain,
                            "threat_model_chain",
                            Mock(
                                invoke=Mock(
                                    return_value={
                                        "parsed": {"status": {}},
                                        "parsing_error": None,
                                    }
                                )
                            ),
                        )
                    with backend_patch, self.assertRaises(RuntimeError):
                        threat_model_chain.generate_threat_model(self.context, "Rust")

    def test_repeated_corrections_do_not_advance_outer_revision_counters(self):
        responses = [
            report_message(
                valid=attempt == 4, call_id=f"revision-{revision}-attempt-{attempt}"
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
        improvements = "Corrected report notes using the existing source excerpt."
        improver = Mock(invoke=Mock(return_value=answer_message(improvements)))
        with (
            tempfile.TemporaryDirectory() as temporary_repository,
            patch.object(
                graph_builder,
                "read_threatmodel_chain",
                Mock(invoke=Mock(return_value=AIMessage(content="Read README.md."))),
            ),
            patch.object(
                graph_builder, "build_familiariser", return_value=familiariser
            ) as build_familiariser,
            patch.object(
                graph_builder, "build_threat_model_improver", return_value=improver
            ) as build_improver,
            patch.object(
                graph_builder,
                "reflect_threat_model",
                side_effect=[reflection_decision(True), reflection_decision(False)],
            ) as critic,
            self.model_responses(responses) as calls,
            redirect_stdout(io.StringIO()),
        ):
            repository = Path(temporary_repository).resolve()
            (repository / "README.md").write_text(
                "Synthetic application component.\n", encoding="utf-8"
            )
            graph = graph_builder.build_graph()
            result = graph.invoke(
                {"repository_path": str(repository), "programming_language": "Rust"}
            )
        self.assertEqual(len(calls), 10)
        build_familiariser.assert_called_once()
        build_improver.assert_called_once()
        self.assertEqual(familiariser.invoke.call_count, 2)
        improver.invoke.assert_called_once()
        self.assertEqual(critic.call_count, 2)
        self.assertEqual(result["iteration_count"], 2)
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["stop_reason"], "approved")
        self.assertEqual(result["threat_model"], minimal_report())
        self.assertEqual(
            result["familiarisation"], "Synthetic overview from src/app.py."
        )
        self.assertEqual(result["improvements"], improvements)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(len(result["history"]), 2)
        self.assertIn("source-1", calls[5]["messages"][-1].content)
        self.assertIn(improvements, calls[5]["messages"][-1].content)
        self.assertIn(
            "Inspect component responsibilities", calls[5]["messages"][-1].content
        )
        self.assertNotIn("correction", graph.get_graph().nodes)


if __name__ == "__main__":
    unittest.main()
