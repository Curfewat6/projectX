import os
import datetime
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


# class AnswerQuestion(BaseModel):
#     """Answer the question. Follow this structure"""

#     answer: str = Field(description="~250 word detailed answer to the question")
#     reflection: Reflection = Field(description="Your reflection to the initial answer")
#     search_queries: List[str] = Field(
#         description="1-3 search queries for researching improvements to address the critique of your current answer"
#     )

class AnswerQuestion(BaseModel):
    """Answer the question. Follow this structure"""
    answer: str = Field(description="~250 word detailed answer to the question")

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Follow this structure"""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )


class _ThreatModelSection(BaseModel):
    """A template section with explicit fields and no unrecognized keys.

    Record unknown prose values as 'Unknown' or 'Not assessed'. Empty lists
    mean no recorded items, not proof that no relevant items exist.
    """

    model_config = ConfigDict(extra="forbid")


class ThreatModelStatus(_ThreatModelSection):
    """Target identity and current review status."""

    target: str = Field(description="Target name, also used in the report title.")
    version_commit_or_build: str | None = Field(
        description="Reviewed version, commit, or build; null if unknown."
    )
    lab_state: str = Field(description="Current state of the review environment.")
    current_decision: Literal["exploring", "validating", "reporting", "parked"] = Field(
        description="Current review decision."
    )


class AttackerModel(_ThreatModelSection):
    """Actor capabilities, prerequisites, and scope assumptions."""

    actor: str = Field(description="Actor considered by this threat model.")
    access_level: str = Field(description="Actor's assumed initial access level.")
    inputs_controlled: List[str] = Field(description="Inputs the actor controls.")
    inputs_not_controlled: List[str] = Field(
        description="Inputs outside the actor's control."
    )
    required_prerequisites: List[str] = Field(
        description="Prerequisites required by the stated actor model."
    )
    explicitly_out_of_scope: List[str] = Field(
        description="Actors, capabilities, or scenarios explicitly excluded."
    )


class CrownJewel(_ThreatModelSection):
    """One security-sensitive asset or behavior."""

    security_sensitive_asset_or_behavior: str = Field(
        description="Asset or behavior requiring protection."
    )
    why_compromise_matters: str = Field(
        description="Impact of compromising this asset or behavior."
    )
    expected_security_boundary: str = Field(
        description="Security boundary expected to protect it."
    )


class EntryPoints(_ThreatModelSection):
    """Entry points grouped by the categories in the template."""

    public_or_low_privilege_entry_points: List[str] = Field(
        description="Public or low-privilege entry points recorded in the review."
    )
    internal_attacker_reachable_transitions: List[str] = Field(
        description="Internal transitions reachable under the stated actor model."
    )
    configuration_dependent_surfaces: List[str] = Field(
        description="Surfaces whose availability depends on configuration."
    )


class TrustBoundary(_ThreatModelSection):
    """One trust boundary and its expected enforcing control."""

    boundary: str = Field(description="Name or description of the boundary.")
    trusted_side: str = Field(description="Components or data on the trusted side.")
    untrusted_side: str = Field(description="Components or data on the untrusted side.")
    enforcing_guard: str = Field(
        description="Guard that is supposed to enforce the boundary."
    )


class HighRiskOperations(_ThreatModelSection):
    """Recorded operations grouped by the template's five categories."""

    parsing_decoding_or_deserialization: List[str] = Field(
        description="Relevant parsing, decoding, or deserialization operations."
    )
    authn_authz_session_token_logic: List[str] = Field(
        description="Relevant authentication, authorization, session, or token logic."
    )
    file_archive_update_plugin_or_template_handling: List[str] = Field(
        description="Relevant file, archive, update, plugin, or template handling."
    )
    native_kernel_ipc_sandbox_or_privilege_boundary: List[str] = Field(
        description="Relevant native, kernel, IPC, sandbox, or privilege boundaries."
    )
    caching_tenant_isolation_or_cross_user_state: List[str] = Field(
        description="Relevant caching, tenant isolation, or cross-user state."
    )


class Invariant(_ThreatModelSection):
    """A security property and the context requested below the numbered slots."""

    statement: str = Field(description="Security property that must remain true.")
    attacker_controlled_input_or_state_transition: str | None = Field(
        description="Relevant controlled input or state transition; null if unknown."
    )
    observable_violation: str | None = Field(
        description="Where a violation would be observable; null if unknown."
    )


class ThinSlice(_ThreatModelSection):
    """One focused review item recorded in First Thin Slices."""

    slice: str = Field(description="Name or scope of the focused review item.")
    entry_point: str = Field(description="Entry point associated with this item.")
    sensitive_sink: str = Field(
        description="Sensitive operation or destination associated with this item."
    )
    invariant_under_test: str = Field(description="Security invariant being assessed.")
    expected_verifier: str = Field(
        description="Recorded verification method, or an explicit unresolved status."
    )


class VariantAnalysis(_ThreatModelSection):
    """Recorded coverage and outcomes of related-issue review."""

    seed_advisory_issue_commit_patch_or_invariant: str | None = Field(
        description="Reference that motivated the review; null if none is recorded."
    )
    sibling_pass: List[str] = Field(
        description="Adjacent APIs, callers, versions, and related paths checked."
    )
    incomplete_fix_pass: List[str] = Field(
        description="Sources, sinks, states, configurations, or input forms checked."
    )
    bypass_regression_pass: List[str] = Field(
        description="Later conversions, mutations, caching, or refactors checked."
    )
    negative_results_and_surfaces_ruled_out: List[str] = Field(
        description="Recorded negative results and surfaces ruled out."
    )
    variants_promoted_for_verification: List[str] = Field(
        description="References to review items awaiting verification."
    )


class VerifierPlan(_ThreatModelSection):
    """Recorded verification approaches; null denotes an unestablished approach."""

    static_proof: str | None = Field(
        description="Recorded static-evidence approach, if established."
    )
    local_model_or_unit_test: str | None = Field(
        description="Recorded local model or unit-test approach, if established."
    )
    integration_or_harness: str | None = Field(
        description="Recorded integration or harness approach, if established."
    )
    disposable_vm_or_crash_only_step: str | None = Field(
        description="Recorded isolated-environment verification approach, if applicable."
    )
    negative_control: str | None = Field(
        description="Recorded negative control, if established."
    )


class NoveltyGate(_ThreatModelSection):
    """Prior-art review record; no advisory match alone does not establish novelty."""

    current_advisories_cves_ghsas_checked: List[str] = Field(
        description="References to current advisories, CVEs, or GHSAs checked."
    )
    historical_advisories_and_release_notes_checked: List[str] = Field(
        description="Historical advisories and release notes checked."
    )
    vendor_upstream_issues_commits_changelogs_and_patches_checked: List[str] = Field(
        description="Vendor or upstream issues, commits, changelogs, and patches checked."
    )
    subsystem_function_names_bug_class_terms_and_expressions_searched: List[str] = (
        Field(
            description="Subsystem/function names, bug-class terms, and old/fixed expressions searched."
        )
    )
    search_date: datetime.date | None = Field(
        description="Actual search date in YYYY-MM-DD format; null if unknown or not searched."
    )
    sources_not_yet_covered: List[str] = Field(
        description="Relevant sources not yet covered by the recorded search."
    )
    candidate_versus_known_issue: str = Field(
        description=(
            "Comparison covering actor, entry point, boundary, root cause, primitive, "
            "affected versions, and expected fix. Record unknown dimensions explicitly."
        )
    )
    classification: Literal[
        "exact duplicate",
        "sibling variant",
        "incomplete fix",
        "fix bypass",
        "regression",
        "independent root cause",
        "unresolved",
    ] = Field(description="Current classification based on the recorded evidence.")
    novelty_confidence_and_remaining_collision_risk: str = Field(
        description="Confidence assessment and remaining risk of overlap with a known issue."
    )


class DecisionLogEntry(_ThreatModelSection):
    """One dated observation, decision, and next action."""

    date: datetime.date | None = Field(
        description="Actual decision date in YYYY-MM-DD format; null if unknown."
    )
    observation: str = Field(description="Observation motivating the decision.")
    decision: str = Field(description="Decision that was made.")
    next_action: str = Field(
        description="Recorded next action or unresolved follow-up."
    )


class ThreatModel(BaseModel):
    """Structured report matching THREAT_MODE_TEMPLATE.md.

    Include all twelve sections. Keep prose concise and mark unknown or
    unassessed information explicitly instead of inventing facts. Numbered
    template examples do not impose fixed lengths on the corresponding lists.
    """

    model_config = ConfigDict(extra="forbid")

    status: ThreatModelStatus = Field(description="Status.")
    attacker_model: AttackerModel = Field(description="Attacker Model.")
    crown_jewels: List[CrownJewel] = Field(description="Crown Jewels.")
    entry_points: EntryPoints = Field(description="Entry Points.")
    trust_boundaries: List[TrustBoundary] = Field(description="Trust Boundaries.")
    high_risk_operations: HighRiskOperations = Field(
        description="High-Risk Operations."
    )
    invariants: List[Invariant] = Field(description="Invariants.")
    first_thin_slices: List[ThinSlice] = Field(description="First Thin Slices.")
    variant_analysis: VariantAnalysis = Field(description="Variant Analysis.")
    verifier_plan: VerifierPlan = Field(description="Verifier Plan.")
    novelty_gate: NoveltyGate = Field(description="Novelty Gate.")
    decision_log: List[DecisionLogEntry] = Field(description="Decision Log.")
    references: List[str] = Field(
        default_factory=list,
        description="Supporting source paths, document references, or URLs.",
    )

if __name__ == "__main__":
    print(f"[INFO] NOT RUNNING Main.py.\n[Info] Now running {os.path.basename(__file__)}")