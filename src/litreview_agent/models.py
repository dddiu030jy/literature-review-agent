from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QueryVariant:
    query: str
    language: str
    facet: str
    rationale: str
    round: int = 0


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    venue: str = ""
    sources: list[str] = field(default_factory=list)
    source_ids: dict[str, str] = field(default_factory=dict)
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    open_access_pdf: str = ""
    citation_count: int = 0
    influential_citation_count: int = 0
    publication_type: str = ""
    language: str = ""
    is_retracted: bool | None = None
    retraction_status: str = "unchecked"
    version_group_id: str = ""
    version_status: str = "unknown"
    metadata_conflicts: list[str] = field(default_factory=list)
    matched_queries: list[str] = field(default_factory=list)
    retrieval_round: int = 0
    relevance_score: float = 0.0
    screening_decision: str = "unreviewed"
    screening_reason: str = ""
    coverage_tags: list[str] = field(default_factory=list)
    preferred_venue_match: bool = False
    extraction_basis: str = "abstract_only"
    full_text_excerpt: str = ""
    full_text_sections: dict[str, str] = field(default_factory=dict)
    pdf_page_count: int = 0
    pdf_pages_extracted: int = 0
    pdf_parse_quality: str = "not_attempted"
    source_warning_flags: list[str] = field(default_factory=list)
    research_question: str = ""
    methods: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    population_sample: str = ""
    geography_context: str = ""
    data_period: str = ""
    study_design: str = ""
    variables_operationalization: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    validation_setting: str = ""
    effect_estimates: list[str] = field(default_factory=list)
    causal_identification: str = ""
    boundary_conditions: list[str] = field(default_factory=list)
    author_conclusions: list[str] = field(default_factory=list)
    analyst_inferences: list[str] = field(default_factory=list)
    rhetorical_role: str = ""
    concept_level: str = ""
    evidence_stages: list[str] = field(default_factory=list)
    change_representation: str = ""
    mechanism: str = ""
    temporal_design: str = ""
    state_output: str = ""
    real_time_vintage: str = ""
    out_of_sample: str = ""
    uncertainty: list[str] = field(default_factory=list)
    costs_constraints: list[str] = field(default_factory=list)
    cross_setting_test: str = ""
    evidence_spans: list[dict[str, Any]] = field(default_factory=list)
    topic_value: str = ""
    theme_ids: list[str] = field(default_factory=list)
    citation_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Paper":
        allowed = cls.__dataclass_fields__.keys()
        normalized = {key: value[key] for key in allowed if key in value}
        legacy_basis = normalized.get("extraction_basis")
        if legacy_basis == "abstract":
            normalized["extraction_basis"] = "abstract_only"
        elif legacy_basis == "full_text":
            normalized["extraction_basis"] = "partial_fulltext"
        return cls(**normalized)


@dataclass
class Theme:
    theme_id: str
    name: str
    organizing_question: str
    synthesis_claim: str
    paper_ids: list[str]
    classification_basis: str
    internal_relations: list[str]
    inherited_question: str = ""
    residual_gap: str = ""
    next_question: str = ""
    relation_to_previous: str = "independent"
    relation_evidence_paper_ids: list[str] = field(default_factory=list)
    relation_support_status: str = "not_verified"
    publication_heading: str = ""
    heading_function: str = ""
    heading_relation_rationale: str = ""
    parallel_group: str = ""
    heading_quality_status: str = "not_checked"


@dataclass
class GapAudit:
    adequate: bool
    coverage_summary: str
    weak_themes: list[dict[str, Any]]
    supplemental_queries: list[QueryVariant]
    rationale: str
    round_index: int = 1
    stop_recommended: bool = False
    stop_reason: str = ""
    draft_gap_summary: str = ""
    missing_evidence_stages: list[str] = field(default_factory=list)
    unsupported_transition_claims: list[str] = field(default_factory=list)


@dataclass
class RunIntegrityReport:
    candidate_count: int
    selected_count: int
    theme_count: int
    supplemental_candidate_count: int
    supplemental_selected_count: int
    rejected_unknown_citation_sentences: list[str]
    uncited_selected_papers: list[str]
    warnings: list[str]
    checks: dict[str, bool]


@dataclass
class ClaimEvidence:
    claim_id: str
    claim_text: str
    supporting_paper_ids: list[str]
    supporting_evidence_spans: list[dict[str, Any]]
    evidence_page_or_section: list[str]
    support_type: str
    confidence: float
    semantic_verification_status: str = "not_human_verified"


@dataclass
class ContentQualityAudit:
    claim_count: int
    claims_with_citations: int
    claims_with_evidence_spans: int
    claims_requiring_human_verification: int
    uncited_factual_sentences: list[str]
    abstract_only_claims: int
    partial_text_claims: int
    human_screening_evaluation_complete: bool
    extraction_fact_check_complete: bool
    claim_evidence_human_audit_complete: bool
    warnings: list[str]
