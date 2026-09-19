from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .io_utils import write_csv, write_json
from .llm import OpenAICompatibleClient
from .evaluation import create_human_evaluation_pack
from .models import ContentQualityAudit, Paper, RunIntegrityReport, Theme
from .prompts import (
    ARGUMENT_AUDIT_PROMPT_VERSION,
    EXTRACT_PROMPT_VERSION,
    QUERY_PROMPT_VERSION,
    REFLECT_PROMPT_VERSION,
    REVISION_PROMPT_VERSION,
    SCREEN_PROMPT_VERSION,
    THEME_PROMPT_VERSION,
    WRITE_PROMPT_VERSION,
)
from .reasoning import FullTextFetcher, Reasoner
from .search import SearchManager, merge_papers
from .writing import (
    ReviewWriter,
    heading_architecture_report,
    citation_counts,
    publication_quality_report,
    render_bibtex,
)


PAPER_FIELDS = [
    "paper_id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "sources",
    "citation_count",
    "publication_type",
    "language",
    "is_retracted",
    "retraction_status",
    "version_group_id",
    "version_status",
    "matched_queries",
    "retrieval_round",
    "relevance_score",
    "screening_decision",
    "screening_reason",
    "coverage_tags",
    "preferred_venue_match",
    "abstract",
]

EXTRACTION_FIELDS = [
    "paper_id",
    "title",
    "extraction_basis",
    "research_question",
    "methods",
    "findings",
    "limitations",
    "population_sample",
    "geography_context",
    "data_period",
    "study_design",
    "variables_operationalization",
    "baselines",
    "metrics",
    "validation_setting",
    "effect_estimates",
    "causal_identification",
    "boundary_conditions",
    "author_conclusions",
    "analyst_inferences",
    "rhetorical_role",
    "concept_level",
    "evidence_stages",
    "change_representation",
    "mechanism",
    "temporal_design",
    "state_output",
    "real_time_vintage",
    "out_of_sample",
    "uncertainty",
    "costs_constraints",
    "cross_setting_test",
    "evidence_spans",
    "topic_value",
    "theme_ids",
    "coverage_tags",
    "preferred_venue_match",
]


class LiteratureReviewPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.out = config.output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.logger = _build_logger(self.out / "run.log")
        self.llm = OpenAICompatibleClient(
            mode=config.llm_mode,
            model=config.llm_model,
            base_url=config.llm_base_url,
            timeout=config.request_timeout,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            retries=config.request_retries,
        )
        self.reasoner = Reasoner(config, self.llm)
        self.search = SearchManager(config)
        self.writer = ReviewWriter(config, self.llm)
        self.warnings: list[str] = []
        self.source_errors: dict[str, str] = {}
        self.stage_timings: dict[str, float] = {}

    @contextmanager
    def _stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stage_timings[name] = round(
                self.stage_timings.get(name, 0.0) + time.perf_counter() - started,
                4,
            )

    def _generate_and_revise_review(
        self, final_selected: list[Paper], final_themes: list[Theme]
    ) -> tuple[str, list[str], list[Any], dict[str, Any], list[dict[str, Any]]]:
        """Generate the final review, feed failed audits back, and re-audit."""
        with self._stage("review_generation"):
            review, rejected, claims = self.writer.write(
                final_selected, final_themes
            )
        (self.out / "11_review_initial.md").write_text(review, encoding="utf-8")
        audit = self.writer.audit_argument(review, final_selected, final_themes)
        write_json(self.out / "15_argument_coherence_audit_round_0.json", audit)
        revision_rounds: list[dict[str, Any]] = []
        for revision_round in range(1, self.config.max_writing_revision_rounds + 1):
            if not self.llm.enabled or not audit.get("rewrite_required", False):
                break
            before_hash = hashlib.sha256(review.encode("utf-8")).hexdigest()
            with self._stage(f"review_revision_round_{revision_round}"):
                revised, revision_rejected, revised_claims = self.writer.revise(
                    review,
                    final_selected,
                    final_themes,
                    audit,
                    round_index=revision_round,
                )
            rejected.extend(revision_rejected)
            changed = _normalized_review_argument(revised) != _normalized_review_argument(
                review
            )
            review = revised
            claims = revised_claims
            (self.out / f"11_review_revision_round_{revision_round}.md").write_text(
                review, encoding="utf-8"
            )
            audit = self.writer.audit_argument(review, final_selected, final_themes)
            write_json(
                self.out / f"15_argument_coherence_audit_round_{revision_round}.json",
                audit,
            )
            revision_rounds.append(
                {
                    "round": revision_round,
                    "input_sha256": before_hash,
                    "output_sha256": hashlib.sha256(review.encode("utf-8")).hexdigest(),
                    "review_changed": changed,
                    "rejected_unknown_citation_sentence_count": len(revision_rejected),
                    "automatic_checks_pass": audit.get("automatic_checks_pass", False),
                    "rewrite_required_after_round": audit.get("rewrite_required", True),
                    "needs_human_review": audit.get("needs_human_review", True),
                }
            )
            if not changed:
                self.warnings.append(
                    f"正式写作第{revision_round}轮审计重写未改变正文，停止继续重写。"
                )
                break
        write_json(
            self.out / "15_writing_revision_report.json",
            {
                "formal_llm_enabled": self.llm.enabled,
                "configured_max_rounds": self.config.max_writing_revision_rounds,
                "executed_rounds": len(revision_rounds),
                "rounds": revision_rounds,
                "final_automatic_checks_pass": audit.get("automatic_checks_pass", False),
                "final_rewrite_required": audit.get("rewrite_required", True),
                "final_needs_human_review": audit.get("needs_human_review", True),
                "status": (
                    "automatic_checks_passed_human_review_pending"
                    if audit.get("automatic_checks_pass", False)
                    else "revision_candidate"
                ),
                "publication_boundary": (
                    "Revision feedback and round artifacts are backstage audit records."
                ),
            },
        )
        return review, rejected, claims, audit, revision_rounds

    def run(self) -> dict[str, Any]:
        started = dt.datetime.now(dt.timezone.utc)
        self.logger.info("run started topic=%s", self.config.topic)
        write_json(self.out / "00_config.json", self.config.to_dict())

        with self._stage("query_generation"):
            queries = self.reasoner.generate_queries()
        write_json(self.out / "01_queries.json", [asdict(query) for query in queries])
        self.logger.info("generated %d query variants", len(queries))

        with self._stage("initial_retrieval"):
            candidates, errors = self.search.search(queries, round_index=0)
        self.source_errors.update(errors)
        if candidates and not self.config.offline_corpus:
            seeds = sorted(
                candidates,
                key=lambda paper: (paper.relevance_score, paper.citation_count),
                reverse=True,
            )
            with self._stage("citation_chasing"):
                chased, chase_errors = self.search.citation_chase(seeds, round_index=0)
            self.source_errors.update(chase_errors)
            candidates = merge_papers(chased, existing=candidates)
        if len(candidates) < self.config.min_candidates and not self.config.offline_corpus:
            top_up = self.reasoner._heuristic_queries()[-2:]
            for query in top_up:
                query.round = 0
            extra, extra_errors = self.search.search(top_up, round_index=0)
            self.source_errors.update(extra_errors)
            candidates = merge_papers(extra, existing=candidates)
        initial_candidate_count = len(candidates)
        if initial_candidate_count < self.config.min_candidates:
            self.warnings.append(
                f"候选论文仅{initial_candidate_count}篇，低于目标"
                f"{self.config.min_candidates}篇；可能是主题过窄或数据源失败。"
            )
        if initial_candidate_count < 3:
            write_json(
                self.out / "run_failure.json",
                {
                    "stage": "retrieval",
                    "candidate_count": initial_candidate_count,
                    "required_for_screening": 3,
                    "source_errors": self.source_errors,
                    "message": "候选论文不足，无法满足最少筛选数量；请扩大主题、年份或数据源。",
                },
            )
            raise RuntimeError(
                f"Only {initial_candidate_count} candidates were retrieved; "
                "at least 3 are required."
            )
        self.logger.info("retrieved %d deduplicated candidates", len(candidates))

        with self._stage("screening_initial"):
            selected_initial = self.reasoner.screen(candidates)
        if len(selected_initial) < self.config.min_selected:
            self.warnings.append(
                f"仅{len(selected_initial)}篇达到相关性阈值{self.config.relevance_threshold}；"
                "未用低相关论文强行补足15篇，数量门记为未通过。"
            )
        if not selected_initial:
            write_json(
                self.out / "run_failure.json",
                {
                    "stage": "screening",
                    "threshold": self.config.relevance_threshold,
                    "candidate_count": initial_candidate_count,
                    "message": "没有论文达到相关性阈值；系统拒绝用低相关论文凑数。请改进检索式或用人工金标准校准阈值。",
                },
            )
            raise RuntimeError(
                "No paper met the relevance threshold; quantity backfill is disabled."
            )
        write_json(
            self.out / "02_candidates.json",
            [paper.to_dict() for paper in candidates],
        )
        write_csv(
            self.out / "02_candidates.csv",
            (paper.to_dict() for paper in candidates),
            PAPER_FIELDS,
        )
        write_json(
            self.out / "03_screened.json",
            [paper.to_dict() for paper in selected_initial],
        )
        write_json(
            self.out / "03_coverage_initial.json",
            self.reasoner.coverage_snapshot(selected_initial),
        )
        self.logger.info("screened %d initial papers", len(selected_initial))

        full_text = FullTextFetcher(self.config, self.search.http)
        with self._stage("fulltext_initial"):
            self.warnings.extend(full_text.enrich(selected_initial))
        with self._stage("extraction_initial"):
            self.reasoner.extract(selected_initial)
        write_json(
            self.out / "04_extractions_initial.json",
            [paper.to_dict() for paper in selected_initial],
        )

        with self._stage("theme_organization_initial"):
            themes_initial = self.reasoner.organize_themes(selected_initial)
        write_json(
            self.out / "05_themes_initial.json",
            [asdict(theme) for theme in themes_initial],
        )
        current_selected = selected_initial[:]
        current_themes = themes_initial
        all_new_supplemental: list[Paper] = []
        all_supplemental_selected: list[Paper] = []
        supplemental_rounds: list[dict[str, Any]] = []
        existing_keys = {_paper_key(paper) for paper in candidates}
        initial_stage_counts = _evidence_stage_counts(current_selected)
        current_draft_audit: dict[str, Any] = {}
        provisional_review = ""
        if self.config.provisional_review_before_reflection:
            with self._stage("provisional_review_generation"):
                provisional_review, provisional_rejected, _ = self.writer.write(
                    current_selected, current_themes
                )
            (self.out / "05_review_provisional.md").write_text(
                provisional_review, encoding="utf-8"
            )
            current_draft_audit = build_draft_gap_audit(
                self.config.topic,
                provisional_review,
                current_selected,
                current_themes,
            )
            current_draft_audit["rejected_unknown_citation_sentences"] = provisional_rejected
            write_json(self.out / "05_draft_gap_audit.json", current_draft_audit)
            provisional_argument_audit = self.writer.audit_argument(
                provisional_review, current_selected, current_themes
            )
            write_json(
                self.out / "05_argument_coherence_audit.json",
                provisional_argument_audit,
            )
        for round_index in range(1, self.config.max_reflection_rounds + 1):
            with self._stage(f"coverage_reflection_round_{round_index}"):
                audit = self.reasoner.reflect(
                    current_themes,
                    current_selected,
                    draft_audit=current_draft_audit,
                    round_index=round_index,
                )
            if round_index == 1:
                write_json(self.out / "06_gap_audit.json", asdict(audit))
                write_json(
                    self.out / "07_supplemental_queries.json",
                    [asdict(query) for query in audit.supplemental_queries],
                )
            if audit.stop_recommended and round_index > self.config.min_reflection_rounds:
                supplemental_rounds.append({
                    "round": round_index,
                    "audit": asdict(audit),
                    "stopped_before_search": True,
                })
                break
            if not audit.supplemental_queries:
                supplemental_rounds.append({
                    "round": round_index,
                    "audit": asdict(audit),
                    "stopped_before_search": True,
                })
                break
            with self._stage(f"supplemental_retrieval_round_{round_index}"):
                supplemental, supplemental_errors = self.search.search(
                    audit.supplemental_queries, round_index=round_index
                )
            self.source_errors.update(supplemental_errors)
            new_supplemental = [
                paper for paper in supplemental if _paper_key(paper) not in existing_keys
            ]
            for paper in new_supplemental:
                existing_keys.add(_paper_key(paper))
            all_new_supplemental.extend(new_supplemental)
            with self._stage(f"supplemental_screening_round_{round_index}"):
                screened_supplemental = self.reasoner.screen(new_supplemental)
            supplemental_selected = screened_supplemental[:6]
            if supplemental_selected:
                with self._stage(f"supplemental_fulltext_round_{round_index}"):
                    self.warnings.extend(full_text.enrich(supplemental_selected))
                with self._stage(f"supplemental_extraction_round_{round_index}"):
                    self.reasoner.extract(supplemental_selected)
                all_supplemental_selected.extend(supplemental_selected)
            unique_gain = len(new_supplemental) / max(1, len(candidates) + len(all_new_supplemental))
            before_count = len(current_selected)
            current_selected = merge_papers(supplemental_selected, existing=current_selected)
            selected_gain = len(current_selected) - before_count
            with self._stage(f"theme_reorganization_round_{round_index}"):
                updated_themes = self.reasoner.organize_themes(current_selected)
            theme_signature_changed = _theme_signature(updated_themes) != _theme_signature(current_themes)
            stage_counts_after = _evidence_stage_counts(current_selected)
            new_evidence_stages = sorted(
                stage
                for stage, count in stage_counts_after.items()
                if count > initial_stage_counts.get(stage, 0)
            )
            resolved_draft_gaps = sorted(
                stage
                for stage in current_draft_audit.get("missing_evidence_stages", [])
                if stage_counts_after.get(stage, 0) > 0
            )
            integration_roles = {
                paper.paper_id: {
                    "rhetorical_role": paper.rhetorical_role,
                    "evidence_stages": paper.evidence_stages,
                    "theme_ids": paper.theme_ids,
                }
                for paper in supplemental_selected
            }
            synthesis_changed = bool(
                selected_gain and (theme_signature_changed or new_evidence_stages)
            )
            supplemental_rounds.append({
                "round": round_index,
                "audit": asdict(audit),
                "new_candidate_count": len(new_supplemental),
                "new_selected_count": selected_gain,
                "unique_retrieval_gain": round(unique_gain, 4),
                "theme_signature_changed": theme_signature_changed,
                "evidence_stage_counts_after": stage_counts_after,
                "new_evidence_stages": new_evidence_stages,
                "resolved_draft_gaps": resolved_draft_gaps,
                "integration_roles": integration_roles,
                "synthesis_changed": synthesis_changed,
            })
            current_themes = updated_themes
            if round_index < self.config.max_reflection_rounds:
                with self._stage(f"provisional_rewrite_round_{round_index + 1}"):
                    provisional_review, provisional_rejected, _ = self.writer.write(
                        current_selected, current_themes
                    )
                (self.out / f"05_review_provisional_round_{round_index + 1}.md").write_text(
                    provisional_review, encoding="utf-8"
                )
                current_draft_audit = build_draft_gap_audit(
                    self.config.topic,
                    provisional_review,
                    current_selected,
                    current_themes,
                )
                current_draft_audit["rejected_unknown_citation_sentences"] = provisional_rejected
                write_json(
                    self.out / f"05_draft_gap_audit_round_{round_index + 1}.json",
                    current_draft_audit,
                )
                write_json(
                    self.out / f"05_argument_coherence_audit_round_{round_index + 1}.json",
                    self.writer.audit_argument(
                        provisional_review, current_selected, current_themes
                    ),
                )
            if (
                round_index >= self.config.min_reflection_rounds
                and selected_gain < self.config.min_supplemental_selected_gain
                and unique_gain < self.config.min_unique_retrieval_gain
                and not theme_signature_changed
            ):
                supplemental_rounds[-1]["stop_reason"] = "marginal_gain_below_threshold"
                break
        write_json(self.out / "08_supplemental_candidates.json", [paper.to_dict() for paper in all_new_supplemental])
        write_json(self.out / "08_supplemental_rounds.json", supplemental_rounds)

        final_selected = current_selected
        final_themes = current_themes
        self.writer.assign_citations(final_selected, final_themes)
        write_json(
            self.out / "09_selected_final.json",
            [paper.to_dict() for paper in final_selected],
        )
        final_coverage = self.reasoner.coverage_snapshot(final_selected)
        write_json(self.out / "09_coverage_final.json", final_coverage)
        write_csv(
            self.out / "09_extractions_final.csv",
            (paper.to_dict() for paper in final_selected),
            EXTRACTION_FIELDS,
        )
        write_json(
            self.out / "10_themes_final.json",
            [asdict(theme) for theme in final_themes],
        )
        heading_architecture = heading_architecture_report(final_themes)
        write_json(self.out / "10_heading_architecture.json", heading_architecture)
        supplemental_ids = {paper.paper_id for paper in all_supplemental_selected}
        integrated_ids = sorted(
            paper.paper_id
            for paper in final_selected
            if paper.paper_id in supplemental_ids and paper.theme_ids
        )
        final_stage_counts = _evidence_stage_counts(final_selected)
        integration_report = {
            "provisional_review_generated_before_reflection": bool(
                self.config.provisional_review_before_reflection
                and (self.out / "05_review_provisional.md").exists()
            ),
            "supplemental_search_executed": any(
                not row.get("stopped_before_search") for row in supplemental_rounds
            ),
            "supplemental_selected_count": len(all_supplemental_selected),
            "supplemental_integrated_count": len(integrated_ids),
            "supplemental_integrated_paper_ids": integrated_ids,
            "initial_evidence_stage_counts": initial_stage_counts,
            "final_evidence_stage_counts": final_stage_counts,
            "new_or_deepened_evidence_stages": sorted(
                stage
                for stage, count in final_stage_counts.items()
                if count > initial_stage_counts.get(stage, 0)
            ),
            "theme_structure_changed": _theme_signature(final_themes)
            != _theme_signature(themes_initial),
            "synthesis_changed": any(
                bool(row.get("synthesis_changed")) for row in supplemental_rounds
            ),
        }
        integration_report["closed_loop_complete"] = bool(
            integration_report["provisional_review_generated_before_reflection"]
            and integration_report["supplemental_search_executed"]
            and integration_report["supplemental_integrated_count"] > 0
            and integration_report["synthesis_changed"]
        )
        (self.out / "10_theme_map.md").write_text(
            render_theme_map(final_themes, final_selected), encoding="utf-8"
        )

        (
            review,
            rejected_citation_sentences,
            claims,
            final_argument_audit,
            writing_revision_rounds,
        ) = self._generate_and_revise_review(final_selected, final_themes)
        review_path = self.out / "11_review_draft.md"
        review_path.write_text(review, encoding="utf-8")
        final_body_citations = set(
            citation_counts(review.split("## 参考文献", 1)[0])
        )
        integrated_ids = sorted(
            paper.paper_id
            for paper in final_selected
            if paper.paper_id in supplemental_ids
            and paper.theme_ids
            and paper.citation_number in final_body_citations
        )
        integration_report["supplemental_integrated_paper_ids"] = integrated_ids
        integration_report["supplemental_integrated_count"] = len(integrated_ids)
        integration_report["provisional_argument_changed"] = bool(
            provisional_review
            and _normalized_review_argument(provisional_review)
            != _normalized_review_argument(review)
        )
        integration_report["synthesis_changed"] = bool(
            integration_report["synthesis_changed"]
            or integration_report["provisional_argument_changed"]
        )
        integration_report["closed_loop_complete"] = bool(
            integration_report["provisional_review_generated_before_reflection"]
            and integration_report["supplemental_search_executed"]
            and integration_report["supplemental_integrated_count"] > 0
            and integration_report["synthesis_changed"]
        )
        write_json(
            self.out / "08_supplemental_integration_report.json",
            integration_report,
        )
        if (
            self.config.require_supplemental_integration
            and not integration_report["closed_loop_complete"]
        ):
            self.warnings.append(
                "补充检索尚未形成可验证的论证整合：新增文献必须通过筛选、分配证据角色，"
                "并改变主题结构、证据阶段覆盖或初稿论证，闭环才算完成。"
            )
        cited_numbers = final_body_citations
        (self.out / "12_references.bib").write_text(
            render_bibtex(
                [
                    paper
                    for paper in final_selected
                    if paper.citation_number in cited_numbers
                ]
            ),
            encoding="utf-8",
        )
        write_json(self.out / "14_claim_evidence_map.json", [asdict(claim) for claim in claims])
        write_json(
            self.out / "15_argument_coherence_audit.json", final_argument_audit
        )
        publication_report = publication_quality_report(
            review, final_themes, final_argument_audit
        )
        write_json(self.out / "16_publication_quality_report.json", publication_report)

        evaluation_status = create_human_evaluation_pack(
            self.out,
            merge_papers(all_new_supplemental, existing=candidates),
            final_selected,
            claims,
            seed=self.config.seed,
        )

        cited = set(citation_counts(review))
        uncited = [
            paper.paper_id
            for paper in final_selected
            if paper.citation_number not in cited
        ]
        unclassified = [paper.paper_id for paper in final_selected if not paper.theme_ids]
        checks = {
            "candidate_floor_met": initial_candidate_count
            >= self.config.min_candidates,
            "selected_floor_met": len(final_selected) >= self.config.min_selected,
            "theme_count_within_configured_range": self.config.theme_min <= len(final_themes) <= self.config.theme_max,
            "unclassified_papers_explicitly_allowed": True,
            "reflection_loop_executed": bool(supplemental_rounds),
            "supplemental_search_executed": any(not row.get("stopped_before_search") for row in supplemental_rounds),
            "provisional_review_preceded_reflection": integration_report[
                "provisional_review_generated_before_reflection"
            ],
            "supplemental_evidence_integrated": bool(
                integration_report["supplemental_integrated_count"]
            ),
            "supplemental_changed_synthesis": integration_report[
                "synthesis_changed"
            ],
            "draft_aware_closed_loop_complete": integration_report[
                "closed_loop_complete"
            ],
            "unknown_citation_sentences_absent": not rejected_citation_sentences,
            "all_selected_papers_cited": not uncited,
            "publication_body_has_no_tables": publication_report["checks"]["no_markdown_tables"],
            "publication_body_has_no_audit_language": publication_report["checks"]["no_backstage_process_language"],
            "publication_references_resolve": publication_report["checks"]["all_citations_have_reference_entries"],
            "publication_reference_entries_are_cited": publication_report["checks"]["all_reference_entries_are_cited"],
            "heading_architecture_automatic_checks_pass": heading_architecture[
                "automatic_checks_pass"
            ],
            "argument_coherence_automatic_checks_pass": final_argument_audit[
                "automatic_checks_pass"
            ],
            "formal_llm_mode_used": self.llm.enabled,
            "human_screening_evaluation_complete": evaluation_status["screening_gold_complete"],
            "extraction_fact_check_complete": evaluation_status["extraction_fact_check_complete"],
            "claim_evidence_human_audit_complete": evaluation_status["claim_evidence_human_audit_complete"],
        }
        integrity = RunIntegrityReport(
            candidate_count=initial_candidate_count,
            selected_count=len(final_selected),
            theme_count=len(final_themes),
            supplemental_candidate_count=len(all_new_supplemental),
            supplemental_selected_count=len(all_supplemental_selected),
            rejected_unknown_citation_sentences=rejected_citation_sentences,
            uncited_selected_papers=uncited,
            warnings=self.warnings
            + self.reasoner.fallbacks
            + self.writer.warnings,
            checks=checks,
        )
        write_json(self.out / "13_run_integrity_report.json", asdict(integrity))
        uncited_factual = _uncited_factual_sentences(review)
        abstract_only_claims = sum(
            claim.support_type == "abstract_or_partial_text_bound"
            and any(
                paper.extraction_basis == "abstract_only"
                for paper in final_selected
                if paper.paper_id in claim.supporting_paper_ids
            )
            for claim in claims
        )
        content_audit = ContentQualityAudit(
            claim_count=len(claims),
            claims_with_citations=len(claims),
            claims_with_evidence_spans=sum(bool(claim.supporting_evidence_spans) for claim in claims),
            claims_requiring_human_verification=len(claims),
            uncited_factual_sentences=uncited_factual,
            abstract_only_claims=abstract_only_claims,
            partial_text_claims=sum(
                any(
                    paper.extraction_basis in {"partial_fulltext", "ocr_or_layout_unreliable"}
                    for paper in final_selected
                    if paper.paper_id in claim.supporting_paper_ids
                )
                for claim in claims
            ),
            human_screening_evaluation_complete=False,
            extraction_fact_check_complete=False,
            claim_evidence_human_audit_complete=False,
            warnings=[
                "自动证据绑定只证明可追溯，不证明语义蕴含；需完成人工论断—证据审核。",
                f"{len(unclassified)}篇论文未被强行归类。" if unclassified else "所有论文均有至少一个语义主题归属。",
            ],
        )
        write_json(self.out / "15_content_quality_audit.json", asdict(content_audit))

        finished = dt.datetime.now(dt.timezone.utc)
        manifest = {
            "run_id": self.out.name,
            "topic": self.config.topic,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_seconds": round((finished - started).total_seconds(), 2),
            "llm_enabled": self.llm.enabled,
            "llm_model": self.config.llm_model if self.llm.enabled else None,
            "llm_parameters": {
                "temperature_default": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
                "timeout_seconds": self.config.request_timeout,
                "retries": self.config.request_retries,
                "seed": self.config.seed,
            },
            "prompt_versions": {
                "query": QUERY_PROMPT_VERSION,
                "screen": SCREEN_PROMPT_VERSION,
                "extract": EXTRACT_PROMPT_VERSION,
                "theme": THEME_PROMPT_VERSION,
                "reflect": REFLECT_PROMPT_VERSION,
                "write": WRITE_PROMPT_VERSION,
                "argument_audit": ARGUMENT_AUDIT_PROMPT_VERSION,
                "writing_revision": REVISION_PROMPT_VERSION,
            },
            "llm_calls": [asdict(call) for call in self.llm.calls],
            "llm_usage_summary": _llm_usage_summary(self.llm.calls, self.llm.failed_calls, self.llm.json_repair_attempts),
            "stage_timings_seconds": self.stage_timings,
            "http": {
                "network_request_count": self.search.http.request_count,
                "cache_hits": self.search.http.cache_hits,
                "network_failures": self.search.http.network_failures,
                "requests": self.search.http.requests,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "os_name": os.name,
                "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            },
            "input_hashes": _reproducibility_hashes(self.config),
            "fallbacks": self.reasoner.fallbacks + self.writer.warnings,
            "source_errors": self.source_errors,
            "artifacts": sorted(
                str(path.relative_to(self.out))
                for path in self.out.rglob("*")
                if path.is_file()
            ),
            "run_integrity_checks": checks,
            "content_quality_validation_complete": evaluation_status["quality_validation_complete"],
            "evidence_basis_counts": _count_values(paper.extraction_basis for paper in final_selected),
            "unclassified_paper_ids": unclassified,
            "coverage_snapshot": final_coverage,
            "publication_quality": publication_report,
            "heading_architecture": heading_architecture,
            "argument_coherence": {
                "automatic_checks_pass": final_argument_audit[
                    "automatic_checks_pass"
                ],
                "rewrite_required": final_argument_audit["rewrite_required"],
                "needs_human_review": final_argument_audit["needs_human_review"],
                "writing_revision_rounds_executed": len(writing_revision_rounds),
            },
        }
        write_json(self.out / "run_manifest.json", manifest)
        delivery_manifest_path = None
        if self.config.create_delivery_views:
            delivery_manifest_path = _create_delivery_views(self.out)
        self.logger.info(
            "run finished selected=%d themes=%d elapsed=%.2fs",
            len(final_selected),
            len(final_themes),
            (finished - started).total_seconds(),
        )
        result = {
            "review_path": review_path,
            "manifest_path": self.out / "run_manifest.json",
            "integrity_path": self.out / "13_run_integrity_report.json",
            "content_quality_path": self.out / "15_content_quality_audit.json",
            "publication_quality_path": self.out / "16_publication_quality_report.json",
            "delivery_manifest_path": delivery_manifest_path,
            "integrity": asdict(integrity),
        }
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
        return result


def render_theme_map(themes: list[Theme], papers: list[Paper]) -> str:
    by_id = {paper.paper_id: paper for paper in papers}
    lines = ["# 主题组织与证据关系图", ""]
    for theme in themes:
        lines.extend(
            [
                f"## {theme.theme_id} {theme.name}",
                "",
                f"- 发表用小标题：{theme.publication_heading or theme.name}",
                f"- 小标题功能：{theme.heading_function}",
                f"- 承接问题：{theme.inherited_question}",
                f"- 组织问题：{theme.organizing_question}",
                f"- 综合判断：{theme.synthesis_claim}",
                f"- 归类依据：{theme.classification_basis}",
                f"- 残余缺口：{theme.residual_gap}",
                f"- 下一问题：{theme.next_question}",
                f"- 与前主题关系：{theme.relation_to_previous}",
                f"- 关系依据：{theme.heading_relation_rationale}",
                f"- 并列组：{theme.parallel_group or '不适用'}",
                f"- 标题质量状态：{theme.heading_quality_status}",
                f"- 关系证据状态：{theme.relation_support_status}",
                f"- 关系证据论文：{', '.join(theme.relation_evidence_paper_ids) or '无；仅编辑性过渡'}",
                "- 文献：",
            ]
        )
        for paper_id in theme.paper_ids:
            paper = by_id.get(paper_id)
            if paper:
                lines.append(
                    f"  - {paper_id}: {paper.title} ({paper.year or 'n.d.'})"
                )
        lines.append("")
    unclassified = [paper for paper in papers if not paper.theme_ids]
    if unclassified:
        lines.extend(["## 未归类文献", "", "下列论文未达到任何主题的语义归属要求，未被机械分配：", ""])
        lines.extend(f"- {paper.paper_id}: {paper.title}" for paper in unclassified)
        lines.append("")
    return "\n".join(lines)


def _paper_key(paper: Paper) -> str:
    if paper.doi:
        return "doi:" + paper.doi.casefold()
    title = " ".join(paper.title.casefold().split())
    return "title:" + title


def _theme_signature(themes: list[Theme]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        sorted((theme.name.casefold(), tuple(sorted(theme.paper_ids))) for theme in themes)
    )


def _normalized_review_argument(review: str) -> str:
    """Compare argument text while ignoring references, numbering, and spacing."""
    body = re.split(r"^##\s+(?:参考文献|References)\s*$", review, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE)[0]
    body = re.sub(r"^#.*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"\[(?:\d+\s*[,;]?\s*)+\]", "", body)
    return re.sub(r"\s+", "", body).casefold()


EVIDENCE_STAGE_ORDER = [
    "concept_definition",
    "mechanism",
    "identification",
    "real_time_inference",
    "prediction",
    "incremental_prediction",
    "decision_change",
    "net_value",
    "external_validity",
]

EVIDENCE_STAGE_LABELS = {
    "concept_definition": "概念与对象界定",
    "mechanism": "形成、持续与转换机制",
    "identification": "识别或测量",
    "real_time_inference": "实时/事前信息集推断",
    "prediction": "样本外预测",
    "incremental_prediction": "相对基准的增量预测",
    "decision_change": "决策或行为改变",
    "net_value": "计入成本约束后的净价值",
    "external_validity": "复现与外部有效性",
}


def _evidence_stage_counts(papers: list[Paper]) -> dict[str, int]:
    return {
        stage: sum(stage in paper.evidence_stages for paper in papers)
        for stage in EVIDENCE_STAGE_ORDER
    }


def build_draft_gap_audit(
    topic: str,
    draft: str,
    papers: list[Paper],
    themes: list[Theme],
) -> dict[str, Any]:
    """Audit the provisional argument without leaking audit prose into publication."""
    stage_counts = _evidence_stage_counts(papers)
    context = f"{topic}\n{draft}".casefold()
    required = {"concept_definition", "mechanism", "identification", "external_validity"}
    conditional_patterns = {
        "real_time_inference": r"real[- ]?time|online|nowcast|实时|在线|事前信息集",
        "prediction": r"predict|forecast|out[- ]?of[- ]?sample|预测|样本外",
        "incremental_prediction": r"incremental|benchmark|baseline|增量|基准",
        "decision_change": r"decision|allocation|policy|intervention|portfolio|决策|配置|政策|干预",
        "net_value": r"transaction cost|turnover|utility|net value|交易成本|换手|效用|净价值|实施成本",
    }
    for stage, pattern in conditional_patterns.items():
        if re.search(pattern, context, re.IGNORECASE):
            required.add(stage)
    missing = [
        stage
        for stage in EVIDENCE_STAGE_ORDER
        if stage in required and stage_counts.get(stage, 0) == 0
    ]

    citation_to_paper = {
        paper.citation_number: paper
        for paper in papers
        if paper.citation_number is not None
    }
    claim_patterns = {
        "mechanism": r"mechanism|cause|formation|persistence|transition|机制|导致|形成|持续|转换",
        "real_time_inference": r"real[- ]?time|online|nowcast|实时|在线|当期识别|事前识别",
        "prediction": r"predict|forecast|out[- ]?of[- ]?sample|预测|样本外",
        "incremental_prediction": r"incremental|beyond (a |the )?baseline|增量|优于基准|超越基准",
        "decision_change": r"decision|allocation|policy change|portfolio|决策|配置|政策改变",
        "net_value": r"net value|transaction cost|turnover|utility|净价值|交易成本|换手|效用",
        "external_validity": r"external validity|generaliz|replicat|cross[- ]?(market|country|setting)|外部有效|推广|复现|跨市场|跨国家",
    }
    unsupported: list[str] = []
    body = draft.split("## 参考文献", 1)[0]
    sentences = re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n+", body)
    for sentence in sentences:
        value = sentence.strip()
        # Chinese scholarly claims are often semantically dense at 20-30 chars.
        if len(value) < 18 or value.startswith("#"):
            continue
        cited_numbers = [
            int(number)
            for group in re.findall(r"\[((?:\d+\s*[,;]?\s*)+)\]", value)
            for number in re.findall(r"\d+", group)
        ]
        cited_papers = [citation_to_paper[number] for number in cited_numbers if number in citation_to_paper]
        for stage, pattern in claim_patterns.items():
            if not re.search(pattern, value, re.IGNORECASE):
                continue
            if not cited_papers or not any(stage in paper.evidence_stages for paper in cited_papers):
                unsupported.append(
                    f"{EVIDENCE_STAGE_LABELS[stage]}论断缺少同阶段证据：{value[:180]}"
                )
    unsupported = list(dict.fromkeys(unsupported))[:12]
    thin_themes = [
        {
            "theme_id": theme.theme_id,
            "theme_name": theme.name,
            "paper_count": len(theme.paper_ids),
        }
        for theme in themes
        if len(theme.paper_ids) < 3
    ]
    return {
        "audit_scope": "provisional_draft_only",
        "summary": (
            f"初稿覆盖{len(themes)}个主题；发现{len(missing)}个证据阶段空缺、"
            f"{len(unsupported)}处可能的阶段越级论断和{len(thin_themes)}个薄主题。"
        ),
        "evidence_stage_counts": stage_counts,
        "required_evidence_stages": [
            stage for stage in EVIDENCE_STAGE_ORDER if stage in required
        ],
        "missing_evidence_stages": missing,
        "unsupported_transition_claims": unsupported,
        "thin_themes": thin_themes,
        "publication_boundary": (
            "本文件仅用于补检与改稿决策，不得复制到综述正文。"
        ),
    }


def _uncited_factual_sentences(review: str) -> list[str]:
    body = review.split("## 参考文献", 1)[0]
    values = re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n+", body)
    flagged = []
    for value in values:
        sentence = value.strip()
        if (
            len(sentence) < 45
            or sentence.startswith(("#", ">", "说明：", "Note:"))
            or re.search(r"\[\d+(?:\s*[,;]\s*\d+)*\]", sentence)
        ):
            continue
        flagged.append(sentence)
    return flagged


def _llm_usage_summary(calls: list[Any], failed_calls: int, json_repairs: int) -> dict[str, Any]:
    known_prompt = [call.prompt_tokens for call in calls if call.prompt_tokens is not None]
    known_completion = [call.completion_tokens for call in calls if call.completion_tokens is not None]
    return {
        "successful_calls": len(calls),
        "failed_attempts": failed_calls,
        "json_repair_attempts": json_repairs,
        "prompt_tokens": sum(known_prompt) if known_prompt else None,
        "completion_tokens": sum(known_completion) if known_completion else None,
        "total_tokens": (
            sum(known_prompt) + sum(known_completion)
            if known_prompt or known_completion
            else None
        ),
        "estimated_cost_usd": None,
        "cost_note": "未内置易过时的模型价格；请按运行日期和模型账单计算。",
    }


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reproducibility_hashes(config: PipelineConfig) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files = {
        "prompts.py": root / "src" / "litreview_agent" / "prompts.py",
        "pyproject.toml": root / "pyproject.toml",
    }
    if config.offline_corpus:
        files["offline_corpus"] = config.offline_corpus
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in files.items()
    }


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _create_delivery_views(root: Path) -> Path:
    """Create a clean publication view and a separate backstage audit view."""
    publication = root / "publication"
    audit = root / "audit"
    publication.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)

    publication_sources = {
        "11_review_draft.md": "review.md",
        "12_references.bib": "references.bib",
    }
    publication_files: list[str] = []
    for source_name, target_name in publication_sources.items():
        source = root / source_name
        if source.exists():
            target = publication / target_name
            shutil.copy2(source, target)
            publication_files.append(str(target.relative_to(root)))

    audit_files: list[str] = []
    excluded = set(publication_sources) | {"17_delivery_manifest.json"}
    for source in sorted(root.iterdir()):
        if not source.is_file() or source.name in excluded:
            continue
        target = audit / source.name
        shutil.copy2(source, target)
        audit_files.append(str(target.relative_to(root)))
    evaluation = root / "evaluation"
    if evaluation.exists():
        target = audit / "evaluation"
        shutil.copytree(evaluation, target, dirs_exist_ok=True)
        audit_files.extend(
            str(path.relative_to(root))
            for path in sorted(target.rglob("*"))
            if path.is_file()
        )

    manifest = {
        "workflow_version": "3.2",
        "publication": {
            "purpose": "Only publication-facing review prose and its cited references.",
            "files": publication_files,
        },
        "audit": {
            "purpose": (
                "Search, screening, extraction, coverage, theme, integrity, "
                "claim-evidence, and human-evaluation records."
            ),
            "files": audit_files,
        },
        "separation_rule": (
            "Backstage workflow records must not be pasted into the review body."
        ),
    }
    path = root / "17_delivery_manifest.json"
    write_json(path, manifest)
    shutil.copy2(path, audit / path.name)
    return path


def _build_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"litreview-agent:{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
