from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .config import PipelineConfig
from .io_utils import compact
from .llm import LLMError, OpenAICompatibleClient
from .models import ClaimEvidence, Paper, Theme
from .prompts import (
    ARGUMENT_AUDIT_PROMPT_VERSION,
    ARGUMENT_AUDIT_SYSTEM,
    ARGUMENT_AUDIT_USER,
    REVISION_PROMPT_VERSION,
    REVISION_SECTION_USER,
    REVISION_SYSTEM,
    WRITE_CONCLUSION_USER,
    WRITE_INTRO_USER,
    WRITE_ONE_SHOT_USER,
    WRITE_PROMPT_VERSION,
    WRITE_SECTION_USER,
    WRITE_SYSTEM,
)


class ReviewWriter:
    def __init__(self, config: PipelineConfig, llm: OpenAICompatibleClient) -> None:
        self.config = config
        self.llm = llm
        self.warnings: list[str] = []

    def assign_citations(self, papers: list[Paper], themes: list[Theme]) -> None:
        by_id = {paper.paper_id: paper for paper in papers}
        ordered_ids: list[str] = []
        for theme in themes:
            for paper_id in theme.paper_ids:
                if paper_id in by_id and paper_id not in ordered_ids:
                    ordered_ids.append(paper_id)
        for paper in papers:
            if paper.paper_id not in ordered_ids:
                ordered_ids.append(paper.paper_id)
        for number, paper_id in enumerate(ordered_ids, start=1):
            by_id[paper_id].citation_number = number

    def write(
        self, papers: list[Paper], themes: list[Theme]
    ) -> tuple[str, list[str], list[ClaimEvidence]]:
        self.assign_citations(papers, themes)
        if self.llm.enabled:
            try:
                review = (
                    self._write_with_llm_one_shot(papers, themes)
                    if self.config.generation_strategy == "one_shot"
                    else self._write_with_llm(papers, themes)
                )
            except LLMError as exc:
                if self.config.llm_mode == "openai":
                    raise
                self.warnings.append(f"review_generation fallback: {exc}")
                review = self._write_heuristic(papers, themes)
        else:
            review = self._write_heuristic(papers, themes)

        cleaned, rejected_sentences = reject_unknown_citation_sentences(
            review, {paper.citation_number for paper in papers if paper.citation_number}
        )
        if self.config.enforce_publication_clean:
            cleaned, removed = clean_publication_prose(cleaned)
            if removed:
                self.warnings.append(
                    f"publication_cleaner removed {len(removed)} process/table blocks"
                )
        claims = build_claim_evidence_map(cleaned, papers)
        cited_numbers = set(citation_counts(cleaned))
        references = render_references(papers, cited_numbers=cited_numbers)
        title = (
            f"# {self.config.topic}：文献综述"
            if self.config.language == "zh"
            else f"# {self.config.topic}: Literature Review"
        )
        return (
            f"{title}\n\n{cleaned.strip()}\n\n{references}\n",
            rejected_sentences,
            claims,
        )

    def revise(
        self,
        review: str,
        papers: list[Paper],
        themes: list[Theme],
        audit: dict[str, Any],
        *,
        round_index: int,
    ) -> tuple[str, list[str], list[ClaimEvidence]]:
        """Feed section-level audit failures back into the main LLM writer."""
        if not self.llm.enabled:
            return review, [], build_claim_evidence_map(review, papers)
        by_id = {paper.paper_id: paper for paper in papers}
        deterministic = {
            str(row.get("theme_id")): row for row in audit.get("sections", [])
        }
        semantic_payload = audit.get("semantic_audit") or {}
        semantic = {
            str(row.get("theme_id")): row
            for row in semantic_payload.get("sections", [])
        }
        revise_all = bool(semantic_payload.get("rewrite_required", False)) and not semantic
        current = review
        existing_sections = {
            _normalize_heading(row["heading"]): row["body"]
            for row in _thematic_sections(review)
        }
        for theme in themes:
            heading = theme.publication_heading or theme.name
            auto_row = deterministic.get(theme.theme_id, {})
            semantic_row = semantic.get(theme.theme_id, {})
            needs_revision = bool(
                revise_all
                or auto_row.get("automatic_checks_pass") is False
                or semantic_row.get("semantic_status") == "fail"
                or semantic_row.get("revision_instruction")
                or semantic_row.get("issues")
                or semantic_row.get("evidence_stage_overreach")
            )
            if not needs_revision:
                continue
            old_body = existing_sections.get(_normalize_heading(heading), "")
            feedback = {
                "automatic": auto_row,
                "semantic": semantic_row,
                "overall_summary": semantic_payload.get("summary", ""),
            }
            try:
                revised = self.llm.complete(
                    stage=f"review_revision_{round_index}_{theme.theme_id}",
                    prompt_version=REVISION_PROMPT_VERSION,
                    system=REVISION_SYSTEM,
                    user=REVISION_SECTION_USER.format(
                        language=self.config.language,
                        topic=self.config.topic,
                        round_index=round_index,
                        publication_heading=heading,
                        theme=json.dumps(
                            {
                                "name": theme.name,
                                "organizing_question": theme.organizing_question,
                                "synthesis_claim": theme.synthesis_claim,
                                "inherited_question": theme.inherited_question,
                                "residual_gap": theme.residual_gap,
                                "next_question": theme.next_question,
                                "internal_relations": theme.internal_relations,
                                "relation_to_previous": theme.relation_to_previous,
                            },
                            ensure_ascii=False,
                        ),
                        feedback=json.dumps(feedback, ensure_ascii=False),
                        current_section=f"## {heading}\n\n{old_body}",
                        papers="\n".join(self._theme_evidence(theme, by_id)),
                    ),
                    temperature=0.15,
                )
            except LLMError:
                if self.config.llm_mode == "openai":
                    raise
                self.warnings.append(
                    f"review_revision_{round_index}_{theme.theme_id} failed; kept previous section"
                )
                continue
            revised = _enforce_section_heading(revised, heading)
            current = _replace_thematic_section(current, heading, revised)

        body = re.split(
            r"^##\s+(?:参考文献|References)\s*$",
            current,
            maxsplit=1,
            flags=re.MULTILINE | re.IGNORECASE,
        )[0].strip()
        cleaned, rejected = reject_unknown_citation_sentences(
            body, {paper.citation_number for paper in papers if paper.citation_number}
        )
        if self.config.enforce_publication_clean:
            cleaned, removed = clean_publication_prose(cleaned)
            if removed:
                self.warnings.append(
                    f"publication_cleaner removed {len(removed)} revision process/table blocks"
                )
        cited_numbers = set(citation_counts(cleaned))
        references = render_references(papers, cited_numbers=cited_numbers)
        revised_review = f"{cleaned.strip()}\n\n{references}\n"
        return revised_review, rejected, build_claim_evidence_map(revised_review, papers)

    def audit_argument(
        self, review: str, papers: list[Paper], themes: list[Theme]
    ) -> dict[str, Any]:
        """Audit section logic; semantic uncertainty is never converted to pass."""
        report = argument_coherence_audit(review, papers, themes)
        if not self.llm.enabled:
            return report
        try:
            semantic = self.llm.complete_json(
                stage="argument_coherence_audit",
                prompt_version=ARGUMENT_AUDIT_PROMPT_VERSION,
                system=ARGUMENT_AUDIT_SYSTEM,
                user=ARGUMENT_AUDIT_USER.format(
                    topic=self.config.topic,
                    themes=json.dumps(
                        [
                            {
                                "theme_id": theme.theme_id,
                                "publication_heading": theme.publication_heading or theme.name,
                                "heading_function": theme.heading_function,
                                "relation_to_previous": theme.relation_to_previous,
                                "heading_relation_rationale": theme.heading_relation_rationale,
                                "parallel_group": theme.parallel_group,
                                "inherited_question": theme.inherited_question,
                                "synthesis_claim": theme.synthesis_claim,
                                "residual_gap": theme.residual_gap,
                                "next_question": theme.next_question,
                                "paper_ids": theme.paper_ids,
                            }
                            for theme in themes
                        ],
                        ensure_ascii=False,
                    ),
                    review=review[:60000],
                ),
                temperature=0.1,
            )
            report["semantic_audit"] = semantic
            report["semantic_audit_source"] = "llm_argument_review"
            report["needs_human_review"] = bool(
                semantic.get("needs_human_review", True)
                or report["needs_human_review"]
            )
            report["rewrite_required"] = bool(
                semantic.get("rewrite_required", False)
                or report["rewrite_required"]
            )
        except LLMError as exc:
            self.warnings.append(f"argument_coherence_audit fallback: {exc}")
            report["semantic_audit_error"] = str(exc)
        return report

    def _theme_map_text(self, themes: list[Theme], by_id: dict[str, Paper]) -> str:
        rows = []
        for theme in themes:
            citations = [
                by_id[paper_id].citation_number
                for paper_id in theme.paper_ids
                if paper_id in by_id
            ]
            rows.append(
                f"{theme.theme_id} {theme.name} | publication_heading={theme.publication_heading or theme.name} | "
                f"heading_function={theme.heading_function} | question={theme.organizing_question} | "
                f"claim={theme.synthesis_claim} | residual_gap={theme.residual_gap} | "
                f"next={theme.next_question} | citations={citations}"
                f" | relation={theme.relation_to_previous} | "
                f"relation_rationale={theme.heading_relation_rationale} | "
                f"parallel_group={theme.parallel_group} | "
                f"relation_support={theme.relation_support_status}"
            )
        return "\n".join(rows)

    def _theme_evidence(
        self, theme: Theme, by_id: dict[str, Paper]
    ) -> list[str]:
        evidence = []
        for paper_id in theme.paper_ids:
            paper = by_id.get(paper_id)
            if not paper:
                continue
            evidence.append(
                f"[{paper.citation_number}] {paper.title}; "
                f"question={compact(paper.research_question, 300)}; "
                f"methods={compact('; '.join(paper.methods), 350)}; "
                f"findings={compact('; '.join(paper.findings), 500)}; "
                f"limitations={compact('; '.join(paper.limitations), 300)}; "
                f"rhetorical_role={paper.rhetorical_role}; "
                f"concept_level={paper.concept_level}; "
                f"evidence_stages={','.join(paper.evidence_stages)}; "
                f"mechanism={compact(paper.mechanism, 220)}; "
                f"change_representation={compact(paper.change_representation, 180)}; "
                f"temporal_design={compact(paper.temporal_design, 180)}; "
                f"state_output={compact(paper.state_output, 180)}; "
                f"real_time_vintage={compact(paper.real_time_vintage, 160)}; "
                f"out_of_sample={compact(paper.out_of_sample, 160)}; "
                f"uncertainty={compact('; '.join(paper.uncertainty), 180)}; "
                f"costs_constraints={compact('; '.join(paper.costs_constraints), 180)}; "
                f"cross_setting_test={compact(paper.cross_setting_test, 180)}; "
                f"population={compact(paper.population_sample, 180)}; "
                f"design={compact(paper.study_design, 180)}; "
                f"metrics={compact('; '.join(paper.metrics), 180)}; "
                f"boundary={compact('; '.join(paper.boundary_conditions), 220)}; "
                f"evidence_spans={json.dumps(paper.evidence_spans[:6], ensure_ascii=False)}; "
                f"basis={paper.extraction_basis}"
            )
        return evidence

    def _write_with_llm(self, papers: list[Paper], themes: list[Theme]) -> str:
        by_id = {paper.paper_id: paper for paper in papers}
        theme_map = self._theme_map_text(themes, by_id)
        intro = self.llm.complete(
            stage="review_introduction",
            prompt_version=WRITE_PROMPT_VERSION,
            system=WRITE_SYSTEM,
            user=WRITE_INTRO_USER.format(
                language=self.config.language,
                topic=self.config.topic,
                themes=theme_map,
            ),
            temperature=0.25,
        )
        sections = []
        for theme in themes:
            evidence = self._theme_evidence(theme, by_id)
            sections.append(
                self.llm.complete(
                    stage=f"review_theme_{theme.theme_id}",
                    prompt_version=WRITE_PROMPT_VERSION,
                    system=WRITE_SYSTEM,
                    user=WRITE_SECTION_USER.format(
                        language=self.config.language,
                        topic=self.config.topic,
                        publication_heading=theme.publication_heading or theme.name,
                        theme=json.dumps(
                            {
                                "name": theme.name,
                                "publication_heading": theme.publication_heading or theme.name,
                                "heading_function": theme.heading_function,
                                "organizing_question": theme.organizing_question,
                                "synthesis_claim": theme.synthesis_claim,
                                "inherited_question": theme.inherited_question,
                                "residual_gap": theme.residual_gap,
                                "next_question": theme.next_question,
                                "internal_relations": theme.internal_relations,
                                "relation_to_previous": theme.relation_to_previous,
                                "heading_relation_rationale": theme.heading_relation_rationale,
                                "parallel_group": theme.parallel_group,
                                "relation_evidence_paper_ids": theme.relation_evidence_paper_ids,
                                "relation_support_status": theme.relation_support_status,
                            },
                            ensure_ascii=False,
                        ),
                        papers="\n".join(evidence),
                    ),
                    temperature=0.25,
                )
            )
            sections[-1] = _enforce_section_heading(
                sections[-1], theme.publication_heading or theme.name
            )
        conclusion = self.llm.complete(
            stage="review_conclusion",
            prompt_version=WRITE_PROMPT_VERSION,
            system=WRITE_SYSTEM,
            user=WRITE_CONCLUSION_USER.format(
                language=self.config.language,
                topic=self.config.topic,
                themes=theme_map,
            ),
            temperature=0.25,
        )
        return "\n\n".join([intro, *sections, conclusion])

    def _write_with_llm_one_shot(self, papers: list[Paper], themes: list[Theme]) -> str:
        by_id = {paper.paper_id: paper for paper in papers}
        rows = [self._theme_map_text(themes, by_id)]
        for paper in papers:
            rows.append(
                f"[{paper.citation_number}] {paper.title}; "
                f"question={compact(paper.research_question, 220)}; "
                f"methods={compact('; '.join(paper.methods), 250)}; "
                f"findings={compact('; '.join(paper.findings), 360)}; "
                f"limitations={compact('; '.join(paper.limitations), 220)}; "
                f"rhetorical_role={paper.rhetorical_role}; "
                f"concept_level={paper.concept_level}; "
                f"evidence_stages={','.join(paper.evidence_stages)}; "
                f"mechanism={compact(paper.mechanism, 180)}; "
                f"out_of_sample={compact(paper.out_of_sample, 130)}; "
                f"costs_constraints={compact('; '.join(paper.costs_constraints), 130)}; "
                f"cross_setting_test={compact(paper.cross_setting_test, 130)}; "
                f"basis={paper.extraction_basis}"
            )
        return self.llm.complete(
            stage="review_one_shot_baseline",
            prompt_version=WRITE_PROMPT_VERSION,
            system=WRITE_SYSTEM,
            user=WRITE_ONE_SHOT_USER.format(
                language=self.config.language,
                topic=self.config.topic,
                evidence="\n".join(rows),
            ),
            temperature=0.25,
        )

    def _write_heuristic(self, papers: list[Paper], themes: list[Theme]) -> str:
        by_id = {paper.paper_id: paper for paper in papers}
        if self.config.language == "zh":
            return self._write_heuristic_zh(by_id, themes)
        return self._write_heuristic_en(by_id, themes)

    def _write_heuristic_zh(
        self, by_id: dict[str, Paper], themes: list[Theme]
    ) -> str:
        lead_numbers = [
            by_id[paper_id].citation_number
            for theme in themes
            for paper_id in theme.paper_ids[:2]
            if paper_id in by_id
        ]
        lines = [
            "## 引言",
            "",
            (
                f"围绕“{self.config.topic}”的研究并非单一方法或结论的累积，而是由概念边界、"
                "解释机制、经验识别和适用条件相互牵引的问题链。已有工作一方面扩展了可观察对象"
                "和分析工具，另一方面也不断暴露不同数据、情境与评价尺度之间难以直接通约的张力"
                f"{cite(lead_numbers[:6])}。因此，真正需要解释的不是哪一种方法占据优势，而是各类"
                "证据在什么问题上形成共识、在什么条件下发生分歧，以及这些分歧如何推动新的研究问题。"
            ),
            "",
            (
                "现有讨论可沿若干彼此承接但并不必然线性的主题展开：基础性研究首先界定对象与"
                "核心争议，方法研究随后把抽象问题转化为可识别的经验命题，应用研究检验这些命题"
                "在不同场景中的稳定性，而对局限和边界的讨论又反过来修正最初的概念与方法假设。"
            ),
        ]
        for index, theme in enumerate(themes, start=1):
            group = [by_id[pid] for pid in theme.paper_ids if pid in by_id]
            if not group:
                continue
            numbers = [paper.citation_number for paper in group if paper.citation_number]
            methods = _representative_items(group, "methods", 4)
            findings = _representative_items(group, "findings", 4)
            limitations = _representative_items(group, "limitations", 4)
            lines.extend(
                [
                    "",
                    f"## {theme.publication_heading or theme.name}",
                    "",
                    (
                        f"{theme.organizing_question} 现有研究的共同起点是：{theme.synthesis_claim}"
                        f"{cite(numbers[:6])}。这些工作之所以构成同一讨论，并非因为结论完全一致，"
                        f"而是因为它们都从{theme.classification_basis}切入同一问题。"
                    ),
                    "",
                    (
                        f"在解释路径上，相关研究主要使用{join_items(methods)}。这些方法提供了互补"
                        "视角：一部分工作强调识别与测量，另一部分工作更关注结果在具体环境中的表现。"
                        f"综合可见，较为稳定的发现包括{join_items(findings)}{cite(numbers[6:12] or numbers[:6])}。"
                        "然而，方法之间的差异并不只是技术选择，它们往往对应不同的研究对象、时间尺度"
                        "和有效性标准，因而相似的结果未必具有相同含义，相反的结果也未必构成直接否定。"
                    ),
                    "",
                    (
                        f"争议进一步集中在{join_items(limitations)}。{theme.residual_gap}"
                        f"{cite(numbers[12:] or numbers[-4:])}。这使讨论自然转向一个更严格的问题："
                        f"{theme.next_question}"
                    ),
                ]
            )
        conclusion_numbers = [
            by_id[pid].citation_number
            for theme in themes
            for pid in theme.paper_ids[-1:]
            if pid in by_id
        ]
        lines.extend(
            [
                "",
                "## 总结与展望",
                "",
                (
                    f"总体而言，关于“{self.config.topic}”的知识积累已经从对象识别推进到机制解释、"
                    "经验检验与边界辨析，但这种推进不是简单的线性替代。不同研究传统在概念、数据和"
                    f"评价标准上的差异，既造成结论分化，也构成相互校正的基础{cite(conclusion_numbers)}。"
                ),
                "",
                (
                    "后续研究的关键不在于继续增加彼此孤立的案例，而在于建立能够比较竞争解释的"
                    "研究设计，报告负结果与条件依赖，扩展跨时期、跨区域和跨制度环境的外部验证，"
                    "并明确区分预测改善、机制识别与因果解释。只有当这些证据边界被同时纳入讨论，"
                    "该领域才能从方法上的局部成功走向可累积、可比较的知识体系。"
                ),
            ]
        )
        return "\n".join(lines)

    def _write_heuristic_zh_legacy(
        self, by_id: dict[str, Paper], themes: list[Theme]
    ) -> str:
        all_numbers = [
            by_id[paper_id].citation_number
            for theme in themes
            for paper_id in theme.paper_ids[:2]
            if paper_id in by_id
        ]
        intro_cites = cite(all_numbers[:4])
        lines = [
            "## 引言",
            "",
            f"围绕“{self.config.topic}”的研究横跨多个问题层级，既包括概念与对象的界定，也涉及方法选择、经验证据和评价边界。"
            f"本综述依据多组检索式获得候选文献，经相关性筛选和结构化抽取后，"
            f"将证据组织为{len(themes)}个主题{intro_cites}。组织原则不是按论文逐篇罗列，"
            "而是区分各研究流之间的演进、互补、竞争、冲突、并行或断裂；只有在抽取证据支持时才使用递进叙事。",
            "",
            "下文首先概括研究对象与主要解释路径，继而比较不同方法及其证据，最后讨论现有研究在可比性、"
            "外部有效性和评价设计方面尚未解决的问题。由于部分记录只能获得摘要，文中的细粒度方法和局限判断"
            "应被视为可追溯的初步综合，而不是对全文证据的最终替代。",
        ]
        for index, theme in enumerate(themes):
            theme_papers = [by_id[pid] for pid in theme.paper_ids if pid in by_id]
            if not theme_papers:
                continue
            numbers = [p.citation_number for p in theme_papers if p.citation_number]
            method_terms = _representative_items(theme_papers, "methods", 3)
            finding_terms = _representative_items(theme_papers, "findings", 3)
            limitation_terms = _representative_items(theme_papers, "limitations", 3)
            early = sorted(theme_papers, key=lambda p: p.year or 9999)[:2]
            recent = sorted(theme_papers, key=lambda p: p.year or 0, reverse=True)[:2]
            relation_cites = cite(
                [p.citation_number for p in early + recent if p.citation_number]
            )
            lines.extend(
                [
                    "",
                    f"## {theme.publication_heading or theme.name}",
                    "",
                    f"{theme.organizing_question} 当前可用记录支持的谨慎综合是："
                    f"{theme.synthesis_claim}{cite(numbers[:4])}。这些文献之所以被归为一组，"
                    f"主要因为{theme.classification_basis}。从整体上看，它们不是相互替代的单一路线，"
                    f"主题与前一组的关系被标记为“{theme.relation_to_previous}”，证据状态为“{theme.relation_support_status}”；"
                    "该标记不等同于已被全文人工核验的学术史结论。",
                    "",
                    f"方法层面，现有研究主要涉及{join_items(method_terms)}。按发表时间和问题设置观察，"
                    f"较早研究通常承担概念界定或基础建模角色，后续研究则将问题推进到更具体的场景与评价要求"
                    f"{relation_cites}。可从当前证据中谨慎概括的发现包括：{join_items(finding_terms)}"
                    f"{cite(numbers[2:7] or numbers[:3])}。这种综合说明，同一主题内部的进展更多表现为"
                    "问题逐步细化、证据类型增加以及方法之间的对照，而不是某一篇论文对其他工作的简单取代。",
                    "",
                    f"不过，该组研究仍暴露出{join_items(limitation_terms)}。"
                    f"{theme.residual_gap or '摘要层面的证据不足以完全判断方法适用边界。'}"
                    f"因此，下一步需要回答的是：{theme.next_question or '如何用可比较的评价设计检验这些结论？'}",
                ]
            )
        conclusion_cites = cite(
            [
                by_id[pid].citation_number
                for theme in themes
                for pid in theme.paper_ids[:1]
                if pid in by_id and by_id[pid].citation_number
            ]
        )
        lines.extend(
            [
                "",
                "## 总结与展望",
                "",
                f"综合各主题可以看到，关于“{self.config.topic}”的文献覆盖了问题界定、方法构造、应用与评价等层面"
                f"{conclusion_cites}。相对稳定的共识是：研究结论必须与具体对象、数据和评价条件相联系；"
                "真正的分歧则集中在方法假设能否跨场景成立，以及不同研究的指标是否足以支撑直接比较。",
                "",
                "现有证据仍有四类值得优先补充的空白：第一，增加能够直接比较主要方法族的统一基准；"
                "第二，报告负结果、边界条件和作者明确陈述的局限；第三，扩大跨地区、跨数据集或跨人群的外部验证；"
                "第四，将摘要层面的初筛进一步升级为全文证据核验。后续研究还应公开检索式、筛选理由和结构化抽取结果，"
                "从而让综述结论能够被复查、更新和重复使用。",
            ]
        )
        return "\n".join(lines)

    def _write_heuristic_en(
        self, by_id: dict[str, Paper], themes: list[Theme]
    ) -> str:
        lines = [
            "## Introduction",
            "",
            f"Research on {self.config.topic} spans conceptual, methodological, empirical, and evaluative questions. "
            f"This review organizes the screened evidence into {len(themes)} connected themes. "
            "The organizing logic is a progression of unresolved questions rather than a paper-by-paper inventory.",
        ]
        for index, theme in enumerate(themes):
            group = [by_id[pid] for pid in theme.paper_ids if pid in by_id]
            nums = [p.citation_number for p in group if p.citation_number]
            lines.extend(
                [
                    "",
                    f"## {theme.publication_heading or theme.name}",
                    "",
                    f"{theme.synthesis_claim}{cite(nums[:5])} The studies are grouped because {theme.classification_basis}. "
                    f"Across the group, methods include {join_items(_representative_items(group, 'methods', 3))}.",
                    "",
                    f"The available evidence suggests {join_items(_representative_items(group, 'findings', 3))}"
                    f"{cite(nums[2:7] or nums[:3])}. However, {theme.residual_gap} "
                    f"The next question is therefore: {theme.next_question}",
                ]
            )
        lines.extend(
            [
                "",
                "## Conclusion and Outlook",
                "",
                "The literature has established a connected chain from problem definition to methods and evaluation, "
                "but stronger comparative benchmarks, explicit boundary conditions, and broader external validation remain priorities.",
            ]
        )
        return "\n".join(lines)


def _enforce_section_heading(text: str, publication_heading: str) -> str:
    """Make the publication heading deterministic without rewriting body prose."""
    required = f"## {publication_heading.strip()}"
    lines = text.strip().splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^##\s+", line.strip()):
            lines[index] = required
            return "\n".join(lines)
        if line.strip():
            break
    return f"{required}\n\n{text.strip()}"


def _replace_thematic_section(text: str, heading: str, replacement: str) -> str:
    """Replace one level-2 thematic section without touching adjacent sections."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$.*?(?=^##\s+|\Z)",
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    normalized = _enforce_section_heading(replacement, heading).strip()
    if pattern.search(text):
        return pattern.sub(normalized + "\n\n", text, count=1)
    reference = re.search(
        r"^##\s+(?:参考文献|References)\s*$",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if reference:
        return (
            text[: reference.start()].rstrip()
            + "\n\n"
            + normalized
            + "\n\n"
            + text[reference.start() :]
        )
    return text.rstrip() + "\n\n" + normalized + "\n"


PUBLICATION_FORBIDDEN_PATTERNS = [
    r"本文采用",
    r"本综述采用",
    r"本文通过.{0,30}检索",
    r"本综述通过.{0,30}检索",
    r"共检索(?:到|出|获得)?\s*\d+",
    r"筛选后纳入",
    r"候选论文",
    r"检索式",
    r"证据矩阵",
    r"质量审计",
    r"质量门",
    r"决策级状态评价框",
    r"自动化检索",
    r"自动化证据",
    r"this review (?:used|uses|searched)",
    r"we searched (?:the|for)",
    r"search strategy",
    r"screening process",
    r"evidence matrix",
    r"quality gate",
]


def clean_publication_prose(text: str) -> tuple[str, list[str]]:
    """Remove backstage process prose and Markdown tables from the review body."""
    forbidden = re.compile("|".join(PUBLICATION_FORBIDDEN_PATTERNS), re.IGNORECASE)
    removed: list[str] = []
    output: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">") or (
            stripped.startswith("|") and stripped.endswith("|")
        ):
            if stripped:
                removed.append(stripped)
            continue
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue
        sentences = re.split(r"(?<=[。！？.!?])\s*", line)
        kept = []
        for sentence in sentences:
            if forbidden.search(sentence):
                removed.append(sentence.strip())
            elif sentence.strip():
                kept.append(sentence.strip())
        if kept:
            output.append("".join(kept) if re.search(r"[\u4e00-\u9fff]", line) else " ".join(kept))
    cleaned = "\n".join(output)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, removed


GENERIC_PUBLICATION_HEADINGS = {
    "研究现状",
    "相关研究",
    "国内研究",
    "国外研究",
    "研究方法",
    "方法",
    "讨论",
    "其他",
    "research status",
    "related work",
    "methods",
    "discussion",
    "other",
}

ALLOWED_HEADING_RELATIONS = {
    "independent",
    "parallel",
    "progressive",
    "contrast",
    "causal_deepening",
    "evidence_escalation",
    "scope_narrowing",
    "boundary_extension",
    "disconnected",
}


def _normalize_heading(value: str) -> str:
    value = re.sub(r"^\d+(?:\.\d+)*[.、．)]\s*", "", value.strip())
    return re.sub(r"\s+", "", value).casefold()


def _thematic_sections(text: str) -> list[dict[str, str]]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    excluded = {
        "引言",
        "总结与展望",
        "结论与展望",
        "参考文献",
        "introduction",
        "conclusion and outlook",
        "conclusion",
        "references",
    }
    sections: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        if _normalize_heading(heading) in {
            _normalize_heading(value) for value in excluded
        }:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append({"heading": heading, "body": text[match.end() : end].strip()})
    return sections


def heading_architecture_report(themes: list[Theme]) -> dict[str, Any]:
    headings = [theme.publication_heading or theme.name for theme in themes]
    normalized = [_normalize_heading(value) for value in headings]
    generic = [
        value
        for value in headings
        if _normalize_heading(value)
        in {_normalize_heading(item) for item in GENERIC_PUBLICATION_HEADINGS}
    ]
    missing_metadata = [
        theme.theme_id
        for index, theme in enumerate(themes)
        if not (theme.publication_heading or theme.name).strip()
        or not theme.heading_function.strip()
        or theme.relation_to_previous not in ALLOWED_HEADING_RELATIONS
        or not theme.heading_relation_rationale.strip()
        or (index and not theme.inherited_question.strip())
        or not theme.next_question.strip()
        or (theme.relation_to_previous == "parallel" and not theme.parallel_group.strip())
    ]
    question_headings = [value for value in headings if value.rstrip().endswith(("?", "？"))]
    checks = {
        "all_publication_headings_present": bool(headings) and all(normalized),
        "publication_headings_unique": len(normalized) == len(set(normalized)),
        "no_generic_publication_headings": not generic,
        "relation_metadata_complete": not missing_metadata,
        "question_headings_used_selectively": len(question_headings) <= max(1, len(headings) // 2),
    }
    parallel_groups: dict[str, list[str]] = {}
    for theme in themes:
        if theme.parallel_group:
            parallel_groups.setdefault(theme.parallel_group, []).append(theme.theme_id)
    return {
        "themes": [
            {
                "theme_id": theme.theme_id,
                "publication_heading": theme.publication_heading or theme.name,
                "heading_function": theme.heading_function,
                "relation_to_previous": theme.relation_to_previous,
                "relation_rationale": theme.heading_relation_rationale,
                "parallel_group": theme.parallel_group,
                "inherited_question": theme.inherited_question,
                "next_question": theme.next_question,
                "quality_status": theme.heading_quality_status,
            }
            for theme in themes
        ],
        "parallel_groups": parallel_groups,
        "checks": checks,
        "automatic_checks_pass": all(checks.values()),
        "generic_headings": generic,
        "missing_relation_metadata_theme_ids": missing_metadata,
        "needs_human_review": [
            "parallel headings use the same substantive object and abstraction level",
            "progressive headings genuinely raise the evidentiary burden",
            "the complete heading sequence is proportionate to the target journal and topic",
        ],
        "publication_boundary": "This architecture is backstage audit material, not review-body prose.",
    }


EVIDENCE_CLAIM_PATTERNS = {
    "mechanism": (
        r"mechanism|caus(?:e|al)|formation (?:mechanism|process)|"
        r"persistence (?:mechanism|driver)|transition (?:mechanism|driver)|"
        r"作用机制|形成机制|持续机制|转换机制|传导机制|驱动因素|导致|因果"
    ),
    "real_time_inference": r"real[- ]?time|online|nowcast|实时|在线|当期识别|事前识别",
    "prediction": r"predict|forecast|out[- ]?of[- ]?sample|预测|样本外",
    "incremental_prediction": r"incremental|beyond (?:a |the )?baseline|增量|优于基准|超越基准",
    "decision_change": r"decision|allocation|policy change|portfolio|决策|配置|政策改变",
    "net_value": r"net value|transaction cost|turnover|utility|净价值|交易成本|换手|效用",
    "external_validity": r"external validity|generaliz|replicat|cross[- ]?(?:market|country|setting)|外部有效|推广|复现|跨市场|跨国家",
}


def _section_evidence_stage_overreach(
    body: str, papers: list[Paper]
) -> list[dict[str, Any]]:
    by_citation = {
        paper.citation_number: paper
        for paper in papers
        if paper.citation_number is not None
    }
    issues: list[dict[str, Any]] = []
    for sentence in re.split(r"(?<=[。！？!?])\s*|\n+", body):
        if len(sentence.strip()) < 18:
            continue
        numbers = [
            int(number)
            for group in re.findall(r"\[((?:\d+\s*[,;]?\s*)+)\]", sentence)
            for number in re.findall(r"\d+", group)
        ]
        cited = [by_citation[number] for number in numbers if number in by_citation]
        for stage, pattern in EVIDENCE_CLAIM_PATTERNS.items():
            if re.search(pattern, sentence, re.IGNORECASE) and (
                not cited or not any(stage in paper.evidence_stages for paper in cited)
            ):
                issues.append(
                    {
                        "stage": stage,
                        "sentence": sentence.strip()[:240],
                        "cited_paper_ids": [paper.paper_id for paper in cited],
                    }
                )
    return issues[:12]


def argument_coherence_audit(
    text: str, papers: list[Paper], themes: list[Theme]
) -> dict[str, Any]:
    """Run deterministic structure checks and reserve semantic claims for review."""
    architecture = heading_architecture_report(themes)
    sections = _thematic_sections(text)
    expected = [theme.publication_heading or theme.name for theme in themes]
    observed = [section["heading"] for section in sections]
    observed_norm = [_normalize_heading(value) for value in observed]
    expected_norm = [_normalize_heading(value) for value in expected]
    by_heading = {
        _normalize_heading(section["heading"]): section for section in sections
    }
    section_reports = []
    obvious_overreach: list[dict[str, Any]] = []
    for index, theme in enumerate(themes):
        heading = theme.publication_heading or theme.name
        section = by_heading.get(_normalize_heading(heading))
        body = section["body"] if section else ""
        citations = sorted(set(citation_counts(body)))
        paragraphs = [
            value.strip()
            for value in re.split(r"\n\s*\n", body)
            if value.strip() and not value.lstrip().startswith("#")
        ]
        comparison = bool(
            re.search(
                r"however|by contrast|whereas|but|complement|tension|然而|但是|相比|相反|互补|分歧|张力|边界",
                body,
                re.IGNORECASE,
            )
        )
        boundary = bool(
            re.search(
                r"limit|boundary|cannot|unclear|not sufficient|局限|边界|不足|尚不能|无法|仍缺",
                body,
                re.IGNORECASE,
            )
        )
        handoff = bool(
            re.search(
                r"next question|therefore.*question|remains unresolved|下一|转向|问题|仍需回答|尚待",
                " ".join(paragraphs[-1:]),
                re.IGNORECASE,
            )
        )
        overreach = _section_evidence_stage_overreach(body, papers) if body else []
        obvious_overreach.extend(
            {"theme_id": theme.theme_id, **issue} for issue in overreach
        )
        required_sources = min(2, len(set(theme.paper_ids)))
        automatic = {
            "expected_heading_present": section is not None,
            "section_has_substantive_paragraphs": len(paragraphs) >= 2,
            "section_has_citations": bool(citations),
            "grouped_evidence_minimum": len(citations) >= required_sources,
            "comparison_or_tension_marker_present": comparison,
            "evidence_boundary_marker_present": boundary,
            "next_question_handoff_marker_present": handoff,
            "no_obvious_evidence_stage_overreach": not overreach,
        }
        section_reports.append(
            {
                "theme_id": theme.theme_id,
                "section_heading": heading,
                "relation_to_previous": theme.relation_to_previous,
                "automatic_checks": automatic,
                "automatic_checks_pass": all(automatic.values()),
                "paragraph_count": len(paragraphs),
                "distinct_citations": citations,
                "evidence_stage_overreach": overreach,
                "semantic_checks": {
                    "heading_content_alignment": "needs_human_review",
                    "inherited_question_answered": "needs_human_review",
                    "synthesis_claim_supported": "needs_human_review",
                    "relation_realized_in_content": (
                        "pass" if index == 0 else "needs_human_review"
                    ),
                    "parallel_or_progressive_granularity": (
                        "not_applicable"
                        if theme.relation_to_previous in {"independent", "disconnected"}
                        else "needs_human_review"
                    ),
                    "recent_chinese_top_journal_and_ml_evidence_integrated_not_decorative": "needs_human_review",
                },
                "revision_instruction": (
                    "Revise this section to restore the heading-to-claim chain and resolve failed automatic checks."
                    if not all(automatic.values())
                    else "No automatic rewrite required; complete semantic human review."
                ),
            }
        )
    checks = {
        "heading_architecture_automatic_checks_pass": architecture["automatic_checks_pass"],
        "all_expected_theme_sections_present": all(value in observed_norm for value in expected_norm),
        "theme_section_order_matches_architecture": [
            value for value in observed_norm if value in expected_norm
        ]
        == expected_norm,
        "no_duplicate_theme_headings": len(observed_norm) == len(set(observed_norm)),
        "all_theme_sections_pass_structural_checks": bool(section_reports)
        and all(row["automatic_checks_pass"] for row in section_reports),
        "no_obvious_evidence_stage_overreach": not obvious_overreach,
    }
    return {
        "audit_scope": "heading_and_section_argument_coherence",
        "heading_architecture": architecture,
        "expected_theme_headings": expected,
        "observed_theme_headings": observed,
        "sections": section_reports,
        "automatic_checks": checks,
        "automatic_checks_pass": all(checks.values()),
        "rewrite_required": not all(checks.values()),
        "needs_human_review": True,
        "semantic_status": "needs_human_review",
        "semantic_review_reasons": [
            "citation presence does not prove semantic support",
            "heading-content alignment and inherited-question fulfillment require contextual judgment",
            "parallelism, progression, and decorative integration cannot be safely passed by lexical checks alone",
        ],
        "obvious_evidence_stage_overreach": obvious_overreach,
        "publication_boundary": "Keep this audit outside the publication-facing review body.",
    }


def publication_quality_report(
    text: str,
    themes: list[Theme] | None = None,
    argument_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_heading = re.search(
        r"^##\s+(?:参考文献|References)\s*$", text, flags=re.MULTILINE | re.IGNORECASE
    )
    body = text[: reference_heading.start()] if reference_heading else text
    references = text[reference_heading.end() :] if reference_heading else ""
    table_lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]
    blockquotes = [line.strip() for line in body.splitlines() if line.strip().startswith(">")]
    forbidden = re.compile("|".join(PUBLICATION_FORBIDDEN_PATTERNS), re.IGNORECASE)
    process_hits = sorted(set(match.group(0) for match in forbidden.finditer(body)))
    body_citations = set(citation_counts(body))
    reference_numbers = {
        int(value)
        for value in re.findall(r"^\[(\d+)\]", references, flags=re.MULTILINE)
    }
    missing_reference_entries = sorted(body_citations - reference_numbers)
    uncited_reference_entries = sorted(reference_numbers - body_citations)
    paragraphs = [
        re.sub(r"\s+", " ", value).strip()
        for value in re.split(r"\n\s*\n", body)
        if value.strip() and not value.lstrip().startswith("#")
    ]
    cited_paragraphs = [value for value in paragraphs if citation_counts(value)]
    single_source_paragraphs = [
        value[:260]
        for value in cited_paragraphs
        if len(set(citation_counts(value))) == 1
    ]
    author_ledger_sentences = [
        sentence.strip()[:260]
        for paragraph in paragraphs
        for sentence in re.split(r"(?<=[。！？.!?])\s*", paragraph)
        if len(
            re.findall(
                r"(?:[A-Z][A-Za-z'’-]+(?:\s+(?:et\s+al\.|等))?|[\u4e00-\u9fff]{2,4}等)"
                r".{0,20}(?:发现|提出|认为|指出|find|show|propose|argue)",
                sentence,
                re.IGNORECASE,
            )
        )
        >= 2
    ]
    multi_source_ratio = (
        round(1 - len(single_source_paragraphs) / len(cited_paragraphs), 4)
        if cited_paragraphs
        else 0.0
    )
    checks = {
        "no_markdown_tables": not table_lines,
        "no_blockquote_disclaimers": not blockquotes,
        "no_backstage_process_language": not process_hits,
        "all_citations_have_reference_entries": not missing_reference_entries,
        "all_reference_entries_are_cited": not uncited_reference_entries,
        "no_obvious_author_ledger_sentences": not author_ledger_sentences,
    }
    heading_report = heading_architecture_report(themes) if themes is not None else None
    if heading_report is not None:
        checks["heading_architecture_automatic_checks_pass"] = heading_report[
            "automatic_checks_pass"
        ]
    if argument_audit is not None:
        checks["argument_coherence_automatic_checks_pass"] = bool(
            argument_audit.get("automatic_checks_pass", False)
        )
    return {
        "checks": checks,
        "publication_ready_automatic_checks": all(checks.values()),
        "table_lines": table_lines,
        "blockquote_lines": blockquotes,
        "backstage_process_hits": process_hits,
        "missing_reference_entries": missing_reference_entries,
        "uncited_reference_entries": uncited_reference_entries,
        "cited_paragraph_count": len(cited_paragraphs),
        "multi_source_cited_paragraph_ratio": multi_source_ratio,
        "single_source_cited_paragraphs": single_source_paragraphs,
        "obvious_author_ledger_sentences": author_ledger_sentences,
        "heading_architecture": heading_report,
        "argument_coherence_summary": (
            {
                "automatic_checks_pass": argument_audit.get("automatic_checks_pass"),
                "rewrite_required": argument_audit.get("rewrite_required"),
                "needs_human_review": argument_audit.get("needs_human_review"),
            }
            if argument_audit is not None
            else None
        ),
        "synthesis_note": (
            "The ratio is a diagnostic, not a hard quality verdict; some precise claims "
            "legitimately rely on one primary source."
        ),
        "note": "Automatic form checks do not replace human claim-to-source verification.",
    }


def sanitize_citations(text: str, allowed: set[int | None]) -> tuple[str, list[int]]:
    """Compatibility helper. New pipeline rejects whole sentences instead."""
    valid = {int(value) for value in allowed if value is not None}
    unknown: list[int] = []

    def replace(match: re.Match[str]) -> str:
        numbers = [int(value) for value in re.findall(r"\d+", match.group(0))]
        kept = [number for number in numbers if number in valid]
        unknown.extend(number for number in numbers if number not in valid)
        return "[" + ", ".join(map(str, kept)) + "]" if kept else ""

    return re.sub(r"\[(?:\d+\s*[,;–-]?\s*)+\]", replace, text), sorted(set(unknown))


def reject_unknown_citation_sentences(
    text: str, allowed: set[int | None]
) -> tuple[str, list[str]]:
    valid = {int(value) for value in allowed if value is not None}
    rejected: list[str] = []
    output_blocks: list[str] = []
    for block in text.split("\n\n"):
        if block.lstrip().startswith("#"):
            output_blocks.append(block)
            continue
        kept_sentences: list[str] = []
        for sentence in re.split(r"(?<=[.!?。！？])\s+", block):
            cited = [
                int(value)
                for group in re.findall(r"\[((?:\d+\s*[,;–-]?\s*)+)\]", sentence)
                for value in re.findall(r"\d+", group)
            ]
            if any(number not in valid for number in cited):
                rejected.append(sentence.strip())
            elif sentence.strip():
                kept_sentences.append(sentence.strip())
        if kept_sentences:
            output_blocks.append(" ".join(kept_sentences))
    return "\n\n".join(output_blocks), rejected


def build_claim_evidence_map(
    review_text: str, papers: list[Paper]
) -> list[ClaimEvidence]:
    by_citation = {
        paper.citation_number: paper for paper in papers if paper.citation_number
    }
    claims: list[ClaimEvidence] = []
    body = review_text.split("## 参考文献", 1)[0]
    sentences = [
        value.strip()
        for value in re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n+", body)
        if value.strip() and not value.lstrip().startswith("#")
    ]
    for sentence in sentences:
        numbers = [
            int(value)
            for group in re.findall(r"\[((?:\d+\s*[,;]?\s*)+)\]", sentence)
            for value in re.findall(r"\d+", group)
        ]
        if not numbers:
            continue
        supporting = [by_citation[number] for number in numbers if number in by_citation]
        spans = [span for paper in supporting for span in paper.evidence_spans[:4]]
        locations = []
        for span in spans:
            section = str(span.get("section") or "unknown")
            page = span.get("page")
            location = f"{section}:p{page}" if page else section
            if location not in locations:
                locations.append(location)
        bases = {paper.extraction_basis for paper in supporting}
        if spans and bases <= {"fulltext_complete", "section_complete"}:
            support_type = "section_text_bound"
            confidence = 0.8
        elif spans:
            support_type = "abstract_or_partial_text_bound"
            confidence = 0.55
        else:
            support_type = "citation_only_no_span"
            confidence = 0.25
        claims.append(
            ClaimEvidence(
                claim_id=f"C{len(claims) + 1:04d}",
                claim_text=sentence,
                supporting_paper_ids=[paper.paper_id for paper in supporting],
                supporting_evidence_spans=spans,
                evidence_page_or_section=locations,
                support_type=support_type,
                confidence=confidence,
            )
        )
    return claims


def render_references(
    papers: list[Paper], *, cited_numbers: set[int] | None = None
) -> str:
    ordered = sorted(
        [
            paper
            for paper in papers
            if paper.citation_number
            and (cited_numbers is None or paper.citation_number in cited_numbers)
        ],
        key=lambda paper: paper.citation_number or 0,
    )
    lines = ["## 参考文献", ""]
    for paper in ordered:
        authors = ", ".join(paper.authors[:8]) or "Unknown author"
        if len(paper.authors) > 8:
            authors += ", et al."
        year = paper.year or "n.d."
        venue = paper.venue or "Unknown venue"
        locator = (
            f"https://doi.org/{paper.doi}"
            if paper.doi
            else (paper.url or paper.open_access_pdf)
        )
        suffix = f" {locator}" if locator else ""
        lines.append(
            f"[{paper.citation_number}] {authors}. ({year}). {paper.title}. "
            f"*{venue}*.{suffix}"
        )
    return "\n".join(lines)


def render_bibtex(papers: list[Paper]) -> str:
    lines = []
    for paper in sorted(
        [p for p in papers if p.citation_number],
        key=lambda p: p.citation_number or 0,
    ):
        first = "paper"
        if paper.authors:
            first = re.sub(r"\W+", "", paper.authors[0].split()[-1]) or "paper"
        key = f"{first}{paper.year or 'nd'}_{paper.citation_number}"
        entry_type = "article" if paper.venue and paper.venue != "arXiv" else "misc"
        lines.append(f"@{entry_type}{{{key},")
        lines.append(f"  title = {{{paper.title}}},")
        if paper.authors:
            lines.append(f"  author = {{{' and '.join(paper.authors)}}},")
        if paper.year:
            lines.append(f"  year = {{{paper.year}}},")
        if paper.venue:
            lines.append(f"  journal = {{{paper.venue}}},")
        if paper.doi:
            lines.append(f"  doi = {{{paper.doi}}},")
        if paper.url:
            lines.append(f"  url = {{{paper.url}}},")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def citation_counts(text: str) -> Counter[int]:
    values = [
        int(value)
        for group in re.findall(r"\[((?:\d+\s*[,;]?\s*)+)\]", text)
        for value in re.findall(r"\d+", group)
    ]
    return Counter(values)


def cite(numbers: list[int | None]) -> str:
    clean = list(dict.fromkeys(int(number) for number in numbers if number))
    return "[" + ", ".join(map(str, clean)) + "]" if clean else ""


def _representative_items(
    papers: list[Paper], field: str, limit: int
) -> list[str]:
    values: list[str] = []
    for paper in papers:
        for item in getattr(paper, field):
            text = compact(item, 180)
            if text and text not in values:
                values.append(text)
    return values[:limit] or ["可用摘要未提供足够细节"]


def join_items(values: list[str]) -> str:
    if not values:
        return "可用材料未明确说明"
    if len(values) == 1:
        return values[0]
    return "；".join(values[:-1]) + "；以及" + values[-1]
