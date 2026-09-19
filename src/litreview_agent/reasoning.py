from __future__ import annotations

import io
import json
import math
import re
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from .config import PipelineConfig
from .io_utils import compact
from .llm import LLMError, OpenAICompatibleClient
from .models import GapAudit, Paper, QueryVariant, Theme
from .prompts import (
    EXTRACT_PROMPT_VERSION,
    EXTRACT_SYSTEM,
    EXTRACT_USER,
    QUERY_PROMPT_VERSION,
    QUERY_SYSTEM,
    QUERY_USER,
    REFLECT_PROMPT_VERSION,
    REFLECT_SYSTEM,
    REFLECT_USER,
    SCREEN_PROMPT_VERSION,
    SCREEN_SYSTEM,
    SCREEN_USER,
    THEME_PROMPT_VERSION,
    THEME_SYSTEM,
    THEME_USER,
)
from .search import CachedHTTP, tokenize


def batched(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


HEADING_RELATIONS = {
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

GENERIC_HEADINGS = {
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


def _clean_publication_heading(value: str, fallback: str) -> tuple[str, str]:
    heading = re.sub(r"^#{1,6}\s*", "", value or "").strip()
    heading = re.sub(r"^\d+(?:\.\d+)*[.、．)]\s*", "", heading).strip()
    heading = heading.rstrip(":：。 ")
    status = "model_generated"
    if not heading or heading.casefold() in GENERIC_HEADINGS:
        heading = re.sub(r"^\d+(?:\.\d+)*[.、．)]\s*", "", fallback).strip()
        status = "fallback_from_theme_name"
    if not heading or heading.casefold() in GENERIC_HEADINGS:
        heading = "研究问题、证据分歧与适用边界"
        status = "needs_human_review"
    return compact(heading, 72), status


def _normalize_heading_relation(value: str, index: int) -> str:
    if index == 0:
        return "independent"
    normalized = (value or "").strip().casefold().replace(" ", "_")
    aliases = {
        "editorial_transition": "progressive",
        "evolution": "progressive",
        "complement": "parallel",
        "complementary": "parallel",
        "conflict": "contrast",
        "extension": "boundary_extension",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in HEADING_RELATIONS else "progressive"


def _infer_heading_function(theme: Theme, index: int, count: int) -> str:
    text = " ".join(
        [theme.name, theme.organizing_question, theme.synthesis_claim, theme.classification_basis]
    ).casefold()
    function_terms = [
        ("net_value", r"net value|transaction cost|utility|净价值|交易成本|效用"),
        ("decision_value", r"decision|allocation|policy|portfolio|决策|配置|政策"),
        ("prediction", r"predict|forecast|out.of.sample|预测|样本外"),
        ("real_time_inference", r"real.time|online|实时|在线|当期"),
        ("mechanism", r"mechanism|formation|persistence|transition|机制|形成|持续|转换"),
        ("identification", r"identif|measure|method|model|识别|测量|方法|模型"),
        ("external_validity", r"external|generaliz|replicat|boundary|外部|推广|复现|边界"),
        ("concept_boundary", r"concept|definition|theory|概念|界定|理论"),
    ]
    for function, pattern in function_terms:
        if re.search(pattern, text, re.IGNORECASE):
            return function
    return "synthesis_boundary" if index == count - 1 else "argument_development"


class Reasoner:
    def __init__(self, config: PipelineConfig, llm: OpenAICompatibleClient) -> None:
        self.config = config
        self.llm = llm
        self.fallbacks: list[str] = []

    def generate_queries(self) -> list[QueryVariant]:
        if self.llm.enabled:
            try:
                data = self.llm.complete_json(
                    stage="query_generation",
                    prompt_version=QUERY_PROMPT_VERSION,
                    system=QUERY_SYSTEM,
                    user=QUERY_USER.format(
                        topic=self.config.topic,
                        count=self.config.query_count,
                        tracks=", ".join(self.config.coverage_tracks),
                        recent_window=self._recent_window_text(),
                        preferred_venues=(
                            ", ".join(self.config.preferred_venues) or "not supplied"
                        ),
                    ),
                    temperature=0.35,
                )
                queries = []
                for item in data.get("queries", [])[: self.config.query_count]:
                    query = str(item.get("query", "")).strip()
                    if query:
                        queries.append(
                            QueryVariant(
                                query=query,
                                language=str(item.get("language", "mixed")),
                                facet=str(item.get("facet", "general")),
                                rationale=str(item.get("rationale", "")),
                            )
                        )
                if len(queries) >= 4:
                    return self._pad_queries(queries)
            except LLMError as exc:
                self.fallbacks.append(f"query_generation: {exc}")
        return self._heuristic_queries()

    def _pad_queries(self, values: list[QueryVariant]) -> list[QueryVariant]:
        seen = {item.query.casefold() for item in values}
        for item in self._heuristic_queries():
            if len(values) >= self.config.query_count:
                break
            if item.query.casefold() not in seen:
                values.append(item)
                seen.add(item.query.casefold())
        return values[: self.config.query_count]

    def _heuristic_queries(self) -> list[QueryVariant]:
        topic = self.config.topic.strip()
        lowered = topic.casefold()
        recent_window = self._recent_window_text()
        venue_hint = " ".join(self.config.preferred_venues[:8])
        if any(term in lowered for term in ("clinical", "patient", "disease", "治疗", "患者", "疾病", "临床")):
            variants = [
                (topic, "mixed", "population-condition", "人群与疾病/条件"),
                (f"{topic} intervention exposure comparator", "en", "intervention-comparator", "干预或暴露及对照"),
                (f"{topic} outcome efficacy safety", "en", "outcome", "结局、有效性与安全性"),
                (f"{topic} randomized cohort case control", "en", "study-design", "研究设计过滤"),
                (f"{topic} systematic review meta-analysis", "en", "review", "高层级证据与术语扩展"),
                (f"{topic} adverse effect heterogeneity limitation", "en", "counter-evidence", "不良结果、异质性与局限"),
                (f"{topic} {recent_window} recent advances", "en", "recent_frontier", "近五年前沿研究"),
                (f"{topic} 中文 核心期刊 临床研究", "zh", "chinese", "中文研究与本土情境"),
            ]
        elif any(term in lowered for term in ("algorithm", "model", "machine learning", "人工智能", "模型", "算法", "机器学习")):
            variants = [
                (topic, "mixed", "object-method", "对象与核心技术"),
                (f"{topic} survey taxonomy", "en", "foundations", "综述、术语与经典基础"),
                (f"{topic} method architecture dataset", "en", "method-data", "方法架构与数据"),
                (f"{topic} application deployment case study", "en", "context", "应用与部署场景"),
                (f"{topic} benchmark metric baseline ablation", "en", "evaluation", "基准、指标与消融"),
                (f"{topic} failure bias robustness limitation", "en", "counter-evidence", "失败、偏差与鲁棒性"),
                (f"{topic} {recent_window} state of the art", "en", "recent_frontier", "近五年前沿研究"),
                (f"{topic} 中文 核心期刊 评估", "zh", "chinese", "中文研究与本土应用"),
            ]
        else:
            variants = [
                (topic, "mixed", "core", "核心主题精确检索"),
                (f"{topic} theory construct review", "en", "theory", "理论、构念与综述"),
                (f"{topic} population context comparative", "en", "population-context", "人群与情境比较"),
                (f"{topic} empirical evidence mechanism", "en", "evidence-mechanism", "经验证据与机制"),
                (f"{topic} competing theory contradictory evidence", "en", "counter-evidence", "竞争解释与矛盾证据"),
                (f"{topic} measurement validity limitation", "en", "measurement", "测量效度与局限"),
                (f"{topic} {recent_window} recent evidence", "en", "recent_frontier", "近五年前沿研究"),
                (f"{topic} 中文 CSSCI 核心期刊 实证研究", "zh", "chinese", "中文研究与本土情境"),
            ]
        if venue_hint:
            variants.insert(
                1,
                (
                    f'{topic} "{venue_hint}"',
                    "en",
                    "preferred_venues",
                    "指定期刊的定向召回；不参与相关性加分",
                ),
            )
        facet_priority = {
            "core": 0,
            "population-condition": 0,
            "object-method": 0,
            "review": 1,
            "foundations": 1,
            "theory": 1,
            "recent_frontier": 2,
            "chinese": 3,
            "preferred_venues": 4,
        }
        variants.sort(key=lambda item: facet_priority.get(item[2], 10))
        return [
            QueryVariant(query=q, language=lang, facet=facet, rationale=why)
            for q, lang, facet, why in variants[: self.config.query_count]
        ]

    def _recent_window_text(self) -> str:
        end = self.config.year_to or dt.date.today().year
        start = max(0, end - max(1, self.config.recent_years) + 1)
        return f"{start}-{end}"

    def screen(self, papers: list[Paper]) -> list[Paper]:
        results: dict[str, dict[str, Any]] = {}
        if self.llm.enabled:
            try:
                for batch in batched(papers, 10):
                    payload = "\n".join(
                        f"{p.paper_id} | {p.title} | {p.year or 'n.d.'} | "
                        f"citations={p.citation_count} | {compact(p.abstract, 900)}"
                        for p in batch
                    )
                    data = self.llm.complete_json(
                        stage="screening",
                        prompt_version=SCREEN_PROMPT_VERSION,
                        system=SCREEN_SYSTEM,
                        user=SCREEN_USER.format(
                            topic=self.config.topic,
                            threshold=self.config.relevance_threshold,
                            papers=payload,
                        ),
                        temperature=0.1,
                    )
                    for item in data.get("results", []):
                        paper_id = str(item.get("paper_id", ""))
                        if paper_id:
                            results[paper_id] = item
            except LLMError as exc:
                self.fallbacks.append(f"screening: {exc}")
                results = {}

        for paper in papers:
            item = results.get(paper.paper_id)
            if item:
                try:
                    score = min(5.0, max(1.0, float(item.get("score", 1))))
                except (TypeError, ValueError):
                    score = self._heuristic_relevance(paper)
                paper.relevance_score = round(score, 2)
                paper.screening_reason = str(item.get("reason", "")).strip()
            else:
                paper.relevance_score = round(self._heuristic_relevance(paper), 2)
                paper.screening_reason = self._heuristic_reason(paper)
            paper.screening_decision = (
                "include"
                if paper.relevance_score >= self.config.relevance_threshold
                else "exclude"
            )
            if paper.is_retracted is True:
                paper.screening_decision = "exclude"
                paper.screening_reason += "；数据源标记为已撤稿，自动排除"
            self._annotate_coverage(paper)

        ordered = sorted(
            papers,
            key=lambda p: (
                p.relevance_score,
                self._coverage_priority(p),
                _title_overlap(p.title, self.config.topic),
            ),
            reverse=True,
        )
        included = [p for p in ordered if p.screening_decision == "include"]
        if self.config.allow_quantity_backfill and len(included) < self.config.min_selected:
            for paper in ordered:
                if paper in included:
                    continue
                paper.screening_decision = "boundary_backfill"
                paper.screening_reason += (
                    "；兼容课程数量要求的边界补入，不计入高相关证据集"
                )
                included.append(paper)
                if len(included) >= self.config.min_selected:
                    break
        chosen = self._select_for_coverage(included)
        final_ids = {p.paper_id for p in chosen}
        for paper in papers:
            if paper.paper_id not in final_ids and paper.screening_decision == "include":
                paper.screening_decision = "reserve"
        return [p for p in ordered if p.paper_id in final_ids]

    def _annotate_coverage(self, paper: Paper) -> None:
        end = self.config.year_to or dt.date.today().year
        recent_cutoff = end - max(1, self.config.recent_years) + 1
        tags: list[str] = []
        if paper.year is not None and paper.year >= recent_cutoff:
            tags.append("recent")
        if _is_chinese_paper(paper):
            tags.append("chinese")
        venue = paper.venue.casefold()
        paper.preferred_venue_match = any(
            value.strip().casefold() in venue
            for value in self.config.preferred_venues
            if value.strip()
        )
        if paper.preferred_venue_match:
            tags.append("preferred_venue")
        paper.coverage_tags = tags

    def _coverage_priority(self, paper: Paper) -> int:
        return sum(
            1
            for tag in ("recent", "chinese", "preferred_venue")
            if tag in paper.coverage_tags
        )

    def _select_for_coverage(self, included: list[Paper]) -> list[Paper]:
        """Meet corpus-diversity targets without changing relevance scores."""
        limit = self.config.target_selected
        chosen: list[Paper] = []

        def add_quota(tag: str, quota: int) -> None:
            if quota <= 0:
                return
            existing = sum(tag in paper.coverage_tags for paper in chosen)
            for paper in included:
                if existing >= quota or len(chosen) >= limit:
                    break
                if paper not in chosen and tag in paper.coverage_tags:
                    chosen.append(paper)
                    existing += 1

        add_quota("recent", self.config.min_recent_selected)
        add_quota("chinese", self.config.min_chinese_selected)
        add_quota("preferred_venue", self.config.min_preferred_venue_selected)
        for paper in included:
            if len(chosen) >= limit:
                break
            if paper not in chosen:
                chosen.append(paper)
        return chosen

    def _heuristic_relevance(self, paper: Paper) -> float:
        topic_tokens = set(tokenize(self.config.topic))
        title_tokens = set(tokenize(paper.title))
        text_tokens = set(tokenize(paper.title + " " + paper.abstract))
        query_token_sets = [set(tokenize(query)) for query in paper.matched_queries]
        query_overlap = max(
            (len(query_tokens & text_tokens) / max(1, len(query_tokens)) for query_tokens in query_token_sets),
            default=0.0,
        )
        query_title_overlap = max(
            (len(query_tokens & title_tokens) / max(1, len(query_tokens)) for query_tokens in query_token_sets),
            default=0.0,
        )
        if not topic_tokens:
            overlap = 0.3
            title_overlap = 0.0
        else:
            overlap = len(topic_tokens & text_tokens) / len(topic_tokens)
            title_overlap = len(topic_tokens & title_tokens) / len(topic_tokens)
        overlap = max(overlap, query_overlap)
        title_overlap = max(title_overlap, query_title_overlap)
        score = 1.2 + overlap * 2.4 + title_overlap * 0.8
        score += 0.0 if paper.abstract else -0.2
        score += min(0.15, len(paper.matched_queries) * 0.03)
        return min(5.0, max(1.0, score))

    def _heuristic_reason(self, paper: Paper) -> str:
        topic_tokens = set(tokenize(self.config.topic))
        matches = sorted(topic_tokens & set(tokenize(paper.title + " " + paper.abstract)))
        if matches:
            return "题名/摘要与主题共享核心概念：" + "、".join(matches[:5])
        return "直接语义证据不足；引用数与开放获取状态不参与相关性加分"

    def extract(self, papers: list[Paper]) -> None:
        results: dict[str, dict[str, Any]] = {}
        if self.llm.enabled:
            try:
                for batch in batched(papers, 5):
                    payload = "\n\n".join(
                        f"ID: {p.paper_id}\nTITLE: {p.title}\nBASIS: "
                        f"{p.extraction_basis}\n<UNTRUSTED_PAPER_TEXT>\n"
                        f"{compact(_paper_evidence_text(p), 11000)}\n"
                        "</UNTRUSTED_PAPER_TEXT>"
                        for p in batch
                    )
                    data = self.llm.complete_json(
                        stage="information_extraction",
                        prompt_version=EXTRACT_PROMPT_VERSION,
                        system=EXTRACT_SYSTEM,
                        user=EXTRACT_USER.format(
                            topic=self.config.topic, papers=payload
                        ),
                        temperature=0.1,
                    )
                    for item in data.get("results", []):
                        paper_id = str(item.get("paper_id", ""))
                        if paper_id:
                            results[paper_id] = item
            except LLMError as exc:
                self.fallbacks.append(f"information_extraction: {exc}")
                results = {}

        for paper in papers:
            item = results.get(paper.paper_id)
            if item:
                paper.research_question = str(
                    item.get("research_question", "")
                ).strip()
                paper.methods = _string_list(item.get("methods"))
                paper.findings = _string_list(item.get("findings"))[:3]
                paper.limitations = _string_list(item.get("limitations"))
                paper.population_sample = str(item.get("population_sample", "")).strip()
                paper.geography_context = str(item.get("geography_context", "")).strip()
                paper.data_period = str(item.get("data_period", "")).strip()
                paper.study_design = str(item.get("study_design", "")).strip()
                paper.variables_operationalization = _string_list(item.get("variables_operationalization"))
                paper.baselines = _string_list(item.get("baselines"))
                paper.metrics = _string_list(item.get("metrics"))
                paper.validation_setting = str(item.get("validation_setting", "")).strip()
                paper.effect_estimates = _string_list(item.get("effect_estimates"))
                paper.causal_identification = str(item.get("causal_identification", "")).strip()
                paper.boundary_conditions = _string_list(item.get("boundary_conditions"))
                paper.author_conclusions = _string_list(item.get("author_conclusions"))
                paper.analyst_inferences = _string_list(item.get("analyst_inferences"))
                paper.rhetorical_role = str(item.get("rhetorical_role", "")).strip()
                paper.concept_level = str(item.get("concept_level", "")).strip()
                paper.evidence_stages = _valid_evidence_stages(
                    item.get("evidence_stages")
                )
                paper.change_representation = str(
                    item.get("change_representation", "")
                ).strip()
                paper.mechanism = str(item.get("mechanism", "")).strip()
                paper.temporal_design = str(
                    item.get("temporal_design", "")
                ).strip()
                paper.state_output = str(item.get("state_output", "")).strip()
                paper.real_time_vintage = str(
                    item.get("real_time_vintage", "")
                ).strip()
                paper.out_of_sample = str(item.get("out_of_sample", "")).strip()
                paper.uncertainty = _string_list(item.get("uncertainty"))
                paper.costs_constraints = _string_list(
                    item.get("costs_constraints")
                )
                paper.cross_setting_test = str(
                    item.get("cross_setting_test", "")
                ).strip()
                paper.evidence_spans = _evidence_spans(item.get("evidence_spans"))
                paper.topic_value = str(item.get("topic_value", "")).strip()
                basis = str(item.get("evidence_basis", paper.extraction_basis))
                paper.extraction_basis = (
                    basis
                    if basis in {
                        "abstract_only", "partial_fulltext", "section_complete",
                        "fulltext_complete", "pdf_parse_failed",
                        "ocr_or_layout_unreliable",
                    }
                    else paper.extraction_basis
                )
            else:
                self._heuristic_extract(paper)

    def _heuristic_extract(self, paper: Paper) -> None:
        text = _paper_evidence_text(paper)
        sentences = split_sentences(text)
        paper.research_question = (
            sentences[0] if sentences else "可用元数据未明确说明研究问题"
        )
        method_words = re.compile(
            r"\b(method|model|framework|approach|algorithm|dataset|experiment|survey|"
            r"propose|using|采用|方法|模型|框架|实验|调查)\b",
            re.IGNORECASE,
        )
        finding_words = re.compile(
            r"\b(result|find|show|demonstrate|improve|outperform|evidence|"
            r"结果|发现|表明|提升|证据)\b",
            re.IGNORECASE,
        )
        limit_words = re.compile(
            r"\b(limit|challenge|future|however|remain|不足|局限|挑战|未来)\b",
            re.IGNORECASE,
        )
        paper.methods = [s for s in sentences if method_words.search(s)][:2]
        paper.findings = [s for s in sentences if finding_words.search(s)][:3]
        paper.limitations = [s for s in sentences if limit_words.search(s)][:2]
        if not paper.methods:
            paper.methods = ["摘要未明确给出可可靠抽取的方法细节"]
        if not paper.findings:
            paper.findings = ["摘要未明确给出可可靠抽取的主要发现"]
        if not paper.limitations:
            paper.limitations = ["摘要未报告作者明确陈述的局限"]
        paper.topic_value = (
            f"该文以“{paper.title}”为对象，为主题的概念、方法或经验证据提供材料；"
            "具体支持范围需以全文复核。"
        )
        paper.rhetorical_role = "empirical_or_conceptual_evidence"
        paper.concept_level = "unclear/verify"
        paper.evidence_stages = _heuristic_evidence_stages(text)
        paper.change_representation = "unclear/verify"
        paper.mechanism = "unclear/verify"
        paper.temporal_design = "unclear/verify"
        paper.state_output = "unclear/verify"
        paper.real_time_vintage = "unclear/verify"
        paper.out_of_sample = "unclear/verify"
        paper.uncertainty = ["unclear/verify"]
        paper.costs_constraints = ["unclear/verify"]
        paper.cross_setting_test = "unclear/verify"
        paper.study_design = _first_matching_sentence(sentences, r"\b(random|cohort|survey|experiment|case study|review|meta-analysis|实验|调查|队列|综述)\b")
        paper.population_sample = _first_matching_sentence(sentences, r"\b(sample|participant|student|patient|city|dataset|样本|参与者|学生|患者|城市|数据集)\b")
        paper.metrics = [_first_matching_sentence(sentences, r"\b(metric|accuracy|f1|auc|rmse|effect size|指标|准确率|效应量)\b")]
        paper.metrics = [value for value in paper.metrics if value]
        for field_name, values in (("research_question", [paper.research_question]), ("methods", paper.methods), ("findings", paper.findings), ("limitations", paper.limitations)):
            for value in values[:3]:
                if value and "未明确" not in value and "未报告" not in value:
                    paper.evidence_spans.append({
                        "field": field_name,
                        "section": _locate_section(paper, value),
                        "page": None,
                        "text": compact(value, 420),
                        "attribution": "author" if field_name != "research_question" else "agent_extract",
                        "confidence": 0.55 if paper.extraction_basis == "abstract_only" else 0.7,
                    })

    def organize_themes(self, papers: list[Paper]) -> list[Theme]:
        if self.llm.enabled:
            try:
                payload = "\n".join(
                    f"{p.paper_id} | {p.title} | Q={compact(p.research_question, 220)} "
                    f"| M={compact('; '.join(p.methods), 260)} "
                    f"| F={compact('; '.join(p.findings), 300)} "
                    f"| L={compact('; '.join(p.limitations), 200)} "
                    f"| concept={p.concept_level} "
                    f"| stages={','.join(p.evidence_stages)} "
                    f"| mechanism={compact(p.mechanism, 140)} "
                    f"| OOS={p.out_of_sample} | costs={compact('; '.join(p.costs_constraints), 120)}"
                    for p in papers
                )
                data = self.llm.complete_json(
                    stage="theme_organization",
                    prompt_version=THEME_PROMPT_VERSION,
                    system=THEME_SYSTEM,
                    user=THEME_USER.format(
                        topic=self.config.topic,
                        theme_min=self.config.theme_min,
                        theme_max=self.config.theme_max,
                        papers=payload,
                    ),
                    temperature=0.2,
                )
                themes = self._parse_themes(data, papers)
                if self.config.theme_min <= len(themes) <= self.config.theme_max:
                    return self._finalize_heading_architecture(themes)
            except LLMError as exc:
                self.fallbacks.append(f"theme_organization: {exc}")
        return self._heuristic_themes(papers)

    def _parse_themes(
        self, data: dict[str, Any], papers: list[Paper]
    ) -> list[Theme]:
        valid_ids = {p.paper_id for p in papers}
        themes: list[Theme] = []
        for index, item in enumerate(data.get("themes", []), start=1):
            ids = [str(x) for x in item.get("paper_ids", []) if str(x) in valid_ids]
            if not ids:
                continue
            themes.append(
                Theme(
                    theme_id=f"T{index}",
                    name=str(item.get("name", f"主题{index}")),
                    organizing_question=str(item.get("organizing_question", "")),
                    synthesis_claim=str(item.get("synthesis_claim", "")),
                    paper_ids=list(dict.fromkeys(ids)),
                    classification_basis=str(item.get("classification_basis", "")),
                    internal_relations=_string_list(item.get("internal_relations")),
                    inherited_question=str(item.get("inherited_question", "")),
                    residual_gap=str(item.get("residual_gap", "")),
                    next_question=str(item.get("next_question", "")),
                    relation_to_previous=str(item.get("relation_to_previous", "independent")),
                    relation_evidence_paper_ids=[
                        str(value)
                        for value in item.get("relation_evidence_paper_ids", [])
                        if str(value) in valid_ids
                    ],
                    relation_support_status=str(item.get("relation_support_status", "not_verified")),
                    publication_heading=str(item.get("publication_heading", "")),
                    heading_function=str(item.get("heading_function", "")),
                    heading_relation_rationale=str(
                        item.get("heading_relation_rationale", "")
                    ),
                    parallel_group=str(item.get("parallel_group", "")),
                )
            )
        self._apply_theme_ids(papers, themes)
        return themes

    def _heuristic_themes(self, papers: list[Paper]) -> list[Theme]:
        categories = [
            {
                "name": "概念基础与问题界定",
                "heading": "研究对象的概念边界与解释起点",
                "function": "concept_boundary",
                "relation": "independent",
                "relation_rationale": "首节区分研究对象、概念层级与竞争性解释，为后续经验证据设定边界。",
                "terms": {
                    "review",
                    "concept",
                    "theory",
                    "theoretical",
                    "definition",
                    "perspective",
                    "综述",
                    "概念",
                    "理论",
                    "框架",
                    "界定",
                },
                "question": f"关于“{self.config.topic}”，现有研究首先确立了哪些对象、概念与核心问题？",
                "claim": "该组文献界定了研究对象、问题边界与主要解释视角，为后续方法比较提供共同起点。",
                "basis": "题名或摘要主要承担概念界定、理论讨论、领域综述或问题框定功能",
                "gap": "概念共识尚不能说明何种数据与方法最适合回答这些问题。",
                "next": "既然问题边界已经明确，研究者如何把它转化为可操作的数据与方法？",
            },
            {
                "name": "数据、方法与技术路线",
                "heading": "可识别经验命题的构造",
                "function": "identification",
                "relation": "evidence_escalation",
                "relation_rationale": "承接概念边界，把抽象研究对象转化为数据、模型与可检验命题。",
                "terms": {
                    "method",
                    "model",
                    "algorithm",
                    "framework",
                    "taxonomy",
                    "perspective",
                    "dataset",
                    "data",
                    "machine",
                    "deep",
                    "network",
                    "regression",
                    "comparison",
                    "controlled",
                    "interview",
                    "survey",
                    "detection",
                    "benchmark",
                    "analysis",
                    "remote",
                    "sensing",
                    "方法",
                    "模型",
                    "算法",
                    "数据",
                    "技术",
                    "遥感",
                    "预测",
                },
                "question": "现有研究如何把核心问题转化为数据、模型与可执行的分析流程？",
                "claim": "该组文献提出或比较了数据来源、建模方法与技术流程，扩展了问题的可测量性。",
                "basis": "题名或摘要重点描述数据、模型、算法、测量工具或技术框架",
                "gap": "方法能力本身并不能保证在真实场景中产生稳定且可迁移的效果。",
                "next": "这些方法在实际应用和经验研究中表现如何，证据是否一致？",
            },
            {
                "name": "应用场景与经验证据",
                "heading": "应用情境中的效应异质性与条件依赖",
                "function": "prediction",
                "relation": "evidence_escalation",
                "relation_rationale": "在可识别性基础上检验方法表现是否跨场景成立，并提高经验有效性的证据要求。",
                "terms": {
                    "application",
                    "empirical",
                    "effect",
                    "impact",
                    "outcome",
                    "performance",
                    "case",
                    "experiment",
                    "student",
                    "teaching",
                    "city",
                    "climate",
                    "temperature",
                    "应用",
                    "效果",
                    "影响",
                    "实证",
                    "实验",
                    "案例",
                    "学生",
                    "教学",
                },
                "question": "不同应用场景中的经验证据如何支持或修正方法层面的预期？",
                "claim": "该组研究把方法置于具体对象与场景中检验，展示了效果的异质性和条件依赖。",
                "basis": "题名或摘要以应用案例、经验比较、效果评估或具体场景为中心",
                "gap": "场景化结果的差异说明，平均效果不足以界定方法的适用边界与外部有效性。",
                "next": "哪些风险、偏差与评价缺口限制了这些结果的推广？",
            },
            {
                "name": "风险、局限与评价边界",
                "heading": "证据边界、评价有效性与跨情境推广",
                "function": "external_validity",
                "relation": "boundary_extension",
                "relation_rationale": "由局部结果转向风险、评价设计与外部有效性，明确结论可以推广到哪里。",
                "terms": {
                    "risk",
                    "limitation",
                    "challenge",
                    "ethics",
                    "ethical",
                    "privacy",
                    "governance",
                    "bias",
                    "evaluation",
                    "assessment",
                    "uncertainty",
                    "风险",
                    "局限",
                    "挑战",
                    "伦理",
                    "隐私",
                    "治理",
                    "偏差",
                    "评价",
                },
                "question": "现有证据在哪些风险、偏差、评价设计和外部有效性方面仍受限制？",
                "claim": "该组文献揭示了技术与应用证据的边界，并把研究议程推进到治理、比较评价和可靠性问题。",
                "basis": "题名或摘要明确讨论风险、伦理、治理、局限、挑战或评价有效性",
                "gap": "现有研究仍缺少跨方法、跨数据和跨场景的一致评价框架。",
                "next": "未来如何建立可复现、可比较且能反映真实边界条件的证据体系？",
            },
        ]
        count = min(self.config.theme_max, len(categories))
        categories = categories[:count]
        groups: dict[int, list[Paper]] = defaultdict(list)
        topic_tokens = set(tokenize(self.config.topic))
        for paper in papers:
            tokens = set(tokenize(paper.title + " " + paper.abstract)) - topic_tokens
            raw_scores = [
                len(tokens & category["terms"]) for category in categories
            ]
            best_score = max(raw_scores, default=0)
            if best_score <= 0:
                continue
            for category_index, score in enumerate(raw_scores):
                if score == best_score or (best_score >= 2 and score >= best_score - 1):
                    groups[category_index].append(paper)

        themes = []
        previous_question = categories[0]["question"]
        active_categories = [index for index in range(count) if groups[index]]
        for output_index, index in enumerate(active_categories):
            category = categories[index]
            ids = [paper.paper_id for paper in groups[index]]
            themes.append(
                Theme(
                    theme_id=f"T{output_index + 1}",
                    name=category["name"],
                    publication_heading=category["heading"],
                    heading_function=category["function"],
                    organizing_question=category["question"],
                    synthesis_claim=category["claim"],
                    paper_ids=ids,
                    classification_basis=category["basis"],
                    internal_relations=[
                        "按发表年份观察概念、方法或应用证据的演进",
                        "比较不同研究对象、数据来源和评价指标形成的互补与差异",
                    ],
                    inherited_question=(previous_question if output_index else ""),
                    residual_gap=category["gap"],
                    next_question=category["next"],
                    relation_to_previous=(
                        "independent" if output_index == 0 else category["relation"]
                    ),
                    heading_relation_rationale=category["relation_rationale"],
                    relation_evidence_paper_ids=[],
                    relation_support_status="heuristic_not_evidence_verified",
                )
            )
            previous_question = category["next"]
        self._apply_theme_ids(papers, themes)
        return self._finalize_heading_architecture(themes)

    def _finalize_heading_architecture(self, themes: list[Theme]) -> list[Theme]:
        """Normalize publication headings without changing semantic clustering."""
        seen: set[str] = set()
        for index, theme in enumerate(themes):
            heading, status = _clean_publication_heading(
                theme.publication_heading, theme.name
            )
            key = re.sub(r"\s+", "", heading).casefold()
            if key in seen:
                heading = f"{heading}：{compact(theme.organizing_question, 24)}"
                key = re.sub(r"\s+", "", heading).casefold()
                status = "needs_human_review"
            seen.add(key)
            theme.publication_heading = heading
            theme.heading_quality_status = status
            theme.relation_to_previous = _normalize_heading_relation(
                theme.relation_to_previous, index
            )
            if not theme.heading_function:
                theme.heading_function = _infer_heading_function(
                    theme, index, len(themes)
                )
            if index == 0:
                theme.heading_relation_rationale = (
                    theme.heading_relation_rationale
                    or "首节建立研究对象、概念边界或初始证据问题。"
                )
                theme.parallel_group = ""
            else:
                theme.heading_relation_rationale = (
                    theme.heading_relation_rationale
                    or f"承接前节留下的“{compact(theme.inherited_question or theme.organizing_question, 80)}”，"
                    f"将讨论推进到{theme.heading_function}层面。"
                )
                if theme.relation_to_previous != "parallel":
                    theme.parallel_group = ""
                elif not theme.parallel_group:
                    theme.parallel_group = f"parallel-{index}"
                    theme.heading_quality_status = "needs_human_review"
        return themes

    def _apply_theme_ids(self, papers: list[Paper], themes: list[Theme]) -> None:
        for paper in papers:
            paper.theme_ids = []
        by_id = {paper.paper_id: paper for paper in papers}
        for theme in themes:
            for paper_id in theme.paper_ids:
                if paper_id in by_id:
                    by_id[paper_id].theme_ids.append(theme.theme_id)

    def coverage_snapshot(self, papers: list[Paper]) -> dict[str, Any]:
        recent = sum("recent" in paper.coverage_tags for paper in papers)
        chinese = sum("chinese" in paper.coverage_tags for paper in papers)
        preferred = sum("preferred_venue" in paper.coverage_tags for paper in papers)
        roles = " ".join(paper.rhetorical_role.casefold() for paper in papers)
        stage_counts = {
            stage: sum(stage in paper.evidence_stages for paper in papers)
            for stage in EVIDENCE_STAGE_ORDER
        }
        return {
            "selected_count": len(papers),
            "recent_window": self._recent_window_text(),
            "recent_count": recent,
            "recent_target": self.config.min_recent_selected,
            "recent_target_met": recent >= self.config.min_recent_selected,
            "chinese_count": chinese,
            "chinese_target": self.config.min_chinese_selected,
            "chinese_target_met": chinese >= self.config.min_chinese_selected,
            "preferred_venues": self.config.preferred_venues,
            "preferred_venue_count": preferred,
            "preferred_venue_target": self.config.min_preferred_venue_selected,
            "preferred_venue_target_met": (
                preferred >= self.config.min_preferred_venue_selected
            ),
            "canonical_role_count": sum(
                any(token in paper.rhetorical_role.casefold() for token in ("canonical", "foundation", "seminal"))
                for paper in papers
            ),
            "counter_evidence_role_count": sum(
                any(token in paper.rhetorical_role.casefold() for token in ("counter", "negative", "failure", "boundary"))
                for paper in papers
            ),
            "evidence_stage_counts": stage_counts,
            "mechanism_target_met": stage_counts["mechanism"] > 0,
            "counter_evidence_target_met": any(
                token in roles for token in ("counter", "negative", "failure", "boundary")
            ),
            "note": (
                "Coverage targets guide corpus composition after relevance screening; "
                "they do not change relevance scores or prove source quality."
            ),
        }

    def reflect(
        self,
        themes: list[Theme],
        papers: list[Paper] | None = None,
        draft_audit: dict[str, Any] | None = None,
        *,
        round_index: int = 1,
    ) -> GapAudit:
        coverage = self.coverage_snapshot(papers or [])
        draft_audit = draft_audit or {}
        if self.llm.enabled:
            try:
                payload = "\n".join(
                    f"{theme.theme_id} | {theme.name} | n={len(theme.paper_ids)} | "
                    f"claim={theme.synthesis_claim} | gap={theme.residual_gap}"
                    for theme in themes
                )
                data = self.llm.complete_json(
                    stage="coverage_reflection",
                    prompt_version=REFLECT_PROMPT_VERSION,
                    system=REFLECT_SYSTEM,
                    user=REFLECT_USER.format(
                        topic=self.config.topic,
                        themes=payload,
                        coverage=json.dumps(coverage, ensure_ascii=False),
                        draft_audit=json.dumps(draft_audit, ensure_ascii=False),
                        count=self.config.supplemental_query_count,
                        round_index=round_index,
                    ),
                    temperature=0.2,
                )
                queries = [
                    QueryVariant(
                        query=str(item.get("query", "")).strip(),
                        language=str(item.get("language", "mixed")),
                        facet=str(item.get("facet", "gap")),
                        rationale=str(item.get("rationale", "")),
                        round=round_index,
                    )
                    for item in data.get("supplemental_queries", [])
                    if str(item.get("query", "")).strip()
                ]
                if queries:
                    return GapAudit(
                        adequate=bool(data.get("adequate", False)),
                        coverage_summary=str(data.get("coverage_summary", "")),
                        weak_themes=list(data.get("weak_themes", [])),
                        supplemental_queries=queries[
                            : self.config.supplemental_query_count
                        ],
                        rationale=str(data.get("rationale", "")),
                        round_index=round_index,
                        stop_recommended=bool(data.get("stop_recommended", False)),
                        stop_reason=str(data.get("stop_reason", "")),
                        draft_gap_summary=str(data.get("draft_gap_summary", "")),
                        missing_evidence_stages=_valid_evidence_stages(
                            data.get("missing_evidence_stages")
                        ),
                        unsupported_transition_claims=_string_list(
                            data.get("unsupported_transition_claims")
                        ),
                    )
            except LLMError as exc:
                self.fallbacks.append(f"coverage_reflection: {exc}")

        weakest = min(themes, key=lambda theme: len(theme.paper_ids))
        queries: list[QueryVariant] = []
        stage_gaps = _string_list(draft_audit.get("missing_evidence_stages"))
        gap_priority = [
            "mechanism",
            "real_time_inference",
            "incremental_prediction",
            "net_value",
            "external_validity",
            "identification",
            "prediction",
            "decision_change",
            "concept_definition",
        ]
        stage_gaps.sort(
            key=lambda stage: (
                gap_priority.index(stage) if stage in gap_priority else len(gap_priority)
            )
        )
        for stage in stage_gaps[:1]:
            queries.append(
                QueryVariant(
                    query=f"{self.config.topic} {stage.replace('_', ' ')} evidence limitations",
                    language="en",
                    facet=f"evidence-stage-{stage}",
                    rationale=f"补足初稿中缺失的 {stage} 证据层",
                    round=round_index,
                )
            )
        if not coverage["recent_target_met"]:
            queries.append(
                QueryVariant(
                    query=f"{self.config.topic} {self._recent_window_text()} recent evidence",
                    language="en",
                    facet="recent-gap",
                    rationale="补足近五年研究覆盖",
                    round=round_index,
                )
            )
        if not coverage["chinese_target_met"]:
            queries.append(
                QueryVariant(
                    query=f"{self.config.topic} 中文 核心期刊 实证研究",
                    language="zh",
                    facet="chinese-gap",
                    rationale="补足中文研究与本土情境",
                    round=round_index,
                )
            )
        if not coverage["preferred_venue_target_met"] and self.config.preferred_venues:
            queries.append(
                QueryVariant(
                    query=f"{self.config.topic} {' '.join(self.config.preferred_venues[:8])}",
                    language="en",
                    facet="preferred-venue-gap",
                    rationale="补足用户指定期刊覆盖；相关性仍单独判断",
                    round=round_index,
                )
            )
        if coverage["canonical_role_count"] == 0:
            queries.append(
                QueryVariant(
                    query=f"{self.config.topic} seminal foundational theory classic paper",
                    language="en",
                    facet="canonical-gap",
                    rationale="补足定义对象、机制或方法谱系所需的经典基础，而非按引用数直接纳入",
                    round=round_index,
                )
            )
        if not coverage["counter_evidence_target_met"]:
            queries.append(
                QueryVariant(
                    query=f"{self.config.topic} negative results failure instability counter evidence",
                    language="en",
                    facet="counter-evidence-gap",
                    rationale="补足反例、失败模式和结论边界，避免单一正向证据主导综合",
                    round=round_index,
                )
            )
        queries.extend([
            QueryVariant(
                query=f"{self.config.topic} {weakest.name} limitations evidence",
                language="mixed",
                facet="coverage-gap",
                rationale="针对文献数量最少的主题补充局限与经验证据",
                round=round_index,
            ),
            QueryVariant(
                query=f"{self.config.topic} benchmark evaluation comparative study",
                language="en",
                facet="evaluation-gap",
                rationale="补充跨方法比较与评价研究",
                round=round_index,
            ),
        ])
        queries = queries[: self.config.supplemental_query_count]
        return GapAudit(
            adequate=False,
            coverage_summary=(
                f"{weakest.name}仅包含{len(weakest.paper_ids)}篇，且评价证据可能不足。"
            ),
            weak_themes=[
                {
                    "theme_id": weakest.theme_id,
                    "gap": "数量或证据类型覆盖不足",
                    "severity": "medium",
                }
            ],
            supplemental_queries=queries,
            rationale="执行规定的一轮覆盖审计，并对最薄弱主题及评价维度补检。",
            round_index=round_index,
            stop_recommended=False,
            stop_reason="尚未完成最低补检轮次。",
            draft_gap_summary=str(draft_audit.get("summary", "")),
            missing_evidence_stages=stage_gaps,
            unsupported_transition_claims=_string_list(
                draft_audit.get("unsupported_transition_claims")
            ),
        )


class FullTextFetcher:
    def __init__(self, config: PipelineConfig, http: CachedHTTP) -> None:
        self.config = config
        self.http = http

    def enrich(self, papers: list[Paper]) -> list[str]:
        warnings: list[str] = []
        if not self.config.fetch_full_text:
            return warnings
        count = 0
        for paper in papers:
            if count >= self.config.max_full_text_papers:
                break
            if not paper.open_access_pdf:
                paper.extraction_basis = "abstract_only"
                continue
            try:
                data = self.http.get_bytes(paper.open_access_pdf)
                if len(data) > 25 * 1024 * 1024:
                    raise RuntimeError("PDF exceeds 25 MB safety limit")
                document = extract_pdf_document(
                    data,
                    max_pages=self.config.max_full_text_pages,
                    max_chars=self.config.max_full_text_chars,
                )
                paper.pdf_page_count = int(document["page_count"])
                paper.pdf_pages_extracted = int(document["pages_extracted"])
                paper.pdf_parse_quality = str(document["parse_quality"])
                paper.full_text_sections = dict(document["sections"])
                paper.full_text_excerpt = str(document["text"])
                paper.extraction_basis = str(document["evidence_basis"])
                paper.source_warning_flags.extend(document["warning_flags"])
                if len(paper.full_text_excerpt) >= 800:
                    count += 1
                else:
                    paper.extraction_basis = "pdf_parse_failed"
            except Exception as exc:
                paper.extraction_basis = "pdf_parse_failed"
                paper.pdf_parse_quality = "failed"
                warnings.append(f"{paper.paper_id} full-text fallback to abstract: {exc}")
        return warnings


def extract_pdf_document(
    data: bytes, *, max_pages: int = 80, max_chars: int = 60000
) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return {
            "text": "", "sections": {}, "page_count": 0, "pages_extracted": 0,
            "parse_quality": "dependency_missing", "evidence_basis": "pdf_parse_failed",
            "warning_flags": ["pypdf_missing"],
        }
    reader = PdfReader(io.BytesIO(data))
    page_count = len(reader.pages)
    page_parts: list[str] = []
    low_text_pages = 0
    injection_flags: list[str] = []
    for page_number, page in enumerate(reader.pages[:max_pages], start=1):
        raw = page.extract_text() or ""
        if len(raw.strip()) < 80:
            low_text_pages += 1
        clean, flags = sanitize_untrusted_text(raw)
        injection_flags.extend(flags)
        page_parts.append(f"[PAGE {page_number}]\n{clean}")
    pages_extracted = len(page_parts)
    combined = "\n".join(page_parts)
    sections = identify_sections(combined)
    selected = _section_aware_excerpt(sections, combined, max_chars)
    unreliable = pages_extracted > 0 and low_text_pages / pages_extracted > 0.35
    complete = pages_extracted == page_count and len(combined) <= max_chars
    section_coverage = {name for name, text in sections.items() if len(text) >= 120}
    target_sections = {"abstract", "methods", "results", "discussion", "limitations", "conclusion"}
    if unreliable:
        basis = "ocr_or_layout_unreliable"
        quality = "low_text_or_layout_unreliable"
    elif complete:
        basis = "fulltext_complete"
        quality = "text_extractable"
    elif len(section_coverage & target_sections) >= 4:
        basis = "section_complete"
        quality = "section_coverage_good"
    else:
        basis = "partial_fulltext"
        quality = "partial"
    warnings = sorted(set(injection_flags))
    if pages_extracted < page_count:
        warnings.append("page_limit_reached")
    if len(combined) > max_chars:
        warnings.append("character_budget_applied_section_aware")
    return {
        "text": selected,
        "sections": sections,
        "page_count": page_count,
        "pages_extracted": pages_extracted,
        "parse_quality": quality,
        "evidence_basis": basis,
        "warning_flags": warnings,
    }


def extract_pdf_text(data: bytes) -> str:
    """Compatibility wrapper; callers should use extract_pdf_document."""
    return str(extract_pdf_document(data)["text"])


def split_sentences(text: str) -> list[str]:
    value = re.sub(r"\s+", " ", text or "").strip()
    if not value:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", value)
    return [part.strip() for part in parts if len(part.strip()) >= 20][:20]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


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

_EVIDENCE_STAGE_ALIASES = {
    "concept": "concept_definition",
    "definition": "concept_definition",
    "concept_definition": "concept_definition",
    "mechanism": "mechanism",
    "identification": "identification",
    "measurement": "identification",
    "real_time": "real_time_inference",
    "real-time": "real_time_inference",
    "real_time_inference": "real_time_inference",
    "prediction": "prediction",
    "forecast": "prediction",
    "incremental": "incremental_prediction",
    "incremental_prediction": "incremental_prediction",
    "decision": "decision_change",
    "decision_change": "decision_change",
    "net_value": "net_value",
    "cost": "net_value",
    "external": "external_validity",
    "replication": "external_validity",
    "external_validity": "external_validity",
}


def _valid_evidence_stages(value: Any) -> list[str]:
    """Normalize only explicitly supported evidence-stage labels."""
    normalized: list[str] = []
    for raw in _string_list(value):
        for token in re.split(r"[,;/|]", raw.casefold()):
            stage = _EVIDENCE_STAGE_ALIASES.get(token.strip().replace(" ", "_"))
            if stage and stage not in normalized:
                normalized.append(stage)
    return sorted(normalized, key=EVIDENCE_STAGE_ORDER.index)


def _heuristic_evidence_stages(text: str) -> list[str]:
    """Conservative fallback: labels a stage only when the source text says so."""
    patterns = {
        "concept_definition": r"(?:\b(?:concept|definition|taxonomy|framework)\b|概念|定义|界定|分类)",
        "mechanism": r"(?:\b(?:mechanism|formation|persistence|transition|cause)\b|机制|形成|持续|转变原因)",
        "identification": r"(?:\b(?:identify|detect|estimate|measurement|classification)\b|识别|检测|估计|测量)",
        "real_time_inference": r"(?:\b(?:real[- ]?time|online|nowcast|vintage)\b|实时|在线|事前信息集)",
        "prediction": r"(?:\b(?:predict|forecast|out[- ]?of[- ]?sample)\b|预测|预报|样本外)",
        "incremental_prediction": r"(?:\b(?:incremental|benchmark|baseline|horse race)\b|增量|基准模型|比较预测)",
        "decision_change": r"(?:\b(?:decision|allocation|policy|intervention|portfolio)\b|决策|配置|政策|干预)",
        "net_value": r"(?:\b(?:transaction cost|turnover|utility|net value|implementation cost)\b|交易成本|换手|效用|净价值|实施成本)",
        "external_validity": r"(?:\b(?:replication|external valid|cross[- ]?(?:market|country|setting)|robustness)\b|复现|外部有效|跨市场|跨国家|稳健性)",
    }
    return [
        stage
        for stage in EVIDENCE_STAGE_ORDER
        if re.search(patterns[stage], text or "", re.IGNORECASE)
    ]


def _title_overlap(title: str, topic: str) -> float:
    topic_tokens = set(tokenize(topic))
    return len(topic_tokens & set(tokenize(title))) / max(1, len(topic_tokens))


def _paper_evidence_text(paper: Paper) -> str:
    if paper.full_text_sections:
        ordered = []
        for name in ("abstract", "methods", "results", "discussion", "limitations", "conclusion", "other"):
            value = paper.full_text_sections.get(name, "")
            if value:
                ordered.append(f"[SECTION {name.upper()}]\n{value}")
        return "\n\n".join(ordered)
    return paper.full_text_excerpt or paper.abstract


def _is_chinese_paper(paper: Paper) -> bool:
    language = (paper.language or "").casefold()
    if language.startswith(("zh", "chi", "zho")):
        return True
    text = f"{paper.title} {paper.abstract[:500]}"
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    visible = len(re.sub(r"\s+", "", text))
    return visible > 0 and chinese / visible >= 0.2


def _first_matching_sentence(sentences: list[str], pattern: str) -> str:
    matcher = re.compile(pattern, re.IGNORECASE)
    return next((sentence for sentence in sentences if matcher.search(sentence)), "")


def _locate_section(paper: Paper, value: str) -> str:
    needle = value[:80].casefold()
    for name, text in paper.full_text_sections.items():
        if needle and needle in text.casefold():
            return name
    return "abstract" if paper.extraction_basis == "abstract_only" else "unknown"


def _evidence_spans(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    spans = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        spans.append(
            {
                "field": str(item.get("field", "")),
                "section": str(item.get("section", "unknown")),
                "page": item.get("page"),
                "text": compact(text, 520),
                "attribution": str(item.get("attribution", "unknown")),
                "confidence": _bounded_float(item.get("confidence"), 0.5),
            }
        )
    return spans


def _bounded_float(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+chatgpt", re.IGNORECASE),
    re.compile(r"忽略.{0,12}(此前|之前|以上).{0,8}指令"),
    re.compile(r"系统提示词"),
]


def sanitize_untrusted_text(text: str) -> tuple[str, list[str]]:
    value = text or ""
    flags: list[str] = []
    for index, pattern in enumerate(INJECTION_PATTERNS, start=1):
        if pattern.search(value):
            flags.append(f"potential_prompt_injection_pattern_{index}")
            value = pattern.sub("[REDACTED_UNTRUSTED_INSTRUCTION]", value)
    value = value.replace("\x00", " ")
    replacement_ratio = value.count("\ufffd") / max(1, len(value))
    if replacement_ratio > 0.01:
        flags.append("encoding_or_ocr_corruption")
    return value.strip(), flags


SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^(abstract|摘要)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("methods", re.compile(r"^(methods?|methodology|materials and methods|研究方法|方法)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("results", re.compile(r"^(results?|findings?|结果|研究结果)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("discussion", re.compile(r"^(discussion|讨论)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("limitations", re.compile(r"^(limitations?|strengths and limitations|局限性?|研究局限)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("conclusion", re.compile(r"^(conclusions?|总结|结论|总结与展望)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("references", re.compile(r"^(references|bibliography|参考文献)\s*$", re.IGNORECASE | re.MULTILINE)),
]


def identify_sections(text: str) -> dict[str, str]:
    markers: list[tuple[int, int, str]] = []
    for name, pattern in SECTION_PATTERNS:
        markers.extend((match.start(), match.end(), name) for match in pattern.finditer(text))
    markers.sort()
    if not markers:
        return {"other": compact(text, len(text))}
    sections: dict[str, str] = {}
    if markers[0][0] > 0:
        sections["other"] = text[: markers[0][0]].strip()
    for index, (start, end, name) in enumerate(markers):
        stop = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        chunk = text[end:stop].strip()
        if chunk:
            sections[name] = (sections.get(name, "") + "\n" + chunk).strip()
    return sections


def _section_aware_excerpt(
    sections: dict[str, str], combined: str, max_chars: int
) -> str:
    if not sections:
        return combined[:max_chars]
    budgets = {
        "abstract": 0.08,
        "methods": 0.20,
        "results": 0.25,
        "discussion": 0.18,
        "limitations": 0.12,
        "conclusion": 0.10,
        "other": 0.07,
    }
    parts: list[str] = []
    used = 0
    for name, share in budgets.items():
        value = sections.get(name, "")
        if not value:
            continue
        budget = max(500, int(max_chars * share))
        excerpt = value[:budget]
        parts.append(f"[SECTION {name.upper()}]\n{excerpt}")
        used += len(excerpt)
    if used < max_chars and sections.get("references"):
        remaining = max_chars - used
        parts.append(f"[SECTION REFERENCES]\n{sections['references'][:remaining]}")
    return "\n\n".join(parts)[:max_chars]
