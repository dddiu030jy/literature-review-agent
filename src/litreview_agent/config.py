from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    topic: str
    output_dir: Path
    language: str = "zh"
    year_from: int | None = 2015
    year_to: int | None = None
    sources: list[str] = field(
        default_factory=lambda: ["openalex", "semantic_scholar", "crossref", "arxiv"]
    )
    query_count: int = 8
    per_query_source_limit: int = 12
    min_candidates: int = 30
    min_selected: int = 15
    target_selected: int = 18
    relevance_threshold: float = 3.2
    allow_quantity_backfill: bool = False
    theme_min: int = 3
    theme_max: int = 6
    min_reflection_rounds: int = 1
    max_reflection_rounds: int = 2
    supplemental_query_count: int = 6
    min_supplemental_selected_gain: int = 1
    require_supplemental_integration: bool = True
    min_unique_retrieval_gain: float = 0.03
    enable_citation_chasing: bool = True
    citation_seed_count: int = 3
    citation_chase_limit: int = 8
    fetch_full_text: bool = True
    max_full_text_papers: int = 6
    max_full_text_chars: int = 60000
    max_full_text_pages: int = 80
    coverage_tracks: list[str] = field(
        default_factory=lambda: [
            "core",
            "canonical",
            "recent_frontier",
            "chinese",
            "mechanism",
            "methods_evaluation",
            "counter_evidence",
            "boundary_conditions",
        ]
    )
    recent_years: int = 5
    min_recent_selected: int = 5
    min_chinese_selected: int = 2
    preferred_venues: list[str] = field(default_factory=list)
    min_preferred_venue_selected: int = 0
    enforce_publication_clean: bool = True
    create_delivery_views: bool = True
    provisional_review_before_reflection: bool = True
    llm_mode: str = "auto"
    llm_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    request_timeout: int = 45
    request_retries: int = 3
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 4096
    generation_strategy: str = "thematic"
    max_writing_revision_rounds: int = 1
    offline_corpus: Path | None = None
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_dir"] = str(self.output_dir)
        value["offline_corpus"] = str(self.offline_corpus) if self.offline_corpus else None
        return value
