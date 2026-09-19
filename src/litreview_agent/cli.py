from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .config import PipelineConfig
from .io_utils import slugify
from .pipeline import LiteratureReviewPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检索、筛选、抽取、组织并撰写带引用的文献综述初稿"
    )
    parser.add_argument("--topic", required=True, help="研究主题（自然语言）")
    parser.add_argument(
        "--out",
        type=Path,
        help="运行输出目录；默认 runs/<时间>-<主题>",
    )
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--year-from", type=int, default=2015)
    parser.add_argument("--year-to", type=int)
    parser.add_argument(
        "--sources",
        default="openalex,semantic_scholar,crossref,arxiv",
        help="逗号分隔的数据源",
    )
    parser.add_argument("--min-candidates", type=int, default=30)
    parser.add_argument("--min-selected", type=int, default=15)
    parser.add_argument("--target-selected", type=int, default=18)
    parser.add_argument("--relevance-threshold", type=float, default=3.2)
    parser.add_argument(
        "--allow-quantity-backfill",
        action="store_true",
        help="课程兼容选项：以边界文献补足数量；默认关闭且不视为高相关证据",
    )
    parser.add_argument("--query-count", type=int, default=8)
    parser.add_argument("--per-source-limit", type=int, default=12)
    parser.add_argument(
        "--recent-years",
        type=int,
        default=5,
        help="近年覆盖窗口长度，默认5年",
    )
    parser.add_argument(
        "--min-recent-selected",
        type=int,
        default=5,
        help="高相关集合中近年论文的目标下限；不足会触发补检",
    )
    parser.add_argument(
        "--min-chinese-selected",
        type=int,
        default=2,
        help="高相关集合中中文论文的目标下限；不足会触发补检",
    )
    parser.add_argument(
        "--preferred-venues",
        default="",
        help="逗号分隔的指定期刊/会议名，仅用于覆盖追踪和定向召回",
    )
    parser.add_argument(
        "--min-preferred-venue-selected",
        type=int,
        default=0,
        help="指定期刊覆盖目标；不会给相关性分数加分",
    )
    parser.add_argument("--max-reflection-rounds", type=int, default=2)
    parser.add_argument("--supplemental-query-count", type=int, default=6)
    parser.add_argument("--theme-min", type=int, default=3)
    parser.add_argument("--theme-max", type=int, default=6)
    parser.add_argument(
        "--generation-strategy",
        choices=["thematic", "one_shot"],
        default="thematic",
        help="分主题生成或一次性生成基线",
    )
    parser.add_argument(
        "--max-writing-revision-rounds",
        type=int,
        default=1,
        help="正式终稿在论证审计失败后的最大LLM重写轮数，默认1轮",
    )
    parser.add_argument(
        "--no-citation-chasing", action="store_true", help="关闭OpenAlex前向/后向引文追踪"
    )
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "openai", "heuristic"],
        default="auto",
        help="auto: 有Key用LLM，否则启发式；openai: 强制LLM；heuristic: 无LLM",
    )
    parser.add_argument(
        "--offline-corpus",
        type=Path,
        help="离线论文 JSON；用于测试和无网络演示",
    )
    parser.add_argument(
        "--no-full-text",
        action="store_true",
        help="只使用摘要，不尝试下载开放获取PDF",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="打印解析后的配置并退出",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("runs") / f"{timestamp}-{slugify(args.topic)}"
    config = PipelineConfig(
        topic=args.topic,
        output_dir=out.resolve(),
        language=args.language,
        year_from=args.year_from,
        year_to=args.year_to,
        sources=[item.strip() for item in args.sources.split(",") if item.strip()],
        min_candidates=args.min_candidates,
        min_selected=args.min_selected,
        target_selected=args.target_selected,
        relevance_threshold=args.relevance_threshold,
        allow_quantity_backfill=args.allow_quantity_backfill,
        query_count=args.query_count,
        per_query_source_limit=args.per_source_limit,
        recent_years=max(1, args.recent_years),
        min_recent_selected=max(0, args.min_recent_selected),
        min_chinese_selected=max(0, args.min_chinese_selected),
        preferred_venues=[
            item.strip() for item in args.preferred_venues.split(",") if item.strip()
        ],
        min_preferred_venue_selected=max(0, args.min_preferred_venue_selected),
        max_reflection_rounds=args.max_reflection_rounds,
        supplemental_query_count=max(1, args.supplemental_query_count),
        theme_min=max(1, args.theme_min),
        theme_max=max(args.theme_min, args.theme_max),
        generation_strategy=args.generation_strategy,
        max_writing_revision_rounds=max(0, args.max_writing_revision_rounds),
        enable_citation_chasing=not args.no_citation_chasing,
        llm_mode=args.llm_mode,
        offline_corpus=args.offline_corpus.resolve()
        if args.offline_corpus
        else None,
        fetch_full_text=not args.no_full_text,
    )
    if args.print_config:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return 0
    try:
        result = LiteratureReviewPipeline(config).run()
    except KeyboardInterrupt:
        print("Run interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1
    print("\n完成。")
    print(f"综述初稿: {result['review_path']}")
    print(f"运行完整性报告: {result['integrity_path']}")
    print(f"内容质量审计: {result['content_quality_path']}")
    print(f"正文质量检查: {result['publication_quality_path']}")
    if result.get("delivery_manifest_path"):
        print(f"分离交付清单: {result['delivery_manifest_path']}")
    print(
        "关键检查: "
        + json.dumps(result["integrity"]["checks"], ensure_ascii=False)
    )
    return 0
