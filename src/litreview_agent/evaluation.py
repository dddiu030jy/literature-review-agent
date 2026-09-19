from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .io_utils import write_csv, write_json
from .models import ClaimEvidence, Paper


def binary_metrics(gold: list[int], predicted: list[int]) -> dict[str, float | int]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must have equal non-zero length")
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, predicted))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, predicted))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, predicted))
    tn = sum(g == 0 and p == 0 for g, p in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(gold)
    return {
        "n": len(gold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "accuracy": round(accuracy, 4),
    }


def cohens_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("rater lists must have equal non-zero length")
    observed = sum(a == b for a, b in zip(rater_a, rater_b)) / len(rater_a)
    p_a = sum(rater_a) / len(rater_a)
    p_b = sum(rater_b) / len(rater_b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    return round((observed - expected) / (1 - expected), 4) if expected < 1 else 1.0


def calibrate_threshold(
    scores: list[float], gold: list[int], thresholds: Iterable[float] | None = None
) -> dict[str, Any]:
    candidates = list(thresholds or [round(2.0 + index * 0.1, 1) for index in range(31)])
    rows = []
    for threshold in candidates:
        metrics = binary_metrics(gold, [int(score >= threshold) for score in scores])
        rows.append({"threshold": threshold, **metrics})
    best = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    return {"recommended_threshold": best["threshold"], "best": best, "curve": rows}


def create_human_evaluation_pack(
    output_dir: Path,
    candidates: list[Paper],
    selected: list[Paper],
    claims: list[ClaimEvidence],
    *,
    seed: int,
) -> dict[str, Any]:
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    screening_rows = sorted(candidates, key=lambda paper: paper.paper_id)[:]
    rng.shuffle(screening_rows)
    screening_rows = screening_rows[: min(50, len(screening_rows))]
    write_csv(
        evaluation_dir / "screening_gold_template.csv",
        (
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "model_score": paper.relevance_score,
                "model_decision": paper.screening_decision,
                "rater_1": "",
                "rater_2": "",
                "adjudicated_gold": "",
                "notes": "",
            }
            for paper in screening_rows
        ),
        ["paper_id", "title", "model_score", "model_decision", "rater_1", "rater_2", "adjudicated_gold", "notes"],
    )
    extraction_rows = selected[:]
    rng.shuffle(extraction_rows)
    extraction_rows = extraction_rows[: min(20, len(extraction_rows))]
    extraction_fields = ["research_question", "methods", "findings", "limitations"]
    write_csv(
        evaluation_dir / "extraction_fact_check_template.csv",
        (
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "field": field_name,
                "extracted_value": getattr(paper, field_name),
                "evidence_basis": paper.extraction_basis,
                "human_supported_0_or_1": "",
                "human_completeness_0_to_2": "",
                "evidence_page_or_section": "",
                "notes": "",
            }
            for paper in extraction_rows
            for field_name in extraction_fields
        ),
        ["paper_id", "title", "field", "extracted_value", "evidence_basis", "human_supported_0_or_1", "human_completeness_0_to_2", "evidence_page_or_section", "notes"],
    )
    write_csv(
        evaluation_dir / "claim_evidence_audit_template.csv",
        (
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "supporting_paper_ids": claim.supporting_paper_ids,
                "evidence_locations": claim.evidence_page_or_section,
                "automatic_support_type": claim.support_type,
                "human_supported_0_or_1": "",
                "direction_correct_0_or_1": "",
                "scope_not_overstated_0_or_1": "",
                "notes": "",
            }
            for claim in claims
        ),
        ["claim_id", "claim_text", "supporting_paper_ids", "evidence_locations", "automatic_support_type", "human_supported_0_or_1", "direction_correct_0_or_1", "scope_not_overstated_0_or_1", "notes"],
    )
    status = {
        "screening_gold_rows": len(screening_rows),
        "screening_gold_complete": False,
        "minimum_required_per_topic": 50,
        "two_independent_raters_required": True,
        "extraction_papers_sampled": len(extraction_rows),
        "extraction_fact_check_complete": False,
        "claim_rows": len(claims),
        "claim_evidence_human_audit_complete": False,
        "quality_validation_complete": False,
        "blocking_reasons": [
            "screening labels have not been supplied by two human raters",
            "extraction factuality has not been checked against source text",
            "claim-evidence direction and scope have not been human-audited",
        ],
    }
    if len(screening_rows) < 50:
        status["blocking_reasons"].insert(
            0,
            f"candidate pool provides only {len(screening_rows)} unique rows; at least 50 are required",
        )
    write_json(evaluation_dir / "evaluation_status.json", status)
    return status


def evaluate_completed_screening_csv(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    complete = [
        row for row in rows
        if row.get("rater_1", "").strip() in {"0", "1"}
        and row.get("rater_2", "").strip() in {"0", "1"}
        and row.get("adjudicated_gold", "").strip() in {"0", "1"}
    ]
    if len(complete) < 50:
        raise ValueError(f"need at least 50 fully labeled rows, got {len(complete)}")
    r1 = [int(row["rater_1"]) for row in complete]
    r2 = [int(row["rater_2"]) for row in complete]
    gold = [int(row["adjudicated_gold"]) for row in complete]
    scores = [float(row["model_score"]) for row in complete]
    current = [int(row.get("model_decision") in {"include", "boundary_backfill"}) for row in complete]
    return {
        "cohens_kappa": cohens_kappa(r1, r2),
        "current_metrics": binary_metrics(gold, current),
        "threshold_calibration": calibrate_threshold(scores, gold),
    }
