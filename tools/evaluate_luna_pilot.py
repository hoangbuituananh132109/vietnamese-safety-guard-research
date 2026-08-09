from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translator.models import TranslationInputItem, TranslationOutputItem, TranslationRequest, TranslationResponse
from translator.validators import TranslationValidationError, quality_warnings, validate_hard_quality, validate_response

PILOT = ROOT / "data" / "luna_pilot_200"
VARIANTS = ["p2_normal_bs30", "p2_bs10", "p2_final_remaining", "p2_final_resume_net"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # File iteration only splits on actual record newlines. str.splitlines()
    # also splits on U+2028/U+2029, which may legitimately occur inside JSON text.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_pair(row: dict[str, Any], prompt_vi: str | None, response_vi: str | None) -> dict[str, Any]:
    source = TranslationInputItem(
        seq=int(row["pilot_seq"]), record_uid=row["record_uid"],
        prompt=row.get("prompt_en") or "", response=row.get("response_en"),
    )
    output = TranslationOutputItem(
        seq=int(row["pilot_seq"]), record_uid=row["record_uid"],
        prompt_vi=prompt_vi or "", response_vi=response_vi, warnings=[],
    )
    hard: list[str] = []
    try:
        request = TranslationRequest(batch_id="evaluation", items=[source])
        response = TranslationResponse(batch_id="evaluation", items=[output])
        validate_response(request, response)
        validate_hard_quality(request, response)
    except TranslationValidationError as exc:
        hard = list(exc.errors)
    heuristic = quality_warnings(
        {"prompt": source.prompt, "response": source.response},
        {"prompt_vi": prompt_vi, "response_vi": response_vi},
    )
    return {"hard": hard, "heuristic": heuristic, "flagged": bool(hard or heuristic)}


def normalized(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def main() -> None:
    rows = read_jsonl(PILOT / "pilot_200.jsonl")
    by_uid = {row["record_uid"]: row for row in rows}
    variant_items: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in VARIANTS:
        items = read_jsonl(PILOT / "runs" / variant / "items.jsonl")
        variant_items[variant] = {
            item["record_uid"]: item for item in items
            if item.get("record_uid") and not item.get("structural_errors")
        }

    merged: list[dict[str, Any]] = []
    missing: list[str] = []
    counters = Counter()
    group_stats: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        uid = row["record_uid"]
        if row["length_bucket"] == "normal":
            order = ["p2_normal_bs30", "p2_bs10", "p2_final_remaining", "p2_final_resume_net"]
        else:
            order = ["p2_bs10", "p2_final_remaining", "p2_final_resume_net", "p2_normal_bs30"]
        selected_variant = next((v for v in order if uid in variant_items[v]), None)
        if selected_variant is None:
            missing.append(uid)
            continue
        luna_item = variant_items[selected_variant][uid]
        luna = validate_pair(row, luna_item.get("prompt_vi"), luna_item.get("response_vi"))
        gemini = validate_pair(row, row.get("gemini_prompt_vi"), row.get("gemini_response_vi"))
        reference = validate_pair(row, row.get("reference_prompt_vi"), row.get("reference_response_vi"))
        if not luna["flagged"] and gemini["flagged"]:
            proxy = "luna_advantage"
        elif luna["flagged"] and not gemini["flagged"]:
            proxy = "gemini_advantage"
        else:
            proxy = "inconclusive"
        counters[f"proxy_{proxy}"] += 1
        counters["luna_flagged"] += int(luna["flagged"])
        counters["gemini_flagged"] += int(gemini["flagged"])
        counters["reference_flagged"] += int(reference["flagged"])
        group_stats[row["pilot_group"]]["translated"] += 1
        group_stats[row["pilot_group"]][f"proxy_{proxy}"] += 1
        group_stats[row["pilot_group"]]["luna_flagged"] += int(luna["flagged"])
        merged.append({
            **row,
            "luna_variant": selected_variant,
            "luna_prompt_vi": luna_item.get("prompt_vi"),
            "luna_response_vi": luna_item.get("response_vi"),
            "luna_model_warnings": luna_item.get("warnings") or [],
            "luna_validation": luna,
            "gemini_validation": gemini,
            "reference_validation": reference,
            "validator_proxy": proxy,
            "luna_exact_gemini": normalized(luna_item.get("prompt_vi")) == normalized(row.get("gemini_prompt_vi"))
                and normalized(luna_item.get("response_vi")) == normalized(row.get("gemini_response_vi")),
            "luna_exact_reference": normalized(luna_item.get("prompt_vi")) == normalized(row.get("reference_prompt_vi"))
                and normalized(luna_item.get("response_vi")) == normalized(row.get("reference_response_vi")),
        })

    out = PILOT / "evaluation"
    out.mkdir(exist_ok=True)
    (out / "merged_items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8"
    )
    audit = [row for row in merged if row["luna_validation"]["flagged"] or row["validator_proxy"] != "inconclusive"]
    (out / "audit_candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit), encoding="utf-8"
    )
    final_quality = json.loads((ROOT / "reports/final_quality/translation_quality_summary.json").read_text(encoding="utf-8"))
    hard_disposition = final_quality.get("validator", {}).get("current_hard_disposition", {})
    summary = {
        "pilot_expected": len(rows),
        "luna_merged": len(merged),
        "missing": len(missing),
        "missing_uids": missing,
        "selection_groups": dict(Counter(row["pilot_group"] for row in rows)),
        "merged_groups": {group: dict(stats) for group, stats in sorted(group_stats.items())},
        "same_validator_proxy": dict(counters),
        "proxy_limitations": [
            "A validator advantage only compares detected structural/untranslated-text risks; it is not a semantic win.",
            "Inconclusive includes both-clean and both-flagged pairs and requires blind human/model judging.",
            "The reference set contains human overrides, so reference validator flags are not automatically reference errors.",
        ],
        "evaluator_calibration_from_final_dataset": {
            "current_hard_disposition": hard_disposition,
            "interpretation": "All hard flags remaining in the final dataset were audited overrides; raw hard flags therefore require review and cannot be treated as automatic model failures.",
        },
        "variants_merged": VARIANTS,
    }
    (out / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
