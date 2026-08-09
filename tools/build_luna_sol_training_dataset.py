from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/luna_sol_dataset_v1"
HYBRID_FINAL = ROOT / "data/final_luna_sol_hybrid_v1"
PURE_FINAL = ROOT / "data/final_luna_sol_pure_v1"
LUNA_PAIRED_FINAL = ROOT / "data/final_luna_sol_paired_v1"
GEMINI_PAIRED_FINAL = ROOT / "data/final_gemini_paired_v1"
SOURCE_PATHS = {
    "train": ROOT / "data/final/nemotron_train_en_vi_v10_final.jsonl",
    "valid": ROOT / "data/final/nemotron_valid_en_vi_v10_final.jsonl",
    "test": ROOT / "data/final/nemotron_test_en_vi_v10_final.jsonl",
}
INITIAL_RUNS = {
    "train": ROOT / "data/luna_overnight/nemotron_train_20260801",
    "valid": ROOT / "data/luna_overnight/nemotron_valid_20260802",
    "test": ROOT / "data/luna_overnight/nemotron_test_20260802",
}
REMAINING = ROOT / "data/luna_remaining_20260803/run_light_retry2"
CONTINUATION = ROOT / "data/luna_remaining_20260803/run_light_retry2_continuation"
RECOVERED = ROOT / "data/luna_remaining_20260803/recovered_without_model.jsonl"
SOL_EXPORT = ROOT / "data/sol_fallback_after_luna/sol_results/sol-fallback-all-results.json"
SOL_REVIEW = ROOT / "data/sol_fallback_after_luna/sol_results/validation/manual_review_decisions.jsonl"
SOL_MISSING_ACCEPTED = ROOT / "data/sol_missing_25/sol_results/validation/accepted_final.jsonl"
KNOWN_ACCEPTED_ISSUES = {
    "en-train-00020107-7cd91d92b3ef": "fenced natural-language jailbreak remained English",
    "en-train-00012179-fb3f7c715e03": "one URL hyphen changed to underscore",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def candidate_fingerprint(candidate: dict[str, Any]) -> str:
    payload = json.dumps(
        [candidate.get("prompt_vi"), candidate.get("response_vi")],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict[str, Any]] = {}
    source_order: dict[str, list[str]] = {}
    split_counts: dict[str, int] = {}
    duplicate_source_uids: list[str] = []
    for split, path in SOURCE_PATHS.items():
        rows = read_jsonl(path)
        split_counts[split] = len(rows)
        source_order[split] = []
        for seq, row in enumerate(rows, 1):
            uid = str(row["record_uid"])
            if uid in sources:
                duplicate_source_uids.append(uid)
            sources[uid] = {**row, "_split_seq": seq}
            source_order[split].append(uid)

    selections: dict[str, dict[str, Any]] = {}
    seen_candidates: dict[str, set[str]] = defaultdict(set)
    selection_history: Counter[str] = Counter()

    def offer(uid: str, candidate: dict[str, Any], provider: str, status: str, priority: int) -> None:
        if uid not in sources or not isinstance(candidate, dict):
            return
        if not all(key in candidate for key in ("prompt_vi", "response_vi")):
            return
        seen_candidates[uid].add(candidate_fingerprint(candidate))
        previous = selections.get(uid)
        if previous is None or priority >= int(previous["priority"]):
            selections[uid] = {
                "candidate": candidate,
                "provider": provider,
                "status": status,
                "priority": priority,
            }

    # Initial accepted Luna runs establish the broad base.
    for split, run in INITIAL_RUNS.items():
        for filename, status in (("passed.jsonl", "hard_pass"), ("needs_audit.jsonl", "warning_pass")):
            for row in read_jsonl(run / filename):
                offer(str(row["record_uid"]), row.get("candidate") or {}, "luna_initial", status, 10)

    # Current-validator recoveries replace the stale initial disposition.
    for row in read_jsonl(RECOVERED):
        offer(str(row["record_uid"]), row.get("candidate") or {}, "luna_recovered", "revalidated_pass", 20)

    # The recovered overlapping run contains complete candidates in these terminal files.
    for filename, status in (("passed.jsonl", "hard_pass"), ("needs_audit.jsonl", "warning_pass")):
        for row in read_jsonl(REMAINING / filename):
            offer(str(row["record_uid"]), row.get("candidate") or {}, "luna_remaining", status, 30)

    for filename, status in (("passed.jsonl", "hard_pass"), ("needs_audit.jsonl", "warning_pass")):
        for row in read_jsonl(CONTINUATION / filename):
            offer(str(row["record_uid"]), row.get("candidate") or {}, "luna_continuation", status, 40)

    # Sol is the final repair authority. The user explicitly accepted the two documented
    # residual issues because their incidence is negligible and they remain auditable.
    sol_export = json.loads(SOL_EXPORT.read_text(encoding="utf-8"))
    review_by_uid = {str(row["record_uid"]): row for row in read_jsonl(SOL_REVIEW)}
    for item in sol_export.get("items") or []:
        uid = str(item.get("record_uid") or "")
        candidate = {key: item.get(key) for key in ("seq", "record_uid", "prompt_vi", "response_vi", "warnings")}
        if uid in KNOWN_ACCEPTED_ISSUES:
            status = "accepted_known_issue"
        elif review_by_uid.get(uid, {}).get("decision") == "pass":
            status = "manual_audit_pass"
        else:
            status = "hard_pass"
        offer(uid, candidate, "sol_web", status, 50)

    # The final infrastructure-timeout records were translated through a separate
    # Sol Web queue. All 25 passed the same hard validator; warning-only ASCII-art
    # records were manually reviewed and remain explicitly marked in provenance.
    for item in read_jsonl(SOL_MISSING_ACCEPTED):
        uid = str(item.get("record_uid") or "")
        candidate = {
            key: item.get(key)
            for key in ("seq", "record_uid", "prompt_vi", "response_vi", "warnings")
        }
        offer(
            uid,
            candidate,
            "sol_web_missing25",
            str(item.get("quality_status") or "hard_pass"),
            60,
        )

    missing = [uid for split in SOURCE_PATHS for uid in source_order[split] if uid not in selections]
    extra_selected = sorted(set(selections) - set(sources))
    conflicting_candidates = sorted(uid for uid, fingerprints in seen_candidates.items() if len(fingerprints) > 1)

    structural_issues: list[dict[str, Any]] = []
    seq_metadata_normalized = 0
    pure_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SOURCE_PATHS}
    hybrid_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SOURCE_PATHS}
    provenance_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    missing_by_split: Counter[str] = Counter()

    for split in SOURCE_PATHS:
        for uid in source_order[split]:
            source = sources[uid]
            selected = selections.get(uid)
            if selected is None:
                missing_by_split[split] += 1
                fallback = dict(source)
                fallback.pop("_split_seq", None)
                fallback["translation_provider"] = "gemini_original_fallback"
                fallback["translation_model"] = str(source.get("translation_model") or "original_v10")
                fallback["translation_disposition"] = "fallback_due_to_luna_infrastructure_failure"
                fallback["needs_followup_review"] = True
                hybrid_rows[split].append(fallback)
                continue

            candidate = selected["candidate"]
            seq = int(source["_split_seq"])
            if int(candidate.get("seq") or -1) != seq:
                # Remaining runs numbered their temporary queue, not the original split.
                # UID is the primary key; final rows retain original source ordering/metadata.
                seq_metadata_normalized += 1
            if str(candidate.get("record_uid") or "") != uid:
                structural_issues.append({"record_uid": uid, "issue": "uid_mismatch"})
            if source.get("prompt_en") == "" and candidate.get("prompt_vi") != "":
                structural_issues.append({"record_uid": uid, "issue": "empty_prompt_not_preserved"})
            if source.get("prompt_en") != "" and not str(candidate.get("prompt_vi") or "").strip():
                structural_issues.append({"record_uid": uid, "issue": "missing_prompt_vi"})
            if source.get("response_en") is None and candidate.get("response_vi") is not None:
                structural_issues.append({"record_uid": uid, "issue": "null_response_not_preserved"})
            if source.get("response_en") == "" and candidate.get("response_vi") != "":
                structural_issues.append({"record_uid": uid, "issue": "empty_response_not_preserved"})
            if source.get("response_en") not in (None, "") and not str(candidate.get("response_vi") or "").strip():
                structural_issues.append({"record_uid": uid, "issue": "missing_response_vi"})

            row = dict(source)
            row.pop("_split_seq", None)
            row["prompt_vi"] = candidate.get("prompt_vi")
            row["response_vi"] = candidate.get("response_vi")
            row["translation_provider"] = selected["provider"]
            row["translation_model"] = (
                "gpt-5.6-sol-web"
                if str(selected["provider"]).startswith("sol_web")
                else "gpt-5.6-luna"
            )
            row["translation_status"] = "translated"
            row["translation_disposition"] = selected["status"]
            row["needs_followup_review"] = selected["status"] in {"warning_pass", "accepted_known_issue"}
            row["validation_warnings"] = (
                [KNOWN_ACCEPTED_ISSUES[uid]] if uid in KNOWN_ACCEPTED_ISSUES else []
            )
            pure_rows[split].append(row)
            hybrid_rows[split].append(row)
            provenance_counts[selected["provider"]] += 1
            quality_counts[selected["status"]] += 1

    for split in SOURCE_PATHS:
        write_jsonl(OUT / f"nemotron_{split}_luna_sol_pure_partial_v1.jsonl", pure_rows[split])
        write_jsonl(OUT / f"nemotron_{split}_luna_sol_gemini_fallback_v1.jsonl", hybrid_rows[split])
        canonical_name = f"nemotron_{split}_en_vi_v10_final.jsonl"
        write_jsonl(HYBRID_FINAL / canonical_name, hybrid_rows[split])
        write_jsonl(PURE_FINAL / canonical_name, pure_rows[split])
        write_jsonl(LUNA_PAIRED_FINAL / canonical_name, pure_rows[split])
        paired_uids = {row["record_uid"] for row in pure_rows[split]}
        gemini_paired = [
            {key: value for key, value in sources[uid].items() if key != "_split_seq"}
            for uid in source_order[split]
            if uid in paired_uids
        ]
        write_jsonl(GEMINI_PAIRED_FINAL / canonical_name, gemini_paired)
    missing_rows = [
        {
            "record_uid": uid,
            "source_split": sources[uid]["source_split"],
            "original_seq": sources[uid]["_split_seq"],
            "prompt_en": sources[uid].get("prompt_en"),
            "response_en": sources[uid].get("response_en"),
            "length_bucket": sources[uid].get("length_bucket"),
            "reason": "no Luna or Sol candidate after infrastructure failures",
        }
        for uid in missing
    ]
    write_jsonl(OUT / "missing_luna_sol_candidates.jsonl", missing_rows)
    write_jsonl(OUT / "structural_issues.jsonl", structural_issues)
    (OUT / "paired_experiment_uids.json").write_text(
        json.dumps({split: [uid for uid in source_order[split] if uid in selections] for split in SOURCE_PATHS}, ensure_ascii=False),
        encoding="utf-8",
    )

    expected_total = sum(split_counts.values())
    pure_total = sum(len(rows) for rows in pure_rows.values())
    hybrid_total = sum(len(rows) for rows in hybrid_rows.values())
    summary = {
        "expected_total": expected_total,
        "expected_by_split": split_counts,
        "unique_source_uids": len(sources),
        "duplicate_source_uids": len(duplicate_source_uids),
        "selected_luna_sol_total": pure_total,
        "selected_by_split": {split: len(rows) for split, rows in pure_rows.items()},
        "coverage_rate": pure_total / expected_total if expected_total else 0,
        "missing_total": len(missing),
        "missing_by_split": dict(missing_by_split),
        "hybrid_total": hybrid_total,
        "hybrid_complete": hybrid_total == expected_total,
        "extra_selected": len(extra_selected),
        "structural_issues": len(structural_issues),
        "temporary_queue_seq_values_normalized": seq_metadata_normalized,
        "uids_with_multiple_candidate_versions": len(conflicting_candidates),
        "provenance_counts": dict(provenance_counts),
        "quality_counts": dict(quality_counts),
        "known_accepted_issue_uids": sorted(KNOWN_ACCEPTED_ISSUES),
        "pure_final_dir": str(PURE_FINAL.relative_to(ROOT)),
        "pure_dataset_train_ready": len(missing) == 0 and not structural_issues and not duplicate_source_uids,
        "hybrid_dataset_train_ready": hybrid_total == expected_total and not structural_issues and not duplicate_source_uids,
    }
    (OUT / "readiness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
