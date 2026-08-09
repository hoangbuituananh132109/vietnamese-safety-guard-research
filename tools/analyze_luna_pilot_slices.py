from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "data/luna_pilot_200"
EVAL_ROOT = PILOT_ROOT / "evaluation"
FINAL_FILES = sorted((ROOT / "data/final").glob("nemotron_*_en_vi_v10_final.jsonl"))

COMMON_DECODED = {
    "a", "about", "all", "and", "are", "as", "at", "be", "because", "by", "can", "could",
    "damn", "do", "does", "explain", "for", "from", "fuck", "fucking", "get", "give", "go",
    "has", "have", "he", "her", "here", "his", "how", "i", "if", "in", "into", "is", "it",
    "like", "make", "me", "more", "my", "no", "not", "of", "on", "one", "or", "our", "out",
    "people", "please", "reveal", "rules", "say", "she", "should", "someone", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "translate", "understand",
    "up", "use", "using", "want", "was", "we", "what", "when", "where", "which", "who", "why",
    "will", "with", "woman", "women", "would", "you", "your",
}
MIXED_TOKEN = re.compile(r"(?i)(?<![a-z0-9])(?=[a-z0-9@$']*[a-z])(?=[a-z0-9@$']*[0-9@$])[a-z0-9@$']{2,}(?![a-z0-9])")
LEET_TRANS = str.maketrans({"4": "a", "3": "e", "0": "o", "1": "i", "5": "s", "7": "t", "@": "a", "$": "s"})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prose_leetspeak(text: str) -> tuple[bool, list[str], list[str]]:
    # Strip regions where mixed alphanumeric tokens are normally identifiers,
    # not prose: URLs, e-mails and long hex hashes.
    cleaned = re.sub(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.\w+|\b[0-9a-f]{16,}\b", " ", text, flags=re.I)
    tokens = MIXED_TOKEN.findall(cleaned)
    tokens = [
        token for token in tokens
        if not re.fullmatch(r"(?i)h[1-6]s?", token)
        and not re.fullmatch(r"(?i)(?:iso)?\d{1,4}[a-z]\d{1,4}", token)
    ]
    decoded = [token.casefold().translate(LEET_TRANS).strip("'") for token in tokens]
    recognized = [word for word in decoded if word in COMMON_DECODED]
    # Two decoded common words are strong evidence of prose. Four or more
    # mixed tokens plus one recognized word catches noisier long leet spans.
    detected = len(recognized) >= 2 or (len(tokens) >= 10 and len(recognized) >= 1)
    return detected, tokens, recognized


def categories(row: dict[str, Any]) -> list[str]:
    return [part.strip() for part in (row.get("violated_categories") or "").split(",") if part.strip()]


def main() -> None:
    pilot = read_jsonl(PILOT_ROOT / "pilot_200.jsonl")
    merged = {row["record_uid"]: row for row in read_jsonl(EVAL_ROOT / "merged_items.jsonl")}
    audit = json.loads((EVAL_ROOT / "manual_disagreement_audit.json").read_text(encoding="utf-8"))
    confirmed_errors = {row["record_uid"] for row in audit["confirmed_luna_errors"]}
    confirmed_false_positives = {
        row["record_uid"] for row in audit["confirmed_evaluator_false_positives_against_luna"]
    }
    partial_advantage = {row["record_uid"] for row in audit["partial_luna_advantage_with_evaluator_false_negative"]}

    details: list[dict[str, Any]] = []
    for row in pilot:
        uid = row["record_uid"]
        source = (row.get("prompt_en") or "") + "\n" + (row.get("response_en") or "")
        has_leet, tokens, recognized = prose_leetspeak(source)
        luna = merged.get(uid)
        if luna is None:
            outcome = "missing_luna"
        elif uid in confirmed_errors:
            outcome = "confirmed_luna_error"
        elif uid in partial_advantage:
            outcome = "partial_luna_advantage_but_imperfect"
        elif uid in confirmed_false_positives:
            outcome = "confirmed_validator_false_positive"
        elif luna["luna_validation"]["flagged"]:
            outcome = "needs_manual_review"
        else:
            outcome = "automatic_clean"
        details.append({
            "pilot_seq": row["pilot_seq"],
            "record_uid": uid,
            "length_bucket": row["length_bucket"],
            "source_chars": len(source) - 1,
            "prompt_label": row.get("prompt_label"),
            "response_label": row.get("response_label"),
            "tag": row.get("tag"),
            "categories": categories(row),
            "prose_leetspeak": has_leet,
            "mixed_tokens": tokens,
            "recognized_deleet_words": recognized,
            "luna_outcome": outcome,
            "luna_completed": luna is not None,
            "luna_validator_flagged": bool(luna and luna["luna_validation"]["flagged"]),
        })

    def table(keys: tuple[str, ...]) -> list[dict[str, Any]]:
        counts = Counter(tuple(str(row[key]) for key in keys) for row in details)
        return [{**dict(zip(keys, values)), "count": count} for values, count in sorted(counts.items())]

    category_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in details:
        row_categories = row["categories"] or ["No violated category"]
        for category in row_categories:
            category_stats[category]["selected"] += 1
            category_stats[category]["leet"] += int(row["prose_leetspeak"])
            category_stats[category]["luna_completed"] += int(row["luna_completed"])
            category_stats[category]["confirmed_luna_error"] += int(row["luna_outcome"] == "confirmed_luna_error")
            category_stats[category]["needs_manual_review"] += int(row["luna_outcome"] == "needs_manual_review")

    summary = {
        "method": {
            "safe_unsafe": "prompt_label from the source dataset; response_label is reported separately and may be null",
            "leetspeak": "At least two mixed letter-digit tokens that decode to common English words, or at least ten mixed tokens with one recognized decoded word; URLs, e-mails, long hashes, H1-H6 headings and ISO-like date/version tokens are excluded",
            "luna_quality": "Confirmed errors/false positives come only from the 22 manually audited validator disagreements; automatic_clean is not a human semantic pass",
        },
        "overall": {
            "selected": len(details),
            "luna_completed": sum(row["luna_completed"] for row in details),
            "prose_leetspeak": sum(row["prose_leetspeak"] for row in details),
            "safe_prompt": sum(row["prompt_label"] == "safe" for row in details),
            "unsafe_prompt": sum(row["prompt_label"] == "unsafe" for row in details),
            "jailbreaking_tag": sum(row["tag"] == "jailbreaking" for row in details),
        },
        "by_length": table(("length_bucket", "luna_outcome")),
        "leet_by_length": table(("length_bucket", "prose_leetspeak")),
        "leet_by_prompt_safety": table(("prompt_label", "prose_leetspeak")),
        "quality_by_prompt_safety": table(("prompt_label", "luna_outcome")),
        "quality_by_tag": table(("tag", "luna_outcome")),
        "response_labels": dict(Counter(str(row["response_label"]) for row in details)),
        "category_stats": {key: dict(value) for key, value in sorted(category_stats.items())},
        "leet_uids": [row["record_uid"] for row in details if row["prose_leetspeak"]],
        "missing_uids": [row["record_uid"] for row in details if not row["luna_completed"]],
    }
    population = Counter()
    for path in FINAL_FILES:
        for row in read_jsonl(path):
            text = (row.get("prompt_en") or row.get("prompt") or "") + "\n" + (row.get("response_en") or row.get("response") or "")
            has_leet, _, _ = prose_leetspeak(text)
            bucket = str(row.get("length_bucket"))
            label = str(row.get("prompt_label"))
            population["selected"] += 1
            population["prose_leetspeak"] += int(has_leet)
            population[f"prompt_{label}"] += 1
            population[f"prompt_{label}_leet"] += int(has_leet)
            population[f"bucket_{bucket}"] += 1
            population[f"bucket_{bucket}_leet"] += int(has_leet)
    summary["full_final_population"] = dict(population)
    out = EVAL_ROOT / "slice_analysis"
    out.mkdir(exist_ok=True)
    (out / "pilot_200_slices.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8"
    )
    (out / "pilot_200_slice_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
