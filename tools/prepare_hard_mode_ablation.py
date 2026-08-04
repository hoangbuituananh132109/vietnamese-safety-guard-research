from __future__ import annotations

import json
import pathlib
import random
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.luna_overnight_runner import length_bucket, route_for

OUT = ROOT / "data" / "luna_hard_ablation_100"


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    source_paths = sorted((ROOT / "data" / "final").glob("nemotron_*_en_vi_v10_final.jsonl"))
    source: dict[str, dict] = {}
    for path in source_paths:
        for row in load(path):
            source[str(row["record_uid"])] = row

    run_dirs = [
        ROOT / "data/luna_overnight/nemotron_train_20260801",
        ROOT / "data/luna_overnight/nemotron_valid_20260802",
        ROOT / "data/luna_overnight/nemotron_test_20260802",
    ]
    hard_uids: set[str] = set()
    evidence: dict[str, list[str]] = {}
    for run_dir in run_dirs:
        for filename, reason in (
            ("exhausted_normal.jsonl", "exhausted_after_retry"),
            ("tail_escalation.jsonl", "tail_escalation"),
            ("needs_audit.jsonl", "audit_candidate"),
        ):
            path = run_dir / filename
            if not path.exists():
                continue
            for row in load(path):
                uid = str(row.get("record_uid") or "")
                if uid in source:
                    hard_uids.add(uid)
                    evidence.setdefault(uid, []).append(reason)

        attempts = run_dir / "attempts.jsonl"
        if attempts.exists():
            for row in load(attempts):
                uid = str(row.get("record_uid") or "")
                hard = row.get("hard_errors") or []
                if uid in source and hard:
                    hard_uids.add(uid)
                    evidence.setdefault(uid, []).append("hard_validator_error")

    # Add pending high-tail/structured records, which are hard because they never completed, not because
    # they already have a model error. They are essential to the mode comparison.
    for path in source_paths:
        split = path.stem.split("_")[1]
        if split not in {"train", "valid", "test"}:
            continue
        output = ROOT / "data/luna_overnight" / f"nemotron_{split}_20260802"
        if split == "train":
            output = ROOT / "data/luna_overnight/nemotron_train_20260801"
        terminal: set[str] = set()
        for filename in ("passed.jsonl", "needs_audit.jsonl", "exhausted_normal.jsonl", "tail_escalation.jsonl"):
            p = output / filename
            if p.exists():
                terminal.update(str(x.get("record_uid")) for x in load(p))
        for row in load(path):
            uid = str(row["record_uid"])
            if uid not in terminal and length_bucket(row) in {"high_tail", "oversized"}:
                hard_uids.add(uid)
                evidence.setdefault(uid, []).append("pending_long_record")

    candidates = [source[uid] for uid in sorted(hard_uids) if uid in source]
    groups: dict[str, list[dict]] = {"high_tail": [], "json": [], "leet": [], "tail_prose": [], "normal_error": []}
    for row in candidates:
        bucket = length_bucket(row)
        route = route_for(row)
        if bucket in {"high_tail", "oversized"}:
            groups["high_tail"].append(row)
        elif route == "json":
            groups["json"].append(row)
        elif route == "leet":
            groups["leet"].append(row)
        elif bucket == "tail":
            groups["tail_prose"].append(row)
        else:
            groups["normal_error"].append(row)

    # Deterministic quotas: 35 long, 25 JSON, 20 leet, 15 tail prose, 5 normal hard errors.
    quotas = {"high_tail": 35, "json": 25, "leet": 20, "tail_prose": 15, "normal_error": 5}
    selected: list[dict] = []
    used: set[str] = set()
    rng = random.Random(560200)
    for group, quota in quotas.items():
        pool = [row for row in groups[group] if row["record_uid"] not in used]
        pool.sort(key=lambda row: (-(len(row.get("prompt_en") or "") + len(row.get("response_en") or "")), rng.random()))
        for row in pool[:quota]:
            used.add(row["record_uid"])
            selected.append(row)
    if len(selected) < 100:
        remainder = [row for row in candidates if row["record_uid"] not in used]
        remainder.sort(key=lambda row: (length_bucket(row), row["record_uid"]))
        selected.extend(remainder[:100 - len(selected)])
    selected = selected[:100]
    selected.sort(key=lambda row: row["record_uid"])

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq, row in enumerate(selected, 1):
        uid = str(row["record_uid"])
        prompt = row.get("prompt_en") or row.get("prompt") or ""
        response = row.get("response_en") if "response_en" in row else row.get("response")
        rows.append({
            "ablation_seq": seq, "record_uid": uid, "source_split": row.get("source_split"),
            "prompt_en": prompt, "response_en": response, "length_bucket": length_bucket(row),
            "route": route_for(row), "tag": row.get("tag"), "prompt_label": row.get("prompt_label"),
            "response_label": row.get("response_label"), "violated_categories": row.get("violated_categories"),
            "evidence": sorted(set(evidence.get(uid, []))),
        })
    for filename in ("ablation_100.jsonl", "luna_medium_100.jsonl", "luna_high_100.jsonl", "luna_xhigh_100.jsonl", "sol_web_100.jsonl"):
        (OUT / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    for index in range(0, len(rows), 20):
        batch = rows[index:index + 20]
        (OUT / f"sol_web_batch_{index // 20 + 1:02d}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in batch), encoding="utf-8",
        )
    manifest = {
        "records": len(rows), "seed": 560200, "quotas": quotas,
        "groups_selected": dict(Counter(
            "high_tail" if row["length_bucket"] in {"high_tail", "oversized"}
            else row["route"] if row["route"] in {"json", "leet"}
            else "tail_prose" if row["length_bucket"] == "tail" else "normal_error"
            for row in rows
        )),
        "modes": ["low_reference", "medium", "high", "xhigh_or_extra_high", "sol_web"],
        "uid_sha256": __import__("hashlib").sha256("\n".join(row["record_uid"] for row in rows).encode()).hexdigest(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
