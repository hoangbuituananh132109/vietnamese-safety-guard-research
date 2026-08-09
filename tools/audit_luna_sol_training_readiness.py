from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_smoke.data import iter_full_examples

DATASETS = {
    "gemini_full": ROOT / "data/final",
    "luna_sol_hybrid": ROOT / "data/final_luna_sol_hybrid_v1",
    "luna_sol_pure": ROOT / "data/final_luna_sol_pure_v1",
    "luna_sol_paired": ROOT / "data/final_luna_sol_paired_v1",
    "gemini_paired": ROOT / "data/final_gemini_paired_v1",
}
OUT = ROOT / "data/luna_sol_dataset_v1/materialization_audit.json"


def profile(final_dir: Path) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    all_uids: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        counters: Counter[str] = Counter()
        record_uids: set[str] = set()
        for example in iter_full_examples(final_dir, split):
            counters["examples"] += 1
            counters[f"view:{example.view}"] += 1
            counters[f"language:{example.language}"] += 1
            counters[f"label:{example.safety_label}"] += 1
            counters[f"tag:{example.tag}"] += 1
            record_uids.add(example.record_uid)
        splits[split] = {
            "record_uids": len(record_uids),
            "counters": dict(sorted(counters.items())),
        }
        all_uids[split] = record_uids
    split_overlap = {
        "train_valid": len(all_uids["train"] & all_uids["valid"]),
        "train_test": len(all_uids["train"] & all_uids["test"]),
        "valid_test": len(all_uids["valid"] & all_uids["test"]),
    }
    return {"splits": splits, "split_uid_overlap": split_overlap}


def main() -> None:
    profiles = {name: profile(path) for name, path in DATASETS.items()}
    full_parity = {
        split: profiles["gemini_full"]["splits"][split]["counters"]
        == profiles["luna_sol_hybrid"]["splits"][split]["counters"]
        for split in ("train", "valid", "test")
    }
    pure_parity = {
        split: profiles["gemini_full"]["splits"][split]["counters"]
        == profiles["luna_sol_pure"]["splits"][split]["counters"]
        for split in ("train", "valid", "test")
    }
    paired_parity = {
        split: profiles["gemini_paired"]["splits"][split]["counters"]
        == profiles["luna_sol_paired"]["splits"][split]["counters"]
        for split in ("train", "valid", "test")
    }
    summary = {
        "profiles": profiles,
        "hybrid_vs_original_manifest_parity": full_parity,
        "pure_vs_original_manifest_parity": pure_parity,
        "paired_luna_vs_gemini_manifest_parity": paired_parity,
        "all_split_overlaps_zero": all(
            value == 0
            for profile_value in profiles.values()
            for value in profile_value["split_uid_overlap"].values()
        ),
        "hybrid_materialization_ready": all(full_parity.values()),
        "pure_materialization_ready": all(pure_parity.values()),
        "paired_experiment_materialization_ready": all(paired_parity.values()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "hybrid_manifest_parity": full_parity,
        "pure_manifest_parity": pure_parity,
        "paired_manifest_parity": paired_parity,
        "all_split_overlaps_zero": summary["all_split_overlaps_zero"],
        "hybrid_examples": {
            split: profiles["luna_sol_hybrid"]["splits"][split]["counters"]["examples"]
            for split in ("train", "valid", "test")
        },
        "pure_examples": {
            split: profiles["luna_sol_pure"]["splits"][split]["counters"]["examples"]
            for split in ("train", "valid", "test")
        },
        "paired_examples": {
            split: profiles["luna_sol_paired"]["splits"][split]["counters"]["examples"]
            for split in ("train", "valid", "test")
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
