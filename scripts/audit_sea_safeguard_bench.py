"""Audit the bundled SEA Safeguard Bench without printing sensitive text."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


GROUPS = {
    "general": {
        "glob": "general/data/*_general.jsonl",
        "tasks_per_row": 2,
        "task_names": ["safeguard_general_prompt", "safeguard_general_response"],
    },
    "cultural_in_the_wild": {
        "glob": "cultural_in_the_wild/data/*_cultural_in_the_wild.jsonl",
        "tasks_per_row": 1,
        "task_names": ["safeguard_cultural_in_the_wild"],
    },
    "cultural_content_generation": {
        "glob": "cultural_content_generation/data/*_cultural_content_generation.jsonl",
        "tasks_per_row": 2,
        "task_names": [
            "safeguard_cultural_content_generation_prompt",
            "safeguard_cultural_content_generation_response",
        ],
    },
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def language_from_filename(path: Path) -> str:
    return path.name.split("_", 1)[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    mapping = {
        "safe": "safe",
        "harmful": "harmful",
        "an toàn": "safe",
        "có hại": "harmful",
    }
    return mapping.get(text, text)


def row_text_pair(row: dict[str, Any]) -> tuple[str, str]:
    prompt = (row.get("prompts") or [{}])[0]
    return str(prompt.get("prompt_text") or ""), str(prompt.get("response_text") or "")


def summarize_file(path: Path, group: str, tasks_per_row: int) -> tuple[dict, list[dict]]:
    rows = list(iter_jsonl(path))
    top_keys: Counter[str] = Counter()
    prompt_keys: Counter[str] = Counter()
    metadata_keys: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {
        "label": Counter(),
        "prompt_label": Counter(),
        "response_label": Counter(),
    }
    missing = Counter()
    metadata_languages = Counter()
    embedded_en_prompt = 0
    embedded_en_response = 0
    for row in rows:
        top_keys.update(row.keys())
        prompts = row.get("prompts") or []
        if not prompts:
            missing["prompts"] += 1
        else:
            prompt = prompts[0]
            prompt_keys.update(prompt.keys())
            if not any(
                str(prompt.get(key) or "").strip()
                for key in ("prompt_text", "local_prompt", "en_prompt")
            ):
                missing["prompt_text"] += 1
            if str(prompt.get("en_prompt") or "").strip():
                embedded_en_prompt += 1
        metadata = row.get("metadata") or {}
        metadata_keys.update(metadata.keys())
        metadata_languages[str(metadata.get("language") or "missing")] += 1
        if str(metadata.get("en_prompt") or "").strip():
            embedded_en_prompt += 1
        if str(metadata.get("en_response") or "").strip():
            embedded_en_response += 1
        for key, counter in label_counts.items():
            if key in row and str(row[key]).strip():
                counter[str(row[key])] += 1
            elif key in row:
                missing[key] += 1

    summary = {
        "group": group,
        "language": language_from_filename(path),
        "path": str(path),
        "rows": len(rows),
        "registered_task_instances": len(rows) * tasks_per_row,
        "sha256": sha256_file(path),
        "top_level_keys": sorted(top_keys),
        "prompt_object_keys": sorted(prompt_keys),
        "metadata_keys": sorted(metadata_keys),
        "metadata_languages": dict(sorted(metadata_languages.items())),
        "label_counts": {
            key: dict(sorted(counter.items()))
            for key, counter in label_counts.items()
            if counter
        },
        "embedded_english": {
            "prompt_rows": embedded_en_prompt,
            "response_rows": embedded_en_response,
        },
        "missing": dict(sorted(missing.items())),
    }
    return summary, rows


def analyze_pairing(rows_by_group_language: dict[tuple[str, str], list[dict]]) -> dict:
    pairing: dict[str, Any] = {}

    general_en = rows_by_group_language.get(("general", "en"), [])
    general_vi = rows_by_group_language.get(("general", "vi"), [])
    aligned = min(len(general_en), len(general_vi))
    prompt_label_matches = 0
    response_label_matches = 0
    topic_matches = 0
    for en_row, vi_row in zip(general_en, general_vi):
        prompt_label_matches += normalized_label(en_row.get("prompt_label")) == normalized_label(
            vi_row.get("prompt_label")
        )
        response_label_matches += normalized_label(
            en_row.get("response_label")
        ) == normalized_label(vi_row.get("response_label"))
        topic_matches += (en_row.get("metadata") or {}).get("topic") == (
            vi_row.get("metadata") or {}
        ).get("topic")
    pairing["general_en_vi"] = {
        "method_available": "row-index inference only; no explicit pair id or embedded English fields",
        "en_rows": len(general_en),
        "vi_rows": len(general_vi),
        "aligned_rows": aligned,
        "prompt_label_matches": prompt_label_matches,
        "response_label_matches": response_label_matches,
        "topic_matches": topic_matches,
        "safe_for_paired_bootstrap_without_derived_pair_id": False,
    }

    content_en = rows_by_group_language.get(("cultural_content_generation", "en"), [])
    content_vi = rows_by_group_language.get(("cultural_content_generation", "vi"), [])
    english_lookup = Counter(row_text_pair(row) for row in content_en)
    explicit_pairs = []
    for row in content_vi:
        metadata = row.get("metadata") or {}
        pair = (str(metadata.get("en_prompt") or ""), str(metadata.get("en_response") or ""))
        explicit_pairs.append(pair)
    pairing["cultural_content_generation_en_vi"] = {
        "method_available": "explicit metadata.en_prompt + metadata.en_response in VI rows",
        "en_rows": len(content_en),
        "vi_rows": len(content_vi),
        "vi_rows_with_explicit_english_pair": sum(bool(a and b) for a, b in explicit_pairs),
        "explicit_pairs_found_in_en_file": sum(english_lookup[pair] > 0 for pair in explicit_pairs),
        "unique_explicit_english_pairs": len(set(explicit_pairs)),
        "safe_for_paired_bootstrap_without_derived_pair_id": True,
    }

    wild_vi = rows_by_group_language.get(("cultural_in_the_wild", "vi"), [])
    pairing["cultural_in_the_wild_en_vi"] = {
        "method_available": "explicit prompts[0].en_prompt embedded beside prompts[0].local_prompt",
        "standalone_en_file": False,
        "vi_rows": len(wild_vi),
        "vi_rows_with_embedded_en_prompt": sum(
            bool(str(((row.get("prompts") or [{}])[0]).get("en_prompt") or "").strip())
            for row in wild_vi
        ),
        "registered_as_english_task_in_task_config": False,
        "safe_for_paired_bootstrap_without_derived_pair_id": True,
    }
    return pairing


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# SEA Safeguard Bench local audit",
        "",
        f"Repository root: `{report['repository_root']}`",
        f"Archive SHA-256: `{report.get('archive_sha256') or 'not found'}`",
        "",
        "## Bundled files and registered task instances",
        "",
        "| Group | Lang | JSON rows | Registered task instances | Embedded EN prompt | Embedded EN response |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["files"]:
        lines.append(
            f"| {item['group']} | {item['language']} | {item['rows']} | "
            f"{item['registered_task_instances']} | "
            f"{item['embedded_english']['prompt_rows']} | "
            f"{item['embedded_english']['response_rows']} |"
        )
    totals = report["totals"]
    lines.extend(
        [
            "",
            f"Total bundled JSON rows: **{totals['json_rows']:,}**.",
            f"Total registered evaluation task instances: **{totals['registered_task_instances']:,}**.",
            "",
            "## EN–VI registered sizes",
            "",
            f"- English: {totals['registered_by_language'].get('en', 0):,} task instances.",
            f"- Vietnamese: {totals['registered_by_language'].get('vi', 0):,} task instances.",
            "",
            "## Pairing findings",
            "",
            "```json",
            json.dumps(report["pairing"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Access conclusion",
            "",
            "The benchmark data is bundled as local JSONL files. The task configs point to local paths, and SeaHelmLocalDataloader calls datasets.load_dataset('json', split='train', data_files=filepath). No Hugging Face dataset ID, gated token, or separate download is required for this archive.",
            "",
            "The loader's `train` split is a technical name created by the Hugging Face JSON loader; these files are evaluation data, not model-training data.",
            "",
            "License metadata is inconsistent: the repository root LICENSE is MIT, while pyproject.toml declares Apache-2.0, and the safeguard data folders have no dedicated dataset license file. Research evaluation is technically possible from the bundled files, but redistribution terms should be confirmed before republishing the dataset or derived text.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path("external/SEA-HELM-main")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/sea_safeguard_bench_audit.json")
    )
    parser.add_argument(
        "--output-md", type=Path, default=Path("reports/sea_safeguard_bench_audit.md")
    )
    args = parser.parse_args()
    safeguard_root = args.repo_root / "seahelm_tasks" / "safety" / "safeguard"
    if not safeguard_root.is_dir():
        raise FileNotFoundError(safeguard_root)

    files: list[dict[str, Any]] = []
    rows_by_group_language: dict[tuple[str, str], list[dict]] = {}
    for group, config in GROUPS.items():
        for path in sorted(safeguard_root.glob(config["glob"])):
            summary, rows = summarize_file(path, group, config["tasks_per_row"])
            files.append(summary)
            rows_by_group_language[(group, summary["language"])] = rows

    registered_by_language: Counter[str] = Counter()
    registered_by_group: Counter[str] = Counter()
    for item in files:
        registered_by_language[item["language"]] += item["registered_task_instances"]
        registered_by_group[item["group"]] += item["registered_task_instances"]

    archive = args.repo_root.parents[1] / "SEA-HELM-main.zip"
    report = {
        "repository_root": str(args.repo_root),
        "archive_sha256": sha256_file(archive) if archive.exists() else None,
        "repository_identity": {
            "zip_has_git_metadata": (args.repo_root / ".git").exists(),
            "readme_update": "19 Jun 2026: Added SEA Safeguard Bench",
            "project_version": "1.2.0",
        },
        "task_names": {
            group: config["task_names"] for group, config in GROUPS.items()
        },
        "files": files,
        "totals": {
            "json_rows": sum(item["rows"] for item in files),
            "registered_task_instances": sum(
                item["registered_task_instances"] for item in files
            ),
            "registered_by_language": dict(sorted(registered_by_language.items())),
            "registered_by_group": dict(sorted(registered_by_group.items())),
        },
        "pairing": analyze_pairing(rows_by_group_language),
        "access": {
            "bundled_locally": True,
            "huggingface_dataset_id": None,
            "gated": False,
            "separate_download_required": False,
            "loader": "datasets.load_dataset('json', split='train', data_files=filepath)",
            "split_note": "train is the local JSON loader split name; data is used for evaluation",
            "license_caveat": (
                "Root LICENSE is MIT, pyproject.toml declares Apache-2.0, and "
                "no dedicated safeguard dataset license file was found"
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))
    print(json.dumps(report["pairing"], ensure_ascii=False, indent=2))
    print(f"json={args.output_json}")
    print(f"markdown={args.output_md}")


if __name__ == "__main__":
    main()
