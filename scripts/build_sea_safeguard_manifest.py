"""Build encoder-friendly SEA Safeguard Bench manifests from bundled JSONL.

The source SEA-HELM tasks are generative Safe/Harmful classification tasks. This
adapter preserves their target semantics while exposing P and PR views directly
for encoder guard inference. It also creates a separate EN-VI paired diagnostic
manifest; derived English counterparts are never presented as official tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def short_hash(value: str) -> str:
    return hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()[:16]


def normalize_label(value: object) -> str:
    label = normalized_text(str(value or ""))
    mapping = {
        "safe": "safe",
        "harmful": "unsafe",
        "an toàn": "safe",
        "có hại": "unsafe",
    }
    if label not in mapping:
        raise ValueError(f"Unsupported SEA Safeguard label: {value!r}")
    return mapping[label]


def serialize(view: str, prompt: str, response: str | None) -> str:
    if view == "P":
        return f"Prompt: {prompt}"
    if view == "PR" and response is not None:
        return f"Prompt: {prompt}\nResponse: {response}"
    raise ValueError(f"Invalid view/response combination: {view}/{response is None}")


def source_ref(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def make_example(
    *,
    subset: str,
    language: str,
    view: str,
    prompt: str,
    response: str | None,
    source_label: str,
    official_registered: bool,
    pair_uid: str | None,
    pairing_provenance: str | None,
    source_path: str,
    source_line: int,
    topic: str | None,
    identity_suffix: str,
) -> dict[str, Any]:
    text = serialize(view, prompt, response)
    safety_scope = "prompt" if view == "P" else "response"
    example = {
        "example_id": f"sea:{subset}:{identity_suffix}:{view}:{language}",
        "benchmark": "sea_safeguard_bench",
        "subset": subset,
        "language": language,
        "view": view,
        "safety_scope": safety_scope,
        "prompt": prompt,
        "response": response,
        "text": text,
        "safety_label": normalize_label(source_label),
        "source_label": source_label,
        "official_registered": official_registered,
        "pair_uid": pair_uid,
        "pairing_provenance": pairing_provenance,
        "source_path": source_path,
        "source_line": source_line,
        "topic": topic,
        "char_count": len(text),
        "whitespace_tokens": len(text.split()),
    }
    if not prompt.strip() or (view == "PR" and not str(response or "").strip()):
        raise ValueError(f"Empty required text in {example['example_id']}")
    return example


def row_prompt_response(row: dict[str, Any]) -> tuple[str, str]:
    prompt = (row.get("prompts") or [{}])[0]
    return str(prompt.get("prompt_text") or ""), str(prompt.get("response_text") or "")


def load_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    return list(iter_jsonl(path))


def build_official(repo_root: Path, safeguard_root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    specifications = [
        ("general", "general/data/en_general.jsonl", "en", ("P", "PR")),
        ("general", "general/data/vi_general.jsonl", "vi", ("P", "PR")),
        (
            "cultural_content_generation",
            "cultural_content_generation/data/en_cultural_content_generation.jsonl",
            "en",
            ("P", "PR"),
        ),
        (
            "cultural_content_generation",
            "cultural_content_generation/data/vi_cultural_content_generation.jsonl",
            "vi",
            ("P", "PR"),
        ),
        (
            "cultural_in_the_wild",
            "cultural_in_the_wild/data/vi_cultural_in_the_wild.jsonl",
            "vi",
            ("P",),
        ),
    ]
    for subset, relative, language, views in specifications:
        path = safeguard_root / relative
        for line_number, row in iter_jsonl(path):
            metadata = row.get("metadata") or {}
            if subset == "cultural_in_the_wild":
                prompts = (row.get("prompts") or [{}])[0]
                prompt, response = str(prompts.get("local_prompt") or ""), None
                labels = {"P": str(row.get("label") or "")}
            else:
                prompt, response_text = row_prompt_response(row)
                response = response_text
                labels = {
                    "P": str(row.get("prompt_label") or ""),
                    "PR": str(row.get("response_label") or ""),
                }
            for view in views:
                pair_uid = None
                provenance = None
                if subset == "general":
                    pair_uid = f"sea:general:{line_number:04d}:{view}"
                    provenance = "inferred_same_row_index_verified_labels_and_topic"
                elif language == "vi":
                    english_prompt = str(metadata.get("en_prompt") or "")
                    if subset == "cultural_in_the_wild":
                        english_prompt = str(
                            ((row.get("prompts") or [{}])[0]).get("en_prompt") or ""
                        )
                    english_response = (
                        str(metadata.get("en_response") or "") if view == "PR" else ""
                    )
                    anchor = f"{view}\n{english_prompt}\n{english_response}"
                    pair_uid = f"sea:{subset}:{short_hash(anchor)}:{view}"
                    provenance = "explicit_embedded_english"
                output.append(
                    make_example(
                        subset=subset,
                        language=language,
                        view=view,
                        prompt=prompt,
                        response=response if view == "PR" else None,
                        source_label=labels[view],
                        official_registered=True,
                        pair_uid=pair_uid,
                        pairing_provenance=provenance,
                        source_path=source_ref(repo_root, path),
                        source_line=line_number,
                        topic=str(metadata.get("topic") or "") or None,
                        identity_suffix=f"official-{line_number:04d}",
                    )
                )
    return output


def build_paired(repo_root: Path, safeguard_root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    general_en_path = safeguard_root / "general/data/en_general.jsonl"
    general_vi_path = safeguard_root / "general/data/vi_general.jsonl"
    en_rows = load_rows(general_en_path)
    vi_rows = load_rows(general_vi_path)
    if len(en_rows) != len(vi_rows):
        raise ValueError("General EN and VI row counts differ; row-index pairing is unsafe")
    for (en_line, en_row), (vi_line, vi_row) in zip(en_rows, vi_rows):
        en_meta, vi_meta = en_row.get("metadata") or {}, vi_row.get("metadata") or {}
        if en_meta.get("topic") != vi_meta.get("topic"):
            raise ValueError(f"General topic mismatch at rows {en_line}/{vi_line}")
        en_prompt, en_response = row_prompt_response(en_row)
        vi_prompt, vi_response = row_prompt_response(vi_row)
        for view, label_key in (("P", "prompt_label"), ("PR", "response_label")):
            en_label, vi_label = en_row.get(label_key), vi_row.get(label_key)
            if normalize_label(en_label) != normalize_label(vi_label):
                raise ValueError(f"General label mismatch at row {en_line}, view {view}")
            pair_uid = f"sea:general:{en_line:04d}:{view}"
            for language, prompt, response, label, path, line_number in (
                ("en", en_prompt, en_response, en_label, general_en_path, en_line),
                ("vi", vi_prompt, vi_response, vi_label, general_vi_path, vi_line),
            ):
                output.append(
                    make_example(
                        subset="general",
                        language=language,
                        view=view,
                        prompt=prompt,
                        response=response if view == "PR" else None,
                        source_label=str(label or ""),
                        official_registered=True,
                        pair_uid=pair_uid,
                        pairing_provenance="inferred_same_row_index_verified_labels_and_topic",
                        source_path=source_ref(repo_root, path),
                        source_line=line_number,
                        topic=str(en_meta.get("topic") or "") or None,
                        identity_suffix=f"pair-{en_line:04d}",
                    )
                )

    content_path = (
        safeguard_root
        / "cultural_content_generation/data/vi_cultural_content_generation.jsonl"
    )
    for line_number, row in iter_jsonl(content_path):
        metadata = row.get("metadata") or {}
        vi_prompt, vi_response = row_prompt_response(row)
        en_prompt = str(metadata.get("en_prompt") or "")
        en_response = str(metadata.get("en_response") or "")
        for view, label_key in (("P", "prompt_label"), ("PR", "response_label")):
            label = str(row.get(label_key) or "")
            pair_anchor = f"{view}\n{en_prompt}\n{en_response if view == 'PR' else ''}"
            pair_uid = f"sea:cultural_content_generation:{short_hash(pair_anchor)}:{view}"
            for language, prompt, response, official in (
                ("en", en_prompt, en_response, False),
                ("vi", vi_prompt, vi_response, True),
            ):
                output.append(
                    make_example(
                        subset="cultural_content_generation",
                        language=language,
                        view=view,
                        prompt=prompt,
                        response=response if view == "PR" else None,
                        source_label=label,
                        official_registered=official,
                        pair_uid=pair_uid,
                        pairing_provenance="explicit_embedded_english",
                        source_path=source_ref(repo_root, content_path),
                        source_line=line_number,
                        topic=str(metadata.get("topic") or "") or None,
                        identity_suffix=f"pair-{line_number:04d}",
                    )
                )

    wild_path = safeguard_root / "cultural_in_the_wild/data/vi_cultural_in_the_wild.jsonl"
    for line_number, row in iter_jsonl(wild_path):
        metadata = row.get("metadata") or {}
        prompts = (row.get("prompts") or [{}])[0]
        en_prompt = str(prompts.get("en_prompt") or "")
        vi_prompt = str(prompts.get("local_prompt") or "")
        label = str(row.get("label") or "")
        pair_uid = f"sea:cultural_in_the_wild:{short_hash(en_prompt)}:P"
        for language, prompt, official in (
            ("en", en_prompt, False),
            ("vi", vi_prompt, True),
        ):
            output.append(
                make_example(
                    subset="cultural_in_the_wild",
                    language=language,
                    view="P",
                    prompt=prompt,
                    response=None,
                    source_label=label,
                    official_registered=official,
                    pair_uid=pair_uid,
                    pairing_provenance="explicit_embedded_english",
                    source_path=source_ref(repo_root, wild_path),
                    source_line=line_number,
                    topic=str(metadata.get("topic") or "") or None,
                    identity_suffix=f"pair-{line_number:04d}",
                )
            )
    return output


def validate_official(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 3430:
        raise ValueError(f"Expected 3,430 official EN/VI instances, got {len(rows):,}")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate example_id in official manifest")
    if not all(row["official_registered"] for row in rows):
        raise ValueError("Unofficial example leaked into official manifest")


def validate_paired(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 3680:
        raise ValueError(f"Expected 3,680 paired instances, got {len(rows):,}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["pair_uid"]), []).append(row)
    if len(groups) != 1840:
        raise ValueError(f"Expected 1,840 pair units, got {len(groups):,}")
    for pair_uid, pair in groups.items():
        if {row["language"] for row in pair} != {"en", "vi"} or len(pair) != 2:
            raise ValueError(f"Malformed language pair: {pair_uid}")
        if len({row["safety_label"] for row in pair}) != 1:
            raise ValueError(f"Cross-language target mismatch: {pair_uid}")
        if len({row["view"] for row in pair}) != 1:
            raise ValueError(f"Cross-language view mismatch: {pair_uid}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["examples"] += 1
        for key in ("subset", "language", "view", "safety_label"):
            counts[f"{key}:{row[key]}"] += 1
        counts[f"cell:{row['subset']}|{row['language']}|{row['view']}|{row['safety_label']}"] += 1
        counts[f"official_registered:{str(row['official_registered']).lower()}"] += 1
    char_lengths = sorted(row["char_count"] for row in rows)
    word_lengths = sorted(row["whitespace_tokens"] for row in rows)

    def percentile(values: list[int], proportion: float) -> int:
        return values[min(len(values) - 1, round((len(values) - 1) * proportion))]

    return {
        "counts": dict(sorted(counts.items())),
        "length_proxy": {
            "characters": {
                "p50": percentile(char_lengths, 0.50),
                "p95": percentile(char_lengths, 0.95),
                "p99": percentile(char_lengths, 0.99),
                "max": char_lengths[-1],
            },
            "whitespace_tokens_not_model_tokens": {
                "p50": percentile(word_lengths, 0.50),
                "p95": percentile(word_lengths, 0.95),
                "p99": percentile(word_lengths, 0.99),
                "max": word_lengths[-1],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("external/SEA-HELM-main"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/benchmarks/sea_safeguard")
    )
    args = parser.parse_args()
    safeguard_root = args.repo_root / "seahelm_tasks/safety/safeguard"
    if not safeguard_root.is_dir():
        raise FileNotFoundError(safeguard_root)

    official = build_official(args.repo_root, safeguard_root)
    paired = build_paired(args.repo_root, safeguard_root)
    validate_official(official)
    validate_paired(paired)

    official_path = args.output_dir / "official_en_vi.jsonl"
    paired_path = args.output_dir / "paired_en_vi.jsonl"
    write_jsonl(official_path, official)
    write_jsonl(paired_path, paired)
    archive = args.repo_root.parents[1] / "SEA-HELM-main.zip"
    summary = {
        "source": {
            "repository_root": str(args.repo_root),
            "archive_sha256": sha256_file(archive) if archive.exists() else None,
            "task_group": "sea_safeguard",
        },
        "official_en_vi": summarize(official),
        "paired_en_vi": {
            **summarize(paired),
            "pair_units": len({row["pair_uid"] for row in paired}),
            "pairing_caveat": (
                "General uses row-index inference after exact label/topic alignment; "
                "cultural subsets contain explicit embedded English counterparts."
            ),
        },
        "semantics": {
            "P": "prompt-only input scored against prompt safety",
            "PR": "prompt plus response input scored against response safety",
            "R_only_official": False,
            "labels": ["safe", "unsafe"],
            "sensitive_label_present": False,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"official={official_path}")
    print(f"paired={paired_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
