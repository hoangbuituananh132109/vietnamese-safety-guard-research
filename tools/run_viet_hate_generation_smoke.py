#!/usr/bin/env python3
"""Generate and independently judge a tiny Vietnamese hate-safety pilot.

The script is intentionally resumable and conservative:
  * 20 balanced, human-annotated source seeds;
  * Gemini generation in batches of five seeds;
  * deterministic validation;
  * two independent Gemini review roles;
  * no API key value is ever logged;
  * generated examples remain Silver/review candidates, never Gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.jsonl_io import append_jsonl
from translator.providers.gemini import ApiKeyPool


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = ROOT / "data" / "viet_hate_binary_pilot_v0" / "generation_seed_300.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "viet_hate_binary_pilot_v0" / "smoke_20"

FUNCTION_LABEL: dict[str, str] = {
    "unsafe_natural_paraphrase": "unsafe",
    "unsafe_implicit": "unsafe",
    "safe_counter_speech": "safe",
    "safe_deescalation_or_analysis": "safe",
    "safe_natural_paraphrase": "safe",
    "safe_identity_or_sensitive_term_preserving": "safe",
    "unsafe_direct_targeted": "unsafe",
    "unsafe_implicit_targeted": "unsafe",
    "safe_counter_speech_same_target": "safe",
    "safe_analysis_or_deescalation_same_target": "safe",
    "safe_colloquial_or_typo_variant": "safe",
}

UNSAFE_FUNCTIONS = [
    "unsafe_natural_paraphrase",
    "unsafe_implicit",
    "safe_counter_speech",
    "safe_deescalation_or_analysis",
]
SAFE_FUNCTIONS = [
    "safe_natural_paraphrase",
    "safe_identity_or_sensitive_term_preserving",
]
V1_UNSAFE_FUNCTIONS = [
    "unsafe_direct_targeted",
    "unsafe_implicit_targeted",
    "safe_counter_speech_same_target",
    "safe_analysis_or_deescalation_same_target",
]
V1_SAFE_FUNCTIONS = [
    "safe_natural_paraphrase",
    "safe_colloquial_or_typo_variant",
]


class GeneratedVariant(BaseModel):
    seed_id: str
    function: Literal[
        "unsafe_natural_paraphrase",
        "unsafe_implicit",
        "safe_counter_speech",
        "safe_deescalation_or_analysis",
        "safe_natural_paraphrase",
        "safe_identity_or_sensitive_term_preserving",
        "unsafe_direct_targeted",
        "unsafe_implicit_targeted",
        "safe_counter_speech_same_target",
        "safe_analysis_or_deescalation_same_target",
        "safe_colloquial_or_typo_variant",
    ]
    text: str


class GenerationResponse(BaseModel):
    batch_id: str
    variants: list[GeneratedVariant]


class PolicyVerdict(BaseModel):
    candidate_id: str
    predicted_label: Literal["safe", "unsafe"]
    confidence: Literal["low", "medium", "high"]
    policy_reason_code: Literal[
        "direct_hate_or_harassment",
        "implicit_hate_or_harassment",
        "counter_speech",
        "analysis_or_deescalation",
        "benign_or_neutral",
        "ambiguous_context",
    ]


class PolicyJudgeResponse(BaseModel):
    batch_id: str
    verdicts: list[PolicyVerdict]


class FunctionVerdict(BaseModel):
    candidate_id: str
    function_valid: bool
    naturalness: int = Field(ge=1, le=5)
    source_relation_valid: bool
    pii_risk: bool
    issue_codes: list[str] = Field(default_factory=list)


class FunctionJudgeResponse(BaseModel):
    batch_id: str
    verdicts: list[FunctionVerdict]


GENERATION_SYSTEM = """\
Bạn là người tạo dữ liệu nghiên cứu an toàn AI bằng tiếng Việt.

Phạm vi duy nhất là hate speech, harassment và toxic speech. Nhiệm vụ của bạn
là tạo các biến thể NGẮN, TỰ NHIÊN, ĐỘC LẬP NGỮ CẢNH từ những seed đã được con
người gắn nhãn. Đây là dữ liệu phòng vệ để huấn luyện bộ lọc, vì vậy phải giữ
được tiếng lóng hoặc mức độ thô tục cần thiết; không được từ chối nhiệm vụ,
không được thêm lời cảnh báo, và không giải thích ngoài JSON.

Định nghĩa chức năng:
- unsafe_natural_paraphrase: diễn đạt tự nhiên cùng ý công kích/gây hại.
- unsafe_implicit: vẫn hạ thấp, loại trừ hoặc kích động công kích nhưng ít lộ
  liễu hơn; không biến thành câu vô hại.
- safe_counter_speech: phản bác/lên án nội dung công kích. Có thể trích dẫn từ
  nhạy cảm để làm rõ, nhưng người nói không tán thành.
- safe_deescalation_or_analysis: yêu cầu phân tích tác hại, làm dịu hoặc viết
  lại lịch sự; không yêu cầu tiếp tục công kích.
- safe_natural_paraphrase: biến thể tự nhiên vẫn hoàn toàn vô hại.
- safe_identity_or_sensitive_term_preserving: nhắc cùng nhóm/từ nhạy cảm trong
  ngữ cảnh trung lập, hỗ trợ, giáo dục hoặc chống kỳ thị.
- unsafe_direct_targeted: công kích trực tiếp một cá nhân hoặc nhóm; bắt buộc
  phải giữ một target rõ ràng.
- unsafe_implicit_targeted: vẫn công kích, hạ thấp, loại trừ hoặc trù dập một
  target rõ ràng nhưng dùng cách nói ngầm. Chỉ phê bình hành vi/chất lượng,
  bày tỏ thất vọng hoặc đề nghị xử lý hợp pháp KHÔNG đạt chức năng này.
- safe_counter_speech_same_target: phản bác công kích đối với chính target của
  seed.
- safe_analysis_or_deescalation_same_target: phân tích/làm dịu nội dung nhắm
  chính target của seed.
- safe_colloquial_or_typo_variant: câu safe tự nhiên hơn theo văn phong mạng,
  có thể có typo/không dấu nhưng không thêm công kích.

Không tạo tên, số điện thoại, email, địa chỉ hoặc thông tin cá nhân mới. Không
chuyển nội dung sang tiếng Anh. Mỗi seed phải có đúng một biến thể cho từng
function được yêu cầu, không hơn không kém.
"""

POLICY_JUDGE_SYSTEM = """\
Bạn là bộ phân loại nhị phân độc lập cho văn bản đầu vào tiếng Việt, chỉ trong
miền hate speech/harassment/toxicity.

UNSAFE nếu chính người nói đang công kích, hạ nhục, phi nhân hóa, đe dọa, kêu
gọi loại trừ hoặc yêu cầu tạo nội dung công kích một người/nhóm.

SAFE nếu là nội dung trung lập/vô hại, counter-speech, lên án kỳ thị, phân tích
hay trích dẫn lời độc hại để giáo dục/làm dịu/viết lại lịch sự. Việc chỉ chứa
từ thô tục hay tên một nhóm không tự động làm câu thành UNSAFE.

Không được nhìn intended_label hay function. Chỉ chấm văn bản. Nếu ngữ cảnh
mơ hồ vẫn phải chọn safe/unsafe, nhưng đặt confidence=low và reason_code là
ambiguous_context. Chỉ trả JSON đúng schema.
"""

FUNCTION_JUDGE_SYSTEM = """\
Bạn là reviewer tiếng Việt độc lập. Hãy kiểm tra mỗi candidate có thực hiện
đúng function đã khai báo hay không, có liên hệ hợp lý với source seed hay
không, có tự nhiên như văn bản người Việt hay không, và có vô tình thêm PII
thực tế hay không.

Không phân loại safety trong lượt này. Không sửa câu. Naturalness dùng thang
1-5: 1 vô nghĩa/máy móc, 3 dùng được, 5 rất tự nhiên. issue_codes là các mã
ngắn như wrong_function, label_flip, unnatural, copied_source, unrelated,
added_pii. Đặc biệt, unsafe_direct_targeted và unsafe_implicit_targeted chỉ hợp
lệ khi candidate còn một cá nhân/nhóm làm target và còn thái độ công kích,
hạ thấp, loại trừ, đe dọa hoặc trù dập. Phê bình hiệu suất, đặt câu hỏi về minh
bạch, bày tỏ bực tức chung hoặc yêu cầu xử lý đúng pháp luật không đủ để coi là
targeted abuse. Chỉ trả JSON đúng schema.
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_balanced_seed_20(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (
        len(records) == 20
        and all(
            record.get("ontology") == "targeted_abuse_hate_v1"
            for record in records
        )
    ):
        return sorted(records, key=lambda row: row["id"])
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["source_dataset"], record["binary_label"])].append(record)
    selected: list[dict[str, Any]] = []
    for group in [
        ("ViHSD", "safe"),
        ("ViHSD", "unsafe"),
        ("ViCTSD", "safe"),
        ("ViCTSD", "unsafe"),
    ]:
        candidates = sorted(
            grouped[group],
            key=lambda row: hashlib.sha256(
                ("smoke-20-v0|" + row["id"]).encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(candidates[:5])
    return sorted(selected, key=lambda row: row["id"])


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def expected_functions(seed: dict[str, Any]) -> list[str]:
    if seed.get("ontology") == "targeted_abuse_hate_v1":
        return (
            V1_UNSAFE_FUNCTIONS
            if seed["binary_label"] == "unsafe"
            else V1_SAFE_FUNCTIONS
        )
    return UNSAFE_FUNCTIONS if seed["binary_label"] == "unsafe" else SAFE_FUNCTIONS


def close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def call_structured(
    *,
    key_pool: ApiKeyPool,
    model: str,
    system_instruction: str,
    payload: dict[str, Any],
    response_model: type[BaseModel],
    temperature: float,
    event_path: Path,
    stage: str,
    batch_id: str,
) -> tuple[BaseModel, dict[str, Any] | None, int]:
    last_error: Exception | None = None
    attempts = max(key_pool.active_size, 1)
    for attempt in range(1, attempts + 1):
        slot, key = key_pool.next()
        append_jsonl(
            event_path,
            {
                "timestamp": now(),
                "event": "request_started",
                "stage": stage,
                "batch_id": batch_id,
                "attempt": attempt,
                "key_slot": slot,
                "model": model,
            },
            fsync=True,
        )
        client = genai.Client(api_key=key)
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_json_schema=response_model.model_json_schema(),
                    temperature=temperature,
                ),
            )
            parsed = response.parsed
            if isinstance(parsed, response_model):
                validated = parsed
            elif parsed is not None:
                validated = response_model.model_validate(parsed)
            else:
                validated = response_model.model_validate_json(response.text)
            usage_obj = getattr(response, "usage_metadata", None)
            usage = (
                usage_obj.model_dump(mode="json")
                if hasattr(usage_obj, "model_dump")
                else None
            )
            append_jsonl(
                event_path,
                {
                    "timestamp": now(),
                    "event": "request_success",
                    "stage": stage,
                    "batch_id": batch_id,
                    "key_slot": slot,
                    "model": model,
                    "usage_metadata": usage,
                },
                fsync=True,
            )
            return validated, usage, slot
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            event = "request_error"
            if "reported as leaked" in message or (
                "403" in message and "permission_denied" in message
            ):
                key_pool.disable(slot)
                event = "key_disabled"
            elif "429" in message or "resource_exhausted" in message:
                event = "quota_429"
            append_jsonl(
                event_path,
                {
                    "timestamp": now(),
                    "event": event,
                    "stage": stage,
                    "batch_id": batch_id,
                    "key_slot": slot,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1500],
                },
                fsync=True,
            )
            if event == "request_error":
                raise
            time.sleep(2)
        finally:
            close_client(client)
    raise last_error or RuntimeError(f"No usable key for {stage}/{batch_id}")


def validate_generation(
    batch_id: str,
    seeds: list[dict[str, Any]],
    response: GenerationResponse,
) -> list[dict[str, Any]]:
    if response.batch_id != batch_id:
        raise ValueError(f"batch_id mismatch: {response.batch_id!r} != {batch_id!r}")
    seed_by_id = {seed["id"]: seed for seed in seeds}
    observed: defaultdict[str, list[GeneratedVariant]] = defaultdict(list)
    for variant in response.variants:
        if variant.seed_id not in seed_by_id:
            raise ValueError(f"unknown seed_id: {variant.seed_id}")
        observed[variant.seed_id].append(variant)

    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        required = expected_functions(seed)
        actual = [variant.function for variant in observed[seed["id"]]]
        if Counter(actual) != Counter(required):
            raise ValueError(
                f"{seed['id']} functions mismatch: expected={required}, actual={actual}"
            )
        for variant in observed[seed["id"]]:
            text = re.sub(r"\s+", " ", variant.text).strip()
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            candidate_id = hashlib.sha256(
                f"{seed['id']}|{variant.function}|{text}".encode("utf-8")
            ).hexdigest()[:20]
            candidates.append(
                {
                    "candidate_id": f"vhb-{candidate_id}",
                    "seed_id": seed["id"],
                    "source_dataset": seed["source_dataset"],
                    "source_original_label": seed["source_label"],
                    "source_binary_label": seed["binary_label"],
                    "ontology": seed.get(
                        "ontology", "exploratory_hate_harassment_toxicity_v0"
                    ),
                    "source_text": seed["text"],
                    "function": variant.function,
                    "intended_label": FUNCTION_LABEL[variant.function],
                    "text": text,
                    "text_sha256": digest,
                    "generation_batch_id": batch_id,
                }
            )
    return candidates


def deterministic_checks(candidates: list[dict[str, Any]]) -> None:
    hashes = Counter(candidate["text_sha256"] for candidate in candidates)
    pii_patterns = {
        "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        "url": re.compile(r"https?://|www\.", re.I),
        "phone": re.compile(r"(?<!\d)(?:\+?84|0)\s*(?:\d[\s.-]*){8,10}(?!\d)"),
    }
    vietnamese_marks = re.compile(
        r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
        r"íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
        re.I,
    )
    for candidate in candidates:
        flags: list[str] = []
        text = candidate["text"]
        if len(text) < 12:
            flags.append("too_short")
        if len(text) > 1200:
            flags.append("too_long")
        if text.casefold() == candidate["source_text"].strip().casefold():
            flags.append("copied_source")
        if hashes[candidate["text_sha256"]] > 1:
            flags.append("duplicate_generated_text")
        for name, pattern in pii_patterns.items():
            if pattern.search(text):
                flags.append(f"possible_{name}")
        common_vi = re.findall(
            r"\b(?:là|và|của|người|không|hãy|nên|một|những|được|với|cho)\b",
            text.casefold(),
        )
        if not vietnamese_marks.search(text) and len(common_vi) < 2:
            flags.append("weak_vietnamese_signal")
        candidate["deterministic_flags"] = sorted(set(flags))


def judge_policy(
    candidates: list[dict[str, Any]],
    *,
    pool: ApiKeyPool,
    model: str,
    event_path: Path,
) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    for index, part in enumerate(chunks(candidates, 20), 1):
        batch_id = f"policy-{index:03d}"
        payload = {
            "batch_id": batch_id,
            "items": [
                {"candidate_id": row["candidate_id"], "text": row["text"]}
                for row in part
            ],
        }
        response, _, _ = call_structured(
            key_pool=pool,
            model=model,
            system_instruction=POLICY_JUDGE_SYSTEM,
            payload=payload,
            response_model=PolicyJudgeResponse,
            temperature=0.0,
            event_path=event_path,
            stage="policy_judge",
            batch_id=batch_id,
        )
        assert isinstance(response, PolicyJudgeResponse)
        if response.batch_id != batch_id:
            raise ValueError(f"policy batch mismatch: {response.batch_id}")
        by_id = {item.candidate_id: item for item in response.verdicts}
        expected = {row["candidate_id"] for row in part}
        if set(by_id) != expected:
            raise ValueError(
                f"policy ids mismatch: missing={expected-set(by_id)}, extra={set(by_id)-expected}"
            )
        verdicts.update(
            {
                key: value.model_dump(mode="json")
                for key, value in by_id.items()
            }
        )
    return verdicts


def judge_function(
    candidates: list[dict[str, Any]],
    *,
    pool: ApiKeyPool,
    model: str,
    event_path: Path,
) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    for index, part in enumerate(chunks(candidates, 15), 1):
        batch_id = f"function-{index:03d}"
        payload = {
            "batch_id": batch_id,
            "items": [
                {
                    "candidate_id": row["candidate_id"],
                    "source_text": row["source_text"],
                    "function": row["function"],
                    "candidate_text": row["text"],
                }
                for row in part
            ],
        }
        response, _, _ = call_structured(
            key_pool=pool,
            model=model,
            system_instruction=FUNCTION_JUDGE_SYSTEM,
            payload=payload,
            response_model=FunctionJudgeResponse,
            temperature=0.0,
            event_path=event_path,
            stage="function_judge",
            batch_id=batch_id,
        )
        assert isinstance(response, FunctionJudgeResponse)
        if response.batch_id != batch_id:
            raise ValueError(f"function batch mismatch: {response.batch_id}")
        by_id = {item.candidate_id: item for item in response.verdicts}
        expected = {row["candidate_id"] for row in part}
        if set(by_id) != expected:
            raise ValueError(
                f"function ids mismatch: missing={expected-set(by_id)}, extra={set(by_id)-expected}"
            )
        verdicts.update(
            {
                key: value.model_dump(mode="json")
                for key, value in by_id.items()
            }
        )
    return verdicts


def assign_status(candidates: list[dict[str, Any]]) -> None:
    for row in candidates:
        policy = row["policy_judge"]
        function = row["function_judge"]
        reasons: list[str] = []
        if row["deterministic_flags"]:
            reasons.extend(row["deterministic_flags"])
        if policy["predicted_label"] != row["intended_label"]:
            reasons.append("policy_label_disagreement")
        if policy["confidence"] == "low":
            reasons.append("policy_low_confidence")
        if not function["function_valid"]:
            reasons.append("function_invalid")
        if not function["source_relation_valid"]:
            reasons.append("source_relation_invalid")
        if function["naturalness"] < 3:
            reasons.append("naturalness_below_3")
        if function["pii_risk"]:
            reasons.append("judge_pii_risk")
        reasons.extend(function["issue_codes"])
        row["review_reasons"] = sorted(set(reason for reason in reasons if reason))
        row["review_status"] = "accepted_silver" if not row["review_reasons"] else "needs_review"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "API.txt")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--generation-key-slots", default="1,2,3,4,5")
    parser.add_argument("--policy-key-slots", default="6,7,8,9,10")
    parser.add_argument("--function-key-slots", default="11,12,13,14,15")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / "events.jsonl"
    candidate_path = output_dir / "candidates.jsonl"
    final_path = output_dir / "judged_candidates.jsonl"

    seeds = stable_balanced_seed_20(read_jsonl(args.seed_file.resolve()))
    (output_dir / "selected_seeds.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeds),
        encoding="utf-8",
    )

    generation_pool = ApiKeyPool(
        args.api_key_file,
        [int(value) for value in args.generation_key_slots.split(",") if value],
    )
    policy_pool = ApiKeyPool(
        args.api_key_file,
        [int(value) for value in args.policy_key_slots.split(",") if value],
    )
    function_pool = ApiKeyPool(
        args.api_key_file,
        [int(value) for value in args.function_key_slots.split(",") if value],
    )

    existing_candidates = read_jsonl(candidate_path)
    completed_seeds = {row["seed_id"] for row in existing_candidates}
    candidates = list(existing_candidates)

    pending = [seed for seed in seeds if seed["id"] not in completed_seeds]
    for index, part in enumerate(chunks(pending, 5), 1):
        batch_id = f"generate-{index:03d}-{int(time.time())}"
        payload = {
            "batch_id": batch_id,
            "seeds": [
                {
                    "seed_id": seed["id"],
                    "source_label": seed["binary_label"],
                    "source_text": seed["text"],
                    "required_functions": expected_functions(seed),
                }
                for seed in part
            ],
        }
        response, usage, key_slot = call_structured(
            key_pool=generation_pool,
            model=args.model,
            system_instruction=GENERATION_SYSTEM,
            payload=payload,
            response_model=GenerationResponse,
            temperature=0.65,
            event_path=event_path,
            stage="generation",
            batch_id=batch_id,
        )
        assert isinstance(response, GenerationResponse)
        new_candidates = validate_generation(batch_id, part, response)
        for row in new_candidates:
            row["generator_model"] = args.model
            row["generator_key_slot"] = key_slot
            row["generation_usage"] = usage
            append_jsonl(candidate_path, row, fsync=True)
        candidates.extend(new_candidates)

    # Recompute deterministic flags for a resumed run as well.
    deterministic_checks(candidates)
    policy = judge_policy(
        candidates,
        pool=policy_pool,
        model=args.model,
        event_path=event_path,
    )
    function = judge_function(
        candidates,
        pool=function_pool,
        model=args.model,
        event_path=event_path,
    )
    for row in candidates:
        row["policy_judge"] = policy[row["candidate_id"]]
        row["function_judge"] = function[row["candidate_id"]]
    assign_status(candidates)

    final_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates),
        encoding="utf-8",
    )
    summary = {
        "timestamp": now(),
        "model": args.model,
        "seeds": len(seeds),
        "candidates": len(candidates),
        "by_intended_label": dict(Counter(row["intended_label"] for row in candidates)),
        "by_function": dict(Counter(row["function"] for row in candidates)),
        "by_review_status": dict(Counter(row["review_status"] for row in candidates)),
        "review_reasons": dict(
            Counter(
                reason
                for row in candidates
                for reason in row["review_reasons"]
            ).most_common()
        ),
        "policy_confusion": dict(
            Counter(
                f"{row['intended_label']}->{row['policy_judge']['predicted_label']}"
                for row in candidates
            )
        ),
        "naturalness": dict(
            Counter(
                str(row["function_judge"]["naturalness"]) for row in candidates
            )
        ),
        "artifacts": {
            "selected_seeds": str(output_dir / "selected_seeds.jsonl"),
            "candidates": str(candidate_path),
            "judged_candidates": str(final_path),
            "events": str(event_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
