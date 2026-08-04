from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official NVIDIA decoder guard baseline.")
    parser.add_argument(
        "--repo-id",
        default="nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/llama_3_1_nemotron_safety_guard_8b_v3"),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.output,
        max_workers=args.max_workers,
        allow_patterns=(
            "*.json",
            "*.safetensors",
            "*.model",
            "README.md",
            "LICENSE*",
        ),
    )
    print(f"download_complete={path}", flush=True)


if __name__ == "__main__":
    main()

