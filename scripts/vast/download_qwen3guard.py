from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen3Guard-Gen-4B"
PINNED_REVISION = "6ec42827da0c1ff11e7a49dc269d2e810d27e108"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and hash the pinned Qwen3Guard checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=PINNED_REVISION)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=args.revision,
        local_dir=args.output,
    )
    records = []
    for path in sorted(args.output.rglob("*")):
        if (
            not path.is_file()
            or ".cache" in path.parts
            or path.name in {"download_manifest.json", "download_manifest.json.tmp"}
        ):
            continue
        records.append(
            {
                "path": path.relative_to(args.output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "model_id": MODEL_ID,
        "revision": args.revision,
        "output": str(args.output),
        "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "records": records,
    }
    temporary = args.output / "download_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "download_manifest.json")
    print(json.dumps({key: manifest[key] for key in ("model_id", "revision", "files", "bytes")}, indent=2))


if __name__ == "__main__":
    main()
