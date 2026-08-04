from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Package exact manifest and benchmark files named in a phase inventory.")
    parser.add_argument("--root", type=Path, default=Path("/workspace/safety-dataset"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    inventory_path = args.inventory if args.inventory.is_absolute() else root / args.inventory
    archive_path = args.archive if args.archive.is_absolute() else root / args.archive
    ready_path = args.ready if args.ready.is_absolute() else root / args.ready
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    paths: list[str] = []
    for entry in inventory.get("manifests", []):
        relative = str(entry["path"])
        candidate = (root / relative).resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"Manifest escapes root: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        paths.append(relative)
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError("Inventory contains no referenced manifests")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    subprocess.run(["tar", "--zstd", "-cf", str(temporary), *paths], cwd=root, check=True)
    os.replace(temporary, archive_path)
    ready = {
        "status": "ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inventory": str(inventory_path),
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "files": paths,
    }
    ready_path.write_text(json.dumps(ready, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ready, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
