from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/workspace/safety-dataset")
EXPECTED_DIRS = (
    "configs",
    "guard_smoke",
    "guard_train",
    "scripts",
    "tests",
    "notebooks",
    "data/guard_full",
    "data/guard_phase0_gliguard_native_512",
    "data/guard_experiments_v2",
    "data/eval_shared_gliguard_512",
    "data/benchmarks/sea_safeguard",
    "models/fastino_gliguard_300m",
    "models/mmbert_small_base_smoke",
)
MODEL_FILES = (
    "models/fastino_gliguard_300m/model.safetensors",
    "models/mmbert_small_base_smoke/model.safetensors",
)
FORBIDDEN_NAMES = {"api.txt", "id_ed25519", "id_ed25519.pub", "id_rsa", "id_rsa.pub"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    directories: dict[str, dict[str, int | bool]] = {}
    for relative in EXPECTED_DIRS:
        path = ROOT / relative
        files = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else []
        directories[relative] = {
            "exists": path.is_dir(),
            "files": len(files),
            "bytes": sum(item.stat().st_size for item in files),
        }

    sensitive_matches: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        lowered = relative.lower()
        if (
            path.name.lower() in FORBIDDEN_NAMES
            or "runner_state" in lowered
            or lowered.startswith("reports/full_run/")
        ):
            sensitive_matches.append(relative)

    payload = {
        "root": str(ROOT),
        "directories": directories,
        "sensitive_filename_matches": sorted(sensitive_matches),
        "model_sha256": {
            relative: sha256(ROOT / relative) if (ROOT / relative).is_file() else None
            for relative in MODEL_FILES
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
