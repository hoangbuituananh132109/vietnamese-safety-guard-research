from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/workspace/safety-dataset")
VENV = Path("/workspace/venvs/nemotron-vllm")
STATE = ROOT / "reports" / "decoder_baseline" / "vllm_install_state.json"
VLLM_VERSION = "0.17.0"
CUDA_VARIANT = "cu128"
VLLM_SPEC = f"vllm=={VLLM_VERSION}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(**values: object) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, object] = {}
    try:
        current = json.loads(STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    current.update(values, updated_at=utc_now())
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(STATE)


def run(command: list[str], stage: str) -> None:
    write_state(status="running", stage=stage, command=command)
    subprocess.run(command, check=True)


def main() -> None:
    started = time.monotonic()
    write_state(
        status="starting",
        started_at=utc_now(),
        venv=str(VENV),
        requested_vllm_version=VLLM_VERSION,
        cuda_variant=CUDA_VARIANT,
        package_spec=VLLM_SPEC,
    )
    python = VENV / "bin" / "python"
    if not python.exists():
        VENV.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "venv", str(VENV)], "create_venv")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "uv"], "install_uv")
    uv = VENV / "bin" / "uv"
    run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python),
            VLLM_SPEC,
            "scikit-learn>=1.5,<2",
            f"--torch-backend={CUDA_VARIANT}",
        ],
        "install_vllm",
    )
    verification = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import torch,vllm,sklearn; "
                "from guard_train.binary_metrics import build_report; "
                "from guard_train.n23_metrics import build_n23_report; "
                "print(vllm.__version__); print(torch.__version__); "
                "print(torch.version.cuda); print(sklearn.__version__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    write_state(
        status="completed",
        stage=None,
        completed_at=utc_now(),
        elapsed_seconds=time.monotonic() - started,
        verification=verification.stdout.strip().splitlines(),
    )
    print(verification.stdout, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_state(status="failed", failed_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        raise
