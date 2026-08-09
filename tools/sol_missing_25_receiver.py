from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/sol_missing_25/sol_results"
MAX_BODY = 100 * 1024 * 1024
BATCH_RE = re.compile(r"^nemotron-sol-missing-25-\d{3}$")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status: int, payload: Any) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._reply(204, {})

    def do_GET(self) -> None:  # noqa: N802
        self._reply(200, {"ok": True, "output": str(OUT)}) if self.path == "/health" else self._reply(404, {"ok": False})

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY:
            raise ValueError("invalid request size")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        return value

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._payload()
            if self.path == "/save-batch":
                batch_id = str(payload.get("batch_id") or "")
                if not BATCH_RE.fullmatch(batch_id):
                    raise ValueError("invalid batch_id")
                destination = OUT / f"{batch_id}-browser-result.json"
                atomic_json(destination, payload)
                self._reply(200, {"ok": True, "saved": str(destination), "items": len(payload.get("items") or [])})
                return
            if self.path == "/save-results":
                destination = OUT / "sol-missing-25-all-results.json"
                atomic_json(destination, payload)
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for item in payload.get("items") or []:
                    if isinstance(item, dict) and BATCH_RE.fullmatch(str(item.get("_sol_batch_id") or "")):
                        grouped[str(item["_sol_batch_id"])].append(item)
                for batch_id, items in grouped.items():
                    atomic_json(OUT / f"{batch_id}-browser-result.json", {"batch_id": batch_id, "items": items, "source": "aggregate-browser-export"})
                self._reply(200, {"ok": True, "saved": str(destination), "batches_saved": len(grouped), "items_saved": sum(map(len, grouped.values()))})
                return
            self._reply(404, {"ok": False, "error": "not found"})
        except (ValueError, json.JSONDecodeError) as error:
            self._reply(400, {"ok": False, "error": str(error)})
        except Exception as error:
            self._reply(500, {"ok": False, "error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", "output": str(OUT)}, ensure_ascii=False), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
