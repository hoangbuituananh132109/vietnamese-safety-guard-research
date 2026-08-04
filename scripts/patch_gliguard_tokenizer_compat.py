"""Make a Transformers-5 tokenizer_config readable by Transformers 4.57.

GLiGuard was saved with Transformers 5.6, where ``extra_special_tokens`` may be
a list.  Transformers 4.57 expects a role-to-token mapping.  This script is an
idempotent local-runtime compatibility patch; it does not change token IDs or
model weights and is unnecessary when the rental GPU uses Transformers 5.6+.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROLE_ORDER = (
    "sep_struct_token",
    "sep_text_token",
    "prompt_task_token",
    "classification_token",
    "entity_token",
    "relation_token",
    "label_token",
    "example_token",
    "output_token",
    "description_token",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tokenizer_config",
        type=Path,
        nargs="?",
        default=Path("models/fastino_gliguard_300m/tokenizer_config.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.tokenizer_config.read_text(encoding="utf-8"))
    special = payload.get("extra_special_tokens")
    if isinstance(special, dict):
        print("already_compatible=true")
        return
    if not isinstance(special, list) or len(special) != len(ROLE_ORDER):
        raise ValueError(
            "Unexpected extra_special_tokens; refusing to guess token roles: "
            f"{special!r}"
        )
    payload["extra_special_tokens"] = dict(zip(ROLE_ORDER, special))
    args.tokenizer_config.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("patched=true")


if __name__ == "__main__":
    main()
