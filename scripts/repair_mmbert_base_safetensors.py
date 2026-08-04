"""Convert a tied-weight mmBERT MLM checkpoint to a base encoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("models/mmbert_small_smoke"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/mmbert_small_base_smoke")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    model = AutoModel.from_config(config)
    safetensors_path = args.source / "model.safetensors"
    pytorch_path = args.source / "pytorch_model.bin"
    if safetensors_path.exists():
        source_state = load_file(str(safetensors_path), device="cpu")
        source_format = "safetensors"
    elif pytorch_path.exists():
        # The current public jhu-clsp/mmBERT-small snapshot is distributed as a
        # PyTorch state dict, while older cached revisions used safetensors.
        # weights_only=True avoids unpickling arbitrary Python objects.
        source_state = torch.load(
            pytorch_path, map_location="cpu", weights_only=True
        )
        source_format = "pytorch_state_dict"
    else:
        raise FileNotFoundError(
            f"Neither {safetensors_path} nor {pytorch_path} exists"
        )
    if not isinstance(source_state, dict):
        raise TypeError(
            f"Unexpected mmBERT checkpoint payload: {type(source_state).__name__}"
        )
    base_state = {
        key.removeprefix("model."): value
        for key, value in source_state.items()
        if key.startswith("model.")
    }
    tied_source = source_state.get("decoder.weight")
    if tied_source is None:
        raise RuntimeError("Expected tied decoder.weight in old mmBERT MLM checkpoint")
    base_state["embeddings.tok_embeddings.weight"] = tied_source
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Base checkpoint conversion mismatch: missing={missing}, unexpected={unexpected}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    tokenizer.save_pretrained(args.output)
    report = {
        "source": str(args.source),
        "source_format": source_format,
        "output": str(args.output),
        "source_tensor_count": len(source_state),
        "base_tensor_count": len(base_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "embedding_shape": list(tied_source.shape),
    }
    (args.output / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
