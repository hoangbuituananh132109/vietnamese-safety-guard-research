---
base_model: Qwen/Qwen3Guard-Gen-4B
library_name: peft
license: apache-2.0
tags: [lora, safety, vietnamese, content-moderation]
---

# Q2 Qwen3Guard EN-VI P/PR LoRA

Adapter-only release candidate for the project's final Q2 decoder run. Users
must obtain `Qwen/Qwen3Guard-Gen-4B` separately.

## Training contract

- Final clean contract: P and PR only; response-only R excluded.
- 101,274 English/Vietnamese rows.
- One epoch; context 2,048; BF16; no quantization.
- LoRA rank 8, alpha 32, dropout 0.05; targets `q_proj`, `v_proj`.
- Effective batch 32.
- Training data: Gemini-derived Vietnamese condition, not the new Luna/Sol
  condition.

## Matched local result

On the immutable 11,736-example P/PR decoder intersection, Q2 reached 87.83%
accuracy and 87.82% macro-F1, with 87.79% safe recall and 87.87% unsafe recall.
These are local subset results, not official Qwen benchmark numbers.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3Guard-Gen-4B"
adapter_id = "REPLACE_WITH_HUB_REPO"
tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto")
model = PeftModel.from_pretrained(base, adapter_id)
```

Before upload, copy only `adapter_model.safetensors` and
`adapter_config.json`, replace the private `/workspace/...` base path in the
config with the public base ID, and add SHA-256 checksums. Do not upload copied
base weights or unchanged tokenizer files.
