# Luna overnight runbook

## Active configuration (2026-08-01)

- Input: `data/final/nemotron_train_en_vi_v10_final.jsonl` (read only).
- Output: `data/luna_overnight/nemotron_train_20260801/`.
- Ten parallel workers; every batch uses a new ephemeral Codex session.
- GPT-5.6 Luna, low reasoning.
- Luna schedule confirmed by the user: 5 credits per 1M uncached input tokens, 0.5 credits per 1M cached
  input tokens and 30 credits per 1M output tokens. The initial invocation's 301.070923-credit estimate was
  therefore correct.
- A short resume was started with an incorrect 0.5 output rate after a pricing misunderstanding and was stopped
  immediately after ten batches. Its telemetry remains append-only. Future resume limit: three hours or
  approximately 20 credits under the restored meter, whichever comes first.
- Per-turn timeout: ten minutes.
- Infrastructure circuit breaker: three consecutive failed batches.

## Routing

- Plain normal prose: P3 sandwich prompt, 30 records / 50k source characters.
- Near-tail prose: 20 records / 45k characters.
- Tail prose: 3 records / 12k characters.
- High-tail and oversized: one record per turn.
- Embedded JSON: P6 byte-preserved-key prompt; at most 10 normal, 2 tail, 1 high-tail.
- Leetspeak: P5 candidate prompt; at most 10 normal, 2 tail, 1 high-tail; every valid candidate
  still goes to `needs_audit.jsonl` instead of automatic pass.

## Durable queues

- `passed.jsonl`: hard-valid, warning-free non-leet records.
- `needs_audit.jsonl`: leet candidates and heuristic warnings.
- `exhausted_normal.jsonl`: normal/near-tail records still invalid after three total attempts.
- `tail_escalation.jsonl`: tail or larger records still invalid after two total attempts.
- `attempts.jsonl`: every per-record attempt and candidate, including failed candidates.
- `raw_batches.jsonl`: append-only batch telemetry and raw finals.
- `manifest.json`: exact model, prompt hashes, input hash, routing and retry policy.
- `summary.json`: written when the supervisor stops normally or at a configured budget.

Successful items are stored before failures are requeued. A failed subset is appended behind the existing queue,
so it does not monopolize a worker. Restart the same command to resume; terminal UIDs are skipped and attempt
counts continue from `attempts.jsonl`.

## Status commands

```powershell
Get-Content .\data\luna_overnight\nemotron_train_20260801\runner.stdout.log -Tail 20
Get-Content .\data\luna_overnight\nemotron_train_20260801\runner.stderr.log -Tail 20
Get-Content .\data\luna_overnight\nemotron_train_20260801\summary.json
```

If `summary.json` is absent, the run is either active or ended before the supervisor could finalize. Inspect the
two log files and `raw_batches.jsonl`; all completed item records remain recoverable.
