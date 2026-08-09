# Luna hard-case routing and reasoning ablation

The remaining records were classified offline after the 2026-08-02 quota stop. No model call is needed for
this routing report.

## Remaining records

| Split | Pending | Main composition |
|---|---:|---|
| Train | 2,059 | 1,007 tail prose, 395 high-tail, 540 normal/near-tail, 5 oversized |
| Valid | 4 | 3 high-tail prose + 1 high-tail leetspeak |
| Test | 823 | 402 normal prose, 175 near-tail prose, 171 tail prose, 24 high-tail prose, 50 structured/leet |

The train remainder is enriched for jailbreak/unsafe content: 1,258/2,059 are tagged jailbreaking and
1,392/2,059 have an unsafe prompt label. The most frequent overlapping categories are Criminal
Planning/Confessions (460), Profanity (349), Illegal Activity (303), Immoral/Unethical (297), Violence
(237), Controlled/Regulated Substances (234), Hate/Identity Hate (222), and Suicide/Self-Harm (190).

The test remainder is even more adversarial: 713/823 are tagged jailbreaking and 614/823 are unsafe. Its
largest categories are Criminal Planning/Confessions (201), Immoral/Unethical (166), Illegal Activity (165),
Controlled/Regulated Substances (140), Manipulation (100), Suicide/Self-Harm (87), Violence (86), and
PII/Privacy (85).

## Routing policy after quota returns

1. Keep all current Luna pass records as provisional. Do not train the final Luna model until the selected
   train/valid/test UID sets are frozen and the remaining hard cases have a documented disposition.
2. For records <= 3,772 source characters with no JSON or leetspeak signal, keep Luna low in batches of 30.
3. For tail prose (2,343–3,772 chars), test Luna `medium` on a small matched slice; use batch size 2–3.
4. For high-tail prose (3,773–25,000 chars), test one record per turn at `high`; if it times out or is
   structurally incomplete, send it to Sol/Terra or web review. Do not repeat a high-tail timeout at the same
   effort.
5. For oversized records (>25,000 chars), never put multiple records in one request. Prefer Sol or Terra
   with chunk-aware document translation; Luna `xhigh` is an ablation only, not the default fallback.
6. For JSON-shaped records, use the key-preservation prompt. If keys still change after one repair, escalate.
7. For leetspeak, use a dedicated route and keep every output in audit until a human or bilingual evaluator
   confirms that the decoded meaning and Vietnamese obfuscation are both correct.

## Reasoning ablation

Use a frozen 24-record development set: 6 tail prose, 6 high-tail prose, 6 JSON, and 6 leetspeak. For each
shape compare `low`, `medium`, `high`, and the CLI's highest supported setting (currently verify whether the
installed build names it `xhigh` or `extra_high` before launching). Record completion, hard errors,
confirmed semantic errors, source/output tokens, latency and credit. Select the lowest effort that matches the
best quality within a predeclared tolerance; do not select by raw validator pass rate alone.

## Final training/evaluation gate

Freeze two matched translations per UID (Gemini and Luna), exclude unresolved audit/escalation records from the
primary claim, and train only after all three split manifests exist. Run the 2x2 cross-test matrix plus SEA
Bench/SEA-HELM. Report the escalated tail subset separately rather than silently mixing Sol/Terra outputs into
the Luna condition.
