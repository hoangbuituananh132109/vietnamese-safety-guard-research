# Sol web vs Luna hard-ablation comparison

The five Sol files in `data/luna_hard_ablation_100/sol_results/` were merged by `record_uid` and checked against the fixed 100-record manifest.

## Structural result

| Provider | Records returned | Unique UIDs | Missing from 100 | Notes |
|---|---:|---:|---:|---|
| Sol web | 100 | 100 | 0 | All five validated files cover the full set |
| Luna Medium | 27 | 27 | 73 | Run stopped by infrastructure circuit breaker |
| Luna High | 32 | 32 | 68 | Run stopped by infrastructure circuit breaker |
| Luna XHigh | 30 | 30 | 70 | Run stopped by infrastructure circuit breaker |

The Luna counts are not quality scores. They are only the records that reached the runner's `passed.jsonl` before the run stopped. The remaining records were mostly long/high-tail items whose batch response was missing one or more required items.

## Comparable automated proxies

On the records returned by each provider, the average translated `prompt_vi` character count divided by source `prompt_en` character count was approximately:

- Sol: 1.053
- Luna Medium: 1.054
- Luna High: 1.059
- Luna XHigh: 1.042

No provider had a source/target length ratio below 0.5 in this slice, so there is no simple evidence of systematic truncation from this length proxy. This does not replace bilingual quality review.

The ASCII-token overlap proxy was 0.310 for Sol and 0.364–0.367 for Luna. This is only a diagnostic signal: named entities, identifiers, URLs, code, leetspeak, and untranslated proper nouns can legitimately increase overlap. It must not be interpreted as a translation score.

## Interpretation

The defensible conclusion from this run is operational rather than semantic: Sol completed all 100 web batches with valid UID coverage, while the three Luna settings did not complete the same 100-record workload under the large-batch configuration. A fair quality comparison still requires rerunning the 68–73 Luna pending UIDs in small batches (one high-tail record or 2–3 normal records per request), then scoring the same UID set with bilingual review or a calibrated evaluator.

Generated machine-readable metrics:

- `data/luna_hard_ablation_100/comparison_sol_vs_luna.json`
- `data/luna_hard_ablation_100/comparison_metrics.json`
