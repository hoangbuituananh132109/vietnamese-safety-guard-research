# Luna/Sol corpus is complete and ready for matched training

Status date: 2026-08-09

## Technical summary

The Luna/Sol Nemotron EN→VI corpus is complete at 45,416/45,416 source UIDs. It contains no Gemini fallback record, no missing/extra/duplicate UID, no structural issue, and no split overlap. The project's production materializer reads the new corpus and the Gemini control into identical train, validation, and test example counts.

The data-quality and materialization gates are passed. GPU training for the Luna/Sol-versus-Gemini comparison has not yet started, so the corpus is train-ready but not experimentally proven superior.

## Full coverage replaced the earlier hybrid result

| Split | Expected UIDs | Luna/Sol UIDs | Materialized examples | Ready |
|---|---:|---:|---:|---|
| Train | 40,007 | 40,007 | 140,136 | Yes |
| Validation | 2,445 | 2,445 | 8,764 | Yes |
| Test | 2,964 | 2,964 | 10,682 | Yes |
| **Total** | **45,416** | **45,416** | **159,582** | **Yes** |

The 2026-08-03 readiness report described a 45,391-record pure subset plus 25 Gemini fallbacks. Those 25 records have now been translated with Sol Web, validated, and merged. The earlier hybrid recommendation is therefore superseded by `data/final_luna_sol_pure_v1`.

## Provider and quality composition

| Provider/stage | Records |
|---|---:|
| Luna initial | 43,014 |
| Luna remaining queue | 2,033 |
| Luna recovery | 7 |
| Luna continuation | 6 |
| Sol Web hard fallback | 331 |
| Sol Web final timeout fallback | 25 |
| **Total** | **45,416** |

Luna contributes 45,060 records (99.216%) and Sol contributes 356 (0.784%). This should be named the “Luna/Sol routing condition,” not a strictly Luna-only corpus.

| Quality disposition | Records |
|---|---:|
| Hard pass | 45,004 |
| Warning pass | 372 |
| Manual-audit pass | 31 |
| Revalidated pass | 7 |
| Accepted known issue | 2 |
| **Total** | **45,416** |

The final 25 Sol records produced 17 automatic hard passes and eight manual-audit passes. All eight warnings came from unchanged or mostly unchanged ASCII-art response bodies; the surrounding natural-language text was translated. No final record had a hard validation failure.

## Scope and definitions

- Grain: one source `record_uid` per JSONL row before materialization.
- Completeness: every source UID has exactly one selected Vietnamese candidate in its original split.
- Structural readiness: UID/schema/null/empty/required-field invariants pass.
- Materialization parity: the guard training loader produces the same view, language, label, and tag counts for Luna/Sol and Gemini.
- Translation quality: structural and heuristic validation plus targeted manual audit; not full human-reference equivalence.

## Methodology and verification

The final 25 browser-result files were aggregated by `tools/validate_sol_missing_25_results.py`. The script rebuilt expected UID membership from the saved batch JSONL files, checked missing/extra/duplicate records, and called the same `validate_item` function used by the main Luna/Sol workflow.

Warning-only ASCII records were finalized through `tools/finalize_sol_missing_25_validation.py`. `tools/build_luna_sol_training_dataset.py` then applied deterministic candidate priority and wrote the complete corpus. `tools/audit_luna_sol_training_readiness.py` passed the outputs through `guard_smoke.data.iter_full_examples`.

Automated verification completed in this session:

- Luna runner tests: 7 passed;
- Python syntax compilation: passed for the new/modified pipeline scripts;
- full UID coverage: passed;
- source UID uniqueness: passed;
- structural issue count: zero;
- train/validation/test intersections: zero;
- pure Luna/Sol versus Gemini materialization parity: true for every split.

## Limitations and robustness requirements

- The validator cannot establish human-level semantic or stylistic equivalence for all records.
- Two previously accepted Sol records retain known minor issues and must remain visible in provenance.
- Results may depend on Sol-routed hard examples; report Luna-only and Sol slices separately.
- Multiple historical candidate versions exist for 1,758 UIDs because of retries and resumes, although only one final candidate is selected.
- Temporary queue sequence metadata was normalized for 2,039 UIDs using the source UID/split order.
- No downstream checkpoint has yet been trained on the new corpus.
- SEA evaluation and the translated-test 2×2 matrix remain pending.

## Recommended next steps

1. Freeze checksums for the full Luna/Sol and Gemini conditions.
2. Build manifests with the existing production manifest builder.
3. Run matched Gemini and Luna/Sol training with an identical configuration and at least three seeds if feasible.
4. Evaluate both models on English Nemotron, Gemini-translated Nemotron, Luna/Sol-translated Nemotron, and SEA paired EN–VI.
5. Report paired uncertainty, provider/style slices, warning slices, length, leetspeak, safe/unsafe, and N23 categories.
6. Publish the pipeline code and compact summaries to GitHub; publish datasets separately after license and safety review.

## Further questions

- Does Luna/Sol improve downstream performance, or only change translation style?
- Does either trained model perform best on its own provider's translated test?
- Are gains concentrated in long, leetspeak, unsafe, or specific N23 categories?
- Does removing all Sol-routed records change the headline conclusion?
- What are normalized cost, latency, and failure rates per million source/output tokens?
- How much blinded human preference agrees with validator dispositions?

Native Vietnamese data generation is outside this report and has no new result in the current phase.
