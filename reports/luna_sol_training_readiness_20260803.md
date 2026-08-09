# Luna/Sol translation dataset — training-readiness report

Date: 2026-08-03
Scope: Nemotron train/valid/test Vietnamese translations produced by Luna and Sol Web, compared with the existing Gemini translation corpus.

## Executive conclusion

The data is ready to begin training.

- For the next full training run, use `data/final_luna_sol_hybrid_v1`. It contains all 45,416 source records and preserves the 25 original Gemini translations only where Luna produced no candidate because of infrastructure failures. In the training split this affects 18 of 40,007 records (0.0450%). Every fallback row is explicitly labeled; there is no silent provider mixing.
- For a controlled Luna/Sol-versus-Gemini research comparison, use the two paired directories `data/final_luna_sol_paired_v1` and `data/final_gemini_paired_v1`. They contain exactly the same UIDs and therefore support an apples-to-apples comparison without the 25 infrastructure failures.
- The two previously identified Sol Web imperfections are accepted as known issues, retained in the dataset, and explicitly flagged. They represent 2 of 45,416 source records (0.0044%) and are not a training blocker.
- The pure Luna/Sol corpus is not literally 100% complete: it covers 45,391 of 45,416 records (99.9450%). The missing 25 are infrastructure/time-out cases, not records that failed the translation quality validator.

## Coverage

| Split | Source records | Luna/Sol records | Missing | Hybrid records |
|---|---:|---:|---:|---:|
| Train | 40,007 | 39,989 | 18 | 40,007 |
| Valid | 2,445 | 2,442 | 3 | 2,445 |
| Test | 2,964 | 2,960 | 4 | 2,964 |
| **Total** | **45,416** | **45,391** | **25** | **45,416** |

Pure Luna/Sol coverage is 99.9450%; the hybrid corpus is complete.

The 25 records without a Luna/Sol candidate comprise:

- train: 15 `high_tail`, 3 `oversized`;
- valid: 3 `high_tail`;
- test: 4 `high_tail`.

They contain 348,225 source characters in total; the largest source record contains 65,894 characters. Their absence is consistent with infrastructure/time-limit pressure on long inputs. They are listed in `data/luna_sol_dataset_v1/missing_luna_sol_candidates.jsonl` for later targeted Luna Medium/High or Sol processing.

## Selected translation provenance

| Provider/stage | Selected records |
|---|---:|
| Luna initial runs | 43,014 |
| Luna remaining queue | 2,033 |
| Sol Web fallback | 331 |
| Luna validator recovery | 7 |
| Luna continuation | 6 |
| **Pure Luna/Sol total** | **45,391** |

Final quality dispositions in the pure corpus are:

| Disposition | Records |
|---|---:|
| Hard pass | 44,987 |
| Warning pass | 372 |
| Manual-audit pass | 23 |
| Revalidated pass | 7 |
| Accepted known issue | 2 |
| **Total** | **45,391** |

The two accepted known issues are:

1. `en-train-00020107-7cd91d92b3ef`: natural-language jailbreak prose inside a fenced block remained in English.
2. `en-train-00012179-fb3f7c715e03`: one URL changed a hyphen to an underscore.

Both remain discoverable through provenance/quality fields and can be replaced later without changing the experimental split.

## Integrity and trainer compatibility

The audit used the project's actual `guard_smoke.data.iter_full_examples` materializer, rather than a standalone row counter.

- Source UIDs: 45,416 unique; 0 duplicates.
- Selected corpus: 0 unexpected UIDs and 0 structural issues.
- Train/valid/test UID intersections: all zero.
- Hybrid manifests have exact parity with the original Gemini corpus for every split.
- Paired Luna/Sol and paired Gemini manifests also have exact parity for every split.
- The canonical filenames expected by `scripts/build_guard_full_manifest.py` are present.

Materialized example counts:

| Corpus | Train | Valid | Test |
|---|---:|---:|---:|
| Full hybrid | 140,136 | 8,764 | 10,682 |
| Paired comparison | 140,038 | 8,748 | 10,662 |

There were 2,039 temporary queue sequence values in resumed results. These were normalized back to the original source order using UID as the primary key. There are no resulting structural errors. A total of 1,758 UIDs had more than one candidate version because of retries/resumes; deterministic source precedence selected exactly one final version per UID, so these are provenance histories rather than duplicate training rows.

## Recommended experimental use

### Full model training

Use `data/final_luna_sol_hybrid_v1` for the immediate full run. The 18 Gemini-fallback training records are only 0.0450% of the training split. Excluding them would make the full run slightly less comparable to prior training and would provide no meaningful purity benefit.

### Translation-pipeline contribution / ablation

For the defensible comparison, train the same model configuration twice:

1. Luna/Sol paired corpus: `data/final_luna_sol_paired_v1`.
2. Gemini paired corpus: `data/final_gemini_paired_v1`.

Keep seed, data views, optimizer, number of updates, checkpoint selection, and evaluation code fixed. Evaluate both models on:

- the identical paired Nemotron validation and test UIDs;
- SEA Bench as an external test set;
- optionally, per-stratum slices such as length, leetspeak/obfuscation, safe/unsafe label, and harm category.

This design attributes the measured difference to the translated text much more cleanly than comparing corpora with different record membership.

## Artifacts

- Machine-readable readiness summary: `data/luna_sol_dataset_v1/readiness_summary.json`
- Materialization audit: `data/luna_sol_dataset_v1/materialization_audit.json`
- Missing 25 records: `data/luna_sol_dataset_v1/missing_luna_sol_candidates.jsonl`
- Train-ready full hybrid: `data/final_luna_sol_hybrid_v1`
- Paired Luna/Sol: `data/final_luna_sol_paired_v1`
- Paired Gemini control: `data/final_gemini_paired_v1`

No GPU training was launched as part of this audit.
