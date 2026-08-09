# Luna reasoning ablation: validator, leetspeak, and timeout analysis

## Technical summary

Higher reasoning produced only a small increase in validator-valid candidates on the 77 UIDs that reached a terminal outcome in Light, Medium, and High: **59/77 (76.6%) for Light, 60/77 (77.9%) for Medium, and 62/77 (80.5%) for High**. This is a three-record gain from Light to High, not a large or monotonic quality jump.

The apparent pass-only increase (`37 -> 40 -> 42`) overstates the evidence because leetspeak candidates are forced into `needs_audit` even when structurally valid, and 19–20 records per mode never received a terminal candidate due to timeout. On leetspeak, Light, Medium, and High all produced the same outcome: **18/20 validator-valid candidates and 2/20 exhausted**. Higher reasoning did not improve the leetspeak pass rate.

Timeout is a pipeline setting, not a Luna service declaration. The runner used a fixed wall-clock timeout of **900 seconds per Codex process**. The fixed retry run timed out on 35/53 Light requests, 37/49 Medium requests, and 40/49 High requests. Higher reasoning therefore increased wall-clock failure pressure on this long-record slice.

## Higher reasoning helps slightly on shared terminal records

| Mode | Pass | Audit | Validator-valid | Valid rate on common 77 |
|---|---:|---:|---:|---:|
| Light | 34 | 25 | 59 | 76.6% |
| Medium | 37 | 23 | 60 | 77.9% |
| High | 40 | 22 | 62 | 80.5% |

The paired comparisons are small:

- Light to Medium, among 79 mutually terminal UIDs: 2 improved and 1 worsened.
- Medium to High, among 77 mutually terminal UIDs: 2 improved and 0 worsened.
- Light to High, among 78 mutually terminal UIDs: 4 improved and 1 worsened.

This supports a modest validator benefit, but not a broad conclusion that higher reasoning always translates better. Missing outcomes are not random: they are concentrated in long records and become more common as reasoning effort rises.

## Route-level results reveal where the gain comes from

| Route | N | Light valid | Medium valid | High valid | Interpretation |
|---|---:|---:|---:|---:|---|
| Prose | 63 | 40 | 41 | 40 | Essentially flat; missing long records dominate |
| JSON | 17 | 5 | 5 | 6 | High recovers one additional structural pass |
| Leetspeak | 20 | 18 | 18 | 18 | No gain from higher reasoning |

`Valid` means `passed` or `needs_audit`. It is not a human bilingual quality score.

## Leetspeak examples show validator blind spots

### Example A — Medium is better than both Light and High on the intended style

UID: `en-train-00038564-33c102411044`

The source ends with `1337 1T Sk1llZ` and contains a leetspeak response.

- Light translates the response into ordinary Vietnamese with almost no reconstructed leetspeak. It is still marked `needs_audit`, because the current hard validator checks copied English leetspeak more strongly than whether Vietnamese leetspeak density was preserved.
- Medium preserves the literal `1337` concept (`Kỹ năng IT cấp 1337`) and reconstructs the response in Vietnamese leetspeak, for example `Đ4u 71ên, b4n 5ẽ...`.
- High reconstructs Vietnamese leetspeak more cleanly, but drops the literal token `1337`; the validator records `prompt_missing_token:1337`.
- XHigh has the same missing-token warning.
- Sol keeps `1337 1T Sk1llZ` in the prompt and uses a consistent Vietnamese leetspeak response. It preserves the token but partially retains the English leet phrase rather than fully translating it.

For this UID, **Medium best satisfies the written policy**: translate the meaning, preserve the important token, and reconstruct leetspeak in Vietnamese.

### Example B — All Luna modes fail, but the validator is probably wrong

UID: `en-train-00035065-66d455d94051`

The record contains a Python program whose string dictionary and output example are intentionally English/leetspeak code. Light, Medium, and High translate the explanatory prose but preserve the Python code. The validator rejects all three with `an English leetspeak line was copied unchanged`.

That conflicts with the translation rule that code and identifiers should be preserved. Sol produces the same general treatment. This should be classified as a **validator false positive**, not evidence that every model failed the translation.

The validator needs code-fence awareness: copied English/leetspeak inside a preserved code block should not fail unless the source explicitly asks for localized code literals.

### Example C — High passes structure while visibly retaining source leetspeak

UID: `en-train-00038437-2bb61e3a8d34`

- Light and Medium translate both the ordinary-English request and its obfuscated line into Vietnamese/leetspeak.
- High translates the ordinary text but visibly retains the beginning of the original English leetspeak line (`e.xpl41n 1n d3t41l...`). It nevertheless reaches `needs_audit` with no hard error.
- XHigh translates and reconstructs that line in Vietnamese leetspeak (`g1ả1 7hích ch1 7iế7...`).
- Sol also reconstructs the line consistently in Vietnamese leetspeak.

This is a **validator false negative** for High. The current copied-English detector misses some mixed-script or punctuated leetspeak.

## Timeout is caused by a fixed 15-minute wall-clock limit

The runner launches:

```text
codex exec --json --output-schema ... --model gpt-5.6-luna
```

and waits with `proc.communicate(..., timeout=900)`. At 900 seconds it kills the process tree. In the fixed retry runs:

| Mode | Requests | Timed out | Timeout rate | Average elapsed |
|---|---:|---:|---:|---:|
| Light | 53 | 35 | 66.0% | 640 s |
| Medium | 49 | 37 | 75.5% | 725 s |
| High | 49 | 40 | 81.6% | 776 s |

These retry runs are tail-heavy, so the percentages do not estimate normal-dataset latency. They do show that a 900-second cap is too short for many long records, especially with higher reasoning.

`--json` streams Codex events to stdout, but the runner currently uses `communicate`, waits for process completion, calls `parse_codex_stream`, and discards the returned event list. Only a completed `agent_message` becomes the final JSON candidate. Reasoning events do not contain a usable translation object, and the current runner does not persist them. Therefore “some reasoning was streamed” does not mean a partial translation can be validated or saved.

## File output can help durability, but does not remove the timeout

The current Codex session uses `--sandbox read-only`, so Luna cannot create a result file itself. There are two file approaches:

1. **CLI-owned final-message file (`--output-last-message`)**: preferred. The CLI writes the final assistant message to a batch-specific file without asking the model to use a filesystem tool. It reduces parsing fragility, but the file is still written only after the final response completes.
2. **Model-owned JSON file**: requires `workspace-write` and a prompt telling Luna to write a batch-specific file. It can preserve work if the agent writes the file before its final chat message, but tool use adds latency and concurrency/file-collision risks. It does not help if reasoning itself consumes the full 900 seconds before any write.

The strongest design is not file output alone. It is:

- one high-tail/oversized UID per session;
- activity-based timeout that resets when a new event arrives;
- a larger absolute cap, such as 30–45 minutes;
- persist every streamed event and stderr incrementally;
- use `--output-last-message` as the canonical completed result;
- optionally allow a batch-specific workspace-write checkpoint file for extreme records;
- never share output paths between workers.

## Recommended validator changes

1. Split `structural_valid`, `hard_quality_valid`, and `manual_audit_required`; do not call all three simply “pass”.
2. Keep leetspeak candidates in audit, but add an explicit Vietnamese-leetspeak reconstruction score/density check.
3. Ignore copied-English/leetspeak checks inside preserved code fences unless localization of string literals is explicitly required.
4. Expand copied-leetspeak detection for punctuation, Unicode homoglyphs, mixed Cyrillic/Latin text, and split tokens.
5. Preserve required semantic/literal tokens such as `1337`, placeholders, URLs, and identifiers with route-aware rules.
6. Calibrate every new rule against the two known counterexamples above before applying it to the full dataset.

## What this experiment establishes

- Higher reasoning yields a small validator-valid gain on comparable terminal records.
- It does not improve the leetspeak valid rate in this sample.
- Higher reasoning materially increases timeout risk under the current 900-second wall-clock limit.
- Validator outcomes currently contain both false positives and false negatives, so they cannot stand alone as translation-quality labels.
- Medium appears to be the best cost/quality candidate for hard leetspeak, while Light remains attractive for ordinary throughput and High should be reserved for validator-selected cases rather than used universally.

## Further questions

- Would activity-based timeout plus `--output-last-message` recover the 19–20 deferred records without raising cost excessively?
- Does a code-aware leetspeak validator change the apparent 18/20 ceiling?
- On a human-reviewed subset, do the extra High passes represent real quality gains or merely validator compliance?
