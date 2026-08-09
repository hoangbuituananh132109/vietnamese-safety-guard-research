# GitHub publication plan for the Luna/Sol work

## Repository state

The public repository is configured as:

- remote: `https://github.com/hoangbuituananh132109/vietnamese-safety-guard-research.git`;
- branch: `main`;
- Git metadata directory: `.git-public`;
- work tree: the project root.

The ordinary `.git` directory is empty, so plain `git status` currently reports that the directory is not a repository. Until this layout is deliberately normalized, use:

```powershell
git --git-dir=.git-public --work-tree=. status
```

Do not delete, move, or replace either Git metadata directory without first backing up `.git-public` and confirming the remote/HEAD state.

## Publication claim

The GitHub update may claim:

- a complete Luna/Sol EN→VI translation pipeline;
- 45,416/45,416 translated UIDs;
- reproducible per-UID retry, validation, hard-case routing, Sol Web fallback, provenance, and training-readiness checks;
- a frozen protocol for comparing the new corpus with the earlier Gemini corpus.

It must not yet claim:

- that Luna/Sol produces better translations than Gemini overall;
- that a Luna/Sol-trained model beats the existing Gemini-trained models;
- that the new pipeline improves SEA Bench;
- that the project has produced a new native Vietnamese safety corpus.

Those claims require the training/evaluation protocol to be executed.

## Never commit these local artifacts

- `API.txt`, `.env`, API keys, tokens, cookies, or authentication files;
- `configs/key_assignments.json` or any provider-account assignment;
- `data/`, translated JSONL, source records, review queues, and browser results;
- `models/`, checkpoints, adapters, caches, and downloaded weights;
- `results/`, raw run outputs, large prediction dumps, and private logs;
- generated `web/` pages because they embed source records and translations;
- virtual environments, `.runtime`, pytest caches, temporary `.agents` files;
- archives, office bundles, and local GPU recovery state.

Dataset release should use a separate Hugging Face dataset repository or gated storage with a dataset card, license review, checksums, provenance fields, and safety-content notice. Large model artifacts belong in a model repository, not Git.

## Recommended public commit scope

### Commit 1 — documentation and claim boundary

- `README.md` update linking the Luna/Sol work;
- `docs/LUNA_SOL_TRANSLATION_RESEARCH_HANDOFF.md`;
- `docs/LUNA_SOL_TRAINING_EVALUATION_PROTOCOL.md`;
- `docs/GITHUB_PUBLICATION_PLAN_LUNA_SOL.md`;
- selected small Markdown reports that contain no embedded source records.

### Commit 2 — translation pipeline and tests

- Luna prompt versions needed to reproduce ablations;
- final Luna and Sol prompt templates;
- Luna runner, queue recovery, retry, and analysis tools;
- Sol Web builders and local result receivers;
- Sol validators/finalizers;
- final dataset merge and materialization-audit tools;
- relevant unit tests.

### Commit 3 — compact evidence

- aggregate JSON summaries with counts and metrics only;
- no raw prompts, translations, source strings, predictions, or credentials;
- checksums and schemas where redistribution is allowed.

Keeping commits thematic makes review and future citation easier than one large commit containing every untracked historical artifact.

## Candidate Luna/Sol code files

At minimum, review these for inclusion:

```text
configs/luna_fresh_translation_prompt_v1.md
configs/luna_fresh_translation_prompt_v2.md
configs/luna_fresh_translation_prompt_v3_sandwich.md
configs/luna_fresh_translation_prompt_v4_leet_demo.md
configs/luna_fresh_translation_prompt_v5_leet_strict.md
configs/luna_fresh_translation_prompt_v6_json_keys.md
configs/sol_web_fallback_repair_prompt.md
configs/sol_web_hard_ablation_prompt.md
docs/luna_hard_case_escalation_plan.md
docs/luna_overnight_runbook.md
docs/translation_pipeline_roadmap.md
scripts/run_luna_overnight_20260801.ps1
scripts/run_luna_remaining_light_retry2.ps1
tools/luna_translation_pilot.py
tools/luna_overnight_runner.py
tools/prepare_luna_remaining_queue.py
tools/recover_overlapped_luna_run.py
tools/build_sol_fallback_after_luna.py
tools/build_sol_missing_25_web.py
tools/sol_result_receiver.py
tools/sol_missing_25_receiver.py
tools/validate_sol_fallback_results.py
tools/validate_sol_missing_25_results.py
tools/finalize_sol_fallback_validation.py
tools/finalize_sol_missing_25_validation.py
tools/build_luna_sol_training_dataset.py
tools/audit_luna_sol_training_readiness.py
tests/test_luna_overnight_runner.py
```

This is a review list, not a command to stage everything blindly.

## Required `.gitignore` additions

Before staging, ensure these patterns exist:

```gitignore
.agents/
configs/key_assignments.json
web/
*.log
GPU_RUN_STATE.md
```

The existing rules already exclude `API.txt`, environments, `data/`, models, results, artifacts, exports, external downloads, archives, and private key formats.

## Pre-commit checks

1. Inspect the exact public-repo status:

```powershell
git --git-dir=.git-public --work-tree=. status --short
```

2. Never use `git add .` in the current dirty work tree. Stage an explicit file list.

3. Search staged content for secrets and local paths:

```powershell
git --git-dir=.git-public --work-tree=. diff --cached --check
git --git-dir=.git-public --work-tree=. diff --cached --name-only
git --git-dir=.git-public --work-tree=. diff --cached | Select-String -Pattern 'API[_-]?KEY|Bearer |sk-[A-Za-z0-9]|AIza|D:\\Downloads|C:\\Users'
```

4. Run focused tests and readiness checks with the local environment.

5. Inspect staged file sizes; do not commit generated datasets or model artifacts.

6. Review the staged diff in the Codex/Git UI before committing.

## Suggested branch and commit flow

After the documentation and ignore rules are reviewed:

```powershell
git --git-dir=.git-public --work-tree=. switch -c feature/luna-sol-translation-pipeline

git --git-dir=.git-public --work-tree=. add -- README.md .gitignore docs/LUNA_SOL_TRANSLATION_RESEARCH_HANDOFF.md docs/LUNA_SOL_TRAINING_EVALUATION_PROTOCOL.md docs/GITHUB_PUBLICATION_PLAN_LUNA_SOL.md
git --git-dir=.git-public --work-tree=. commit -m "docs: document Luna Sol translation study"
```

Stage pipeline code in a second explicit command after reviewing the candidate list. Push only after tests, secret scanning, license checks, and final diff review:

```powershell
git --git-dir=.git-public --work-tree=. push -u origin feature/luna-sol-translation-pipeline
```

Open a draft pull request whose description states that downstream Luna/Sol-versus-Gemini training is pending.

## Data and release documentation still needed

Before a public dataset release, add:

- source dataset identity and license;
- translation-provider/model/date metadata;
- record-level provenance schema;
- validator version and known limitations;
- harmful-content and privacy notice;
- checksums and split counts;
- intended uses and out-of-scope uses;
- explanation of the 356 Sol-routed records and two accepted known issues;
- statement that native Vietnamese data generation is not part of this release.

## Current blockers to a final paper claim

- no Luna/Sol-trained downstream checkpoint yet;
- no 2×2 translated-test comparison;
- no new SEA Bench result;
- no blinded human Gemini-versus-Luna/Sol preference study;
- provider cost/latency accounting not yet normalized;
- native Vietnamese data track remains parked.
