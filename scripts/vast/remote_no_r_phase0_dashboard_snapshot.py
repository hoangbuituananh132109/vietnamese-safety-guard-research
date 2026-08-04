from __future__ import annotations

import json
import os


os.environ.update(
    {
        "PHASE0_CONFIG_PATH": "configs/phase0_no_r_experiments.json",
        "PHASE0_EXTRA_CONFIG_PATHS": "configs/e6_no_r_binary_ablation.json",
        "PHASE0_STATE_PATH": "reports/e6_no_r_binary_ablation/pipeline_state.json",
        "PHASE0_COUNT_CACHE_PATH": "reports/no_r_phase0/dashboard_manifest_counts.json",
        "PHASE0_EXPERIMENT_RUN_ROOT": "reports/no_r_phase0/experiment_runs",
        "PHASE0_EVALUATION_MATRIX_ROOT": "reports/no_r_phase0/evaluation_matrix",
        "PHASE0_REFERENCE_RUN_ID": "E3NR-M-E-8K",
        "PHASE0_RUN_ORDER": ",".join(
            [
                "E1NR-G-EV-512",
                "E2NR-M-EV-GLI-COMPAT-8K",
                "E3NR-M-E-8K",
                "E4NR-M-EV-MATCHED-8K",
                "E5NR-M-EV-FULL-8K",
                "E6NR-M-EV-FULL-BIN-8K",
                "E7NR-M-SCHEMA-EV-8K",
            ]
        ),
        "PHASE0_RUN_NAMES_JSON": json.dumps(
            {
                "E1NR-G-EV-512": "E1 no-R · GLiGuard EN+VI 512",
                "E2NR-M-EV-GLI-COMPAT-8K": "E2 no-R · mmBERT EN+VI GLi-compatible",
                "E3NR-M-E-8K": "E3 no-R · mmBERT English-only 8K",
                "E4NR-M-EV-MATCHED-8K": "E4 no-R · mmBERT matched EN+VI 8K",
                "E5NR-M-EV-FULL-8K": "E5 no-R · mmBERT full EN+VI 8K",
                "E6NR-M-EV-FULL-BIN-8K": "E6 no-R · mmBERT binary-only EN+VI 8K",
                "E7NR-M-SCHEMA-EV-8K": "E7 no-R · mmBERT dynamic schema 8K",
            }
        ),
    }
)

from remote_training_dashboard_snapshot import main


if __name__ == "__main__":
    main()
