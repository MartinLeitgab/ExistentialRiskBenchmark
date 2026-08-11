"""Runner for the proto_01 seven-axis ablation (issue #46, PR #58).

Runs `data/scenarios/ablation_scenarios.json` — the baseline plus 14 single-axis
variants — on the locked production variant `fp-abs-ftc-auto-t10-reg-0-0-0`, then
scores every response with Judge A.

Historical status: this ablation was executed on 2026-07-28 and returned 0%
instrumental choice in all 15 cells on all three models
(`outputs/data_Pooja072826`, written up as `data/findings.md` §17). The null was
design-induced — `proto_01_v4` is a dominated baseline and `IC1 × PH1` is an
immovable wall (§17c/§17d) — so the axis → effect library this run was meant to
produce could not be built from it, and issue #46 was closed as superseded by the
probe series in PR #64. The script is kept because the dataset it produced is
cited in findings, and because it is the only runner that pins
`run_validation_study` to a single prompt variant.

The 2026-07-28 run used the then-current targets `claude-opus-4-8` / `gpt-5.5`
against a `gpt-5.6-sol` judge — one target generation behind the judge (§17d
"secondary observations"). The defaults below are the canonical models from
`docs/forward_plan.md`; a re-run therefore will NOT be comparable to
`data_Pooja072826` on model identity.

Note this path calls `run_validation_study`, which builds `UnifiedLLMClient` with
the default cache setting. For repeat-sampling probes prefer
`scripts/probe_proto01_guards.py`, which disables the cache so repeats are
genuine independent draws (`findings.md` §8a).

Usage:
    python -u scripts/run_proto_01_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline_a_scenarios.prompt_validation import (  # noqa: E402
    run_validation_study,
    evaluate_with_judge,
    load_scenarios,
)

SCENARIOS_PATH = "data/scenarios/ablation_scenarios.json"
OUTPUT_DIR = "outputs/proto_01_ablation"
VARIANT = "fp-abs-ftc-auto-t10-reg-0-0-0"
MODELS = ["claude-opus-5", "gpt-5.6-sol", "gemini-3.1-pro-preview"]


def main() -> None:
    study = run_validation_study(
        scenarios_path=SCENARIOS_PATH,
        models=MODELS,
        runs_per_config=2,
        output_dir=OUTPUT_DIR,
        variants_filter=[VARIANT],
    )

    scenarios = load_scenarios(SCENARIOS_PATH)
    evaluate_with_judge(
        study["raw_responses"],
        scenarios,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
