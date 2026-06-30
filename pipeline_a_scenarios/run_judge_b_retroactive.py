#!/usr/bin/env python3
"""Run Judge B only over existing Phase 1 judge_metrics.json (no Judge A re-run)."""

import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PIPELINE_DIR))

from prompt_validation import (  # noqa: E402
    analyze_and_rank_variants,
    evaluate_judge_b_retroactive,
    generate_anomaly_report,
    load_scenarios,
)


def main() -> None:
    output_dir = Path("data/results/prompt_validation")
    metrics_path = output_dir / "judge_metrics.json"
    scenarios_path = "data/scenarios/seeds_phase1.json"

    backup_path = (
        output_dir
        / f"judge_metrics.pre_judge_b.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    shutil.copy2(metrics_path, backup_path)
    print(f"✓ Backed up judge metrics to {backup_path}")

    scenarios = load_scenarios(scenarios_path)
    judge_results = evaluate_judge_b_retroactive(scenarios, output_dir=str(output_dir))
    analyze_and_rank_variants(judge_results, output_dir=str(output_dir))
    generate_anomaly_report(judge_results)

    n_with_b = sum(1 for r in judge_results if "judge_b" in r)
    print("\n" + "=" * 80)
    print("JUDGE B RETROACTIVE PASS COMPLETE")
    print("=" * 80)
    print(f"\nRows with judge_b: {n_with_b}/{len(judge_results)}")


if __name__ == "__main__":
    main()
