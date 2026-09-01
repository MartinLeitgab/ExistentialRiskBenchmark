#!/usr/bin/env python3
"""Run Judge B only over existing Phase 1 judge_metrics.json (no Judge A re-run)."""

from __future__ import annotations

import argparse
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
from utils.judge_analysis import build_judge_b_phase1_report  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "data" / "results" / "prompt_validation"
APRIL24_OUTPUT = REPO_ROOT / "outputs" / "data_Riccardo042426" / "prompt_validation"
DEFAULT_SCENARIOS = REPO_ROOT / "data" / "scenarios" / "seeds_phase1.json"


def resolve_output_dir(explicit: str | None) -> Path:
    """Prefer explicit path, then April 24 export, then default Phase 1 results."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    if (APRIL24_OUTPUT / "judge_metrics.json").exists():
        return APRIL24_OUTPUT.resolve()
    return DEFAULT_OUTPUT.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retroactive Judge B pass over existing judge_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory containing judge_metrics.json "
            f"(default: {APRIL24_OUTPUT} if present, else {DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIOS),
        help="Path to seeds_phase1.json",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Override path to judge_metrics.json",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip Judge B API calls; re-analyse existing judge_b rows only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run Judge B on all rows (clears existing judge_b first)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip pre-run backup of judge_metrics.json",
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Label for judge_b_phase1_report (default: output dir name)",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)
    metrics_path = Path(args.metrics_path).resolve() if args.metrics_path else output_dir / "judge_metrics.json"
    scenarios_path = Path(args.scenarios).resolve()
    dataset_label = args.dataset_label or output_dir.parent.name

    if not metrics_path.exists():
        raise SystemExit(f"judge_metrics.json not found: {metrics_path}")
    if not scenarios_path.exists():
        raise SystemExit(f"scenarios file not found: {scenarios_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_backup and not args.analyze_only:
        backup_path = (
            output_dir
            / f"judge_metrics.pre_judge_b.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(metrics_path, backup_path)
        print(f"✓ Backed up judge metrics to {backup_path}")

    scenarios = load_scenarios(str(scenarios_path))

    if args.analyze_only:
        import json

        with open(metrics_path, encoding="utf-8") as handle:
            judge_results = json.load(handle)
        n_with_b = sum(1 for r in judge_results if "judge_b" in r)
        print(f"Analyze-only: {n_with_b}/{len(judge_results)} rows have judge_b")
        if n_with_b == 0:
            raise SystemExit("No judge_b rows — run without --analyze-only first")
    else:
        judge_results = evaluate_judge_b_retroactive(
            scenarios,
            output_dir=str(output_dir),
            metrics_path=str(metrics_path),
            force=args.force,
        )

    analyze_and_rank_variants(
        judge_results,
        output_dir=str(output_dir),
        dataset_label=dataset_label,
    )
    generate_anomaly_report(judge_results, output_dir=str(output_dir))

    report = build_judge_b_phase1_report(judge_results, dataset_label=dataset_label)
    gate = report["decision_gate"]["status"]
    n_with_b = sum(1 for r in judge_results if "judge_b" in r)

    print("\n" + "=" * 80)
    print("JUDGE B RETROACTIVE PASS COMPLETE")
    print("=" * 80)
    print(f"\nOutput dir: {output_dir}")
    print(f"Rows with judge_b: {n_with_b}/{len(judge_results)}")
    print(f"Decision gate: {gate}")
    for entry in report.get("judge_b_by_model", []):
        val = entry.get("judge_b_validation", {})
        print(
            f"  {entry['model']}: "
            f"RI={entry['rationalization_index']:.1f} "
            f"washing={entry['alignment_washing_rate']:.2%} "
            f"faking={entry['alignment_faking_rate']:.2%} "
            f"IVT ρ={val.get('ivt_correlation_rho')} status={val.get('status')}"
        )


if __name__ == "__main__":
    main()
