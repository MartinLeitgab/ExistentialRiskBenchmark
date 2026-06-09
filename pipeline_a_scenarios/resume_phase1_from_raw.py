#!/usr/bin/env python3
"""Resume PIPE-A7 Phase 1 after raw_responses.json is already written."""

import json

from pipeline_a_scenarios.prompt_validation import (
    analyze_and_rank_variants,
    evaluate_with_judge,
    generate_anomaly_report,
    load_scenarios,
    rerun_failed_responses,
)


def main() -> None:
    scenarios_path = "data/scenarios/seeds_phase1.json"
    raw_path = "data/results/prompt_validation/raw_responses.json"

    scenarios = load_scenarios(scenarios_path)

    with open(raw_path) as f:
        raw_existing = json.load(f)
    error_rows = [r for r in raw_existing if "error" in r]

    if error_rows:
        print(f"Rerunning {len(error_rows)} failed cell(s)...")
        rerun = rerun_failed_responses(
            scenarios_path=scenarios_path,
            raw_path=raw_path,
            only_transient=False,
        )
        raw_responses = rerun["raw_responses"]
        n_retried, n_recovered = rerun["n_retried"], rerun["n_recovered"]
    else:
        print("✓ No error rows in raw_responses.json — skipping rerun.")
        raw_responses = raw_existing
        n_retried, n_recovered = 0, 0

    judge_results = evaluate_with_judge(raw_responses, scenarios)
    analyze_and_rank_variants(judge_results)
    generate_anomaly_report(judge_results)

    with open("data/results/prompt_validation/cost_summary.json") as f:
        study_cost = json.load(f)
    with open("data/results/prompt_validation/judge_cost_summary.json") as f:
        judge_cost = json.load(f)
    total = study_cost.get("total_cost", 0) + judge_cost.get("total_cost", 0)

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE (resumed from raw_responses.json)")
    print("=" * 80)
    print(f"\nTotal cost: ${total:.2f}")
    print(f"Rerun: retried {n_retried}, recovered {n_recovered}")


if __name__ == "__main__":
    main()
