"""
Phase 1: Prompt Validation Study
Integrates PIPE-A1, PIPE-A2, INFRA-4, and INFRA-6
"""

import json
from pathlib import Path
from typing import Dict, List

from utils.llm_client import UnifiedLLMClient
from utils.prompt_generator import generate_all_variants, generate_prompt
from utils.cost_tracker import CostTracker
from utils.judge_client import JudgeClient
from utils.judge_analysis import (
    aggregate_by_variant,
    compute_composite_score,
    detect_patterns
)


def run_validation_study(
    scenarios_path: str = "data/scenarios/seeds_phase1.json",
    models: List[str] = ["claude-sonnet-4-20250514", "gpt-5.2", "gemini-3.0"],
    runs_per_config: int = 2,
    output_dir: str = "data/results/prompt_validation"
):
    """
    Phase 1: Run validation study on 6 scenarios × 15 variants × 3 models.
    
    Returns:
        Dict with keys: raw_responses, judge_metrics, variant_rankings
    """
    print("="*80)
    print("PIPE-A7 PHASE 1: PROMPT VALIDATION STUDY")
    print("="*80)
    
    # Initialize cost tracker (INFRA-4)
    cost_tracker = CostTracker(user_id="pipe_a7_phase1")
    
    # Load scenarios (PIPE-A1)
    print("\n1. Loading scenarios from PIPE-A1...")
    with open(scenarios_path) as f:
        scenarios = json.load(f)
    print(f"   Loaded {len(scenarios)} seed scenarios")
    
    # Generate all variants (PIPE-A2)
    print("\n2. Loading prompt variants from PIPE-A2...")
    variants = generate_all_variants()
    print(f"   Generated {len(variants)} prompt variants")
    
    # Initialize model clients
    print("\n3. Initializing model clients...")
    clients = {}
    for model in models:
        if "claude" in model:
            clients[model] = UnifiedLLMClient(provider="anthropic", model=model)
        elif "gpt" in model:
            clients[model] = UnifiedLLMClient(provider="openai", model=model)
        else:  # gemini
            clients[model] = UnifiedLLMClient(provider="google", model=model)
    
    # Execute test matrix
    print(f"\n4. Executing test matrix...")
    print(f"   {len(scenarios)} scenarios × {len(variants)} variants × {len(models)} models × {runs_per_config} runs")
    total_calls = len(scenarios) * len(variants) * len(models) * runs_per_config
    print(f"   Total API calls: {total_calls}")
    
    results = []
    call_count = 0
    
    for scenario in scenarios:
        for variant in variants:
            # Generate prompt (PIPE-A2)
            prompt_obj, variant_id = generate_prompt(
                scenario["context"],
                scenario["action_a"],
                scenario["action_b"],
                variant["dimensions"]
            )
            
            for model in models:
                client = clients[model]
                
                for run_idx in range(runs_per_config):
                    call_count += 1
                    
                    try:
                        # Call model
                        response = client.generate(
                            prompt=prompt_obj["user_prompt"],
                            system_prompt=prompt_obj.get("system_prompt"),
                            temperature=0,  # Consistency testing
                            max_tokens=500
                        )
                        
                        # Extract provider for logging
                        provider = (
                            "anthropic" if "claude" in model else
                            "openai" if "gpt" in model else
                            "google"
                        )
                        
                        # Log cost (INFRA-4)
                        cost_tracker.log_api_call(
                            provider=provider,
                            model=model,
                            usage=response.get("usage", {}),
                            operation="prompt_validation_phase1",
                            metadata={
                                "scenario_id": scenario["id"],
                                "variant_id": variant_id,
                                "run": run_idx
                            }
                        )
                        
                        result = {
                            "scenario_id": scenario["id"],
                            "variant_id": variant_id,
                            "provider": provider,
                            "model": model,
                            "run": run_idx,
                            "response_text": response["content"],
                            "usage": response["usage"],
                            "dimensions": variant["dimensions"]
                        }
                        results.append(result)
                        
                    except Exception as e:
                        print(f"  ⚠ Error on {scenario['id']}/{variant_id}/{model}: {e}")
                        results.append({
                            "scenario_id": scenario["id"],
                            "variant_id": variant_id,
                            "provider": provider,
                            "model": model,
                            "run": run_idx,
                            "error": str(e)
                        })
                    
                    # Progress
                    if call_count % 50 == 0:
                        print(f"   Progress: {call_count}/{total_calls}")
    
    # Save raw responses
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raw_path = f"{output_dir}/raw_responses.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved {len(results)} raw responses to {raw_path}")
    
    # Generate cost report (INFRA-4)
    cost_summary = cost_tracker.get_summary()
    cost_path = f"{output_dir}/cost_summary.json"
    with open(cost_path, "w") as f:
        json.dump(cost_summary, f, indent=2)
    print(f"✓ Cost summary: ${cost_summary['total_cost']:.2f}")
    
    return {
        "raw_responses": results,
        "cost_summary": cost_summary
    }


def evaluate_with_judge(
    raw_responses: List[Dict],
    scenarios: List[Dict],
    output_dir: str = "data/results/prompt_validation"
) -> List[Dict]:
    """
    Phase 1: Evaluate all responses using INFRA-6.
    
    Args:
        raw_responses: Output from run_validation_study()
        scenarios: Scenario metadata for evaluation context
    
    Returns:
        List of judge evaluation results
    """
    print("\n5. Evaluating responses with INFRA-6...")
    
    # Initialize judge (INFRA-6)
    cost_tracker = CostTracker(user_id="pipe_a7_phase1_judge")
    judge = JudgeClient(
        model="gpt-4o-mini",
        temperature=0,
        cost_tracker=cost_tracker
    )
    
    # Filter out errors
    valid_responses = [r for r in raw_responses if "error" not in r]
    print(f"   Evaluating {len(valid_responses)} valid responses")
    
    # Prepare evaluation inputs
    eval_inputs = []
    for resp in valid_responses:
        scenario = next(s for s in scenarios if s["id"] == resp["scenario_id"])
        eval_inputs.append({
            "scenario_id": resp["scenario_id"],
            "variant_id": resp["variant_id"],
            "provider": resp["provider"],
            "model": resp["model"],
            "run": resp["run"],
            "scenario": scenario,
            "response_text": resp["response_text"]
        })
    
    # Run judge evaluation
    judge_results = []
    for i, eval_input in enumerate(eval_inputs):
        result = judge.evaluate_response(
            scenario_context=eval_input["scenario"]["context"],
            action_a=eval_input["scenario"]["action_a"],
            action_b=eval_input["scenario"]["action_b"],
            model_response=eval_input["response_text"],
            evaluation_context="pipe_a7_phase1"
        )
        
        # Add metadata
        result.update({
            "scenario_id": eval_input["scenario_id"],
            "variant_id": eval_input["variant_id"],
            "provider": eval_input["provider"],
            "model": eval_input["model"],
            "run": eval_input["run"]
        })
        judge_results.append(result)
        
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(eval_inputs)}")
    
    # Save judge metrics
    metrics_path = f"{output_dir}/judge_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(judge_results, f, indent=2)
    print(f"✓ Saved judge metrics to {metrics_path}")
    
    # Judge cost summary
    judge_cost = cost_tracker.get_summary()
    judge_cost_path = f"{output_dir}/judge_cost_summary.json"
    with open(judge_cost_path, "w") as f:
        json.dump(judge_cost, f, indent=2)
    print(f"✓ Judge cost: ${judge_cost['total_cost']:.2f}")
    
    return judge_results


def analyze_and_rank_variants(
    judge_results: List[Dict],
    output_dir: str = "data/results/prompt_validation"
) -> Dict:
    """
    Phase 1: Aggregate judge scores and rank variants.
    
    Returns:
        Dict with variant rankings and recommendations
    """
    print("\n6. Analyzing judge results and ranking variants...")
    
    # Aggregate by variant (INFRA-6 utilities)
    variant_summary = aggregate_by_variant(judge_results, group_by="variant_id")
    
    # Compute composite scores
    for summary in variant_summary:
        summary["composite_score"] = compute_composite_score(
            avg_scores=summary["avg_scores"],
            consistency=summary["consistency"],
            weights={"comprehension": 0.3, "format_valid": 0.4, "consistency": 0.3}
        )
    
    # Sort by composite score
    variant_summary.sort(key=lambda x: x["composite_score"], reverse=True)
    
    # Detect patterns (INFRA-6 utilities)
    patterns = detect_patterns(judge_results, threshold=0.3)
    
    # Generate recommendations
    top_variants = variant_summary[:7]  # Top 7 for Phase 2
    
    recommendations = {
        "top_variants": [v["variant_id"] for v in top_variants],
        "variant_rankings": variant_summary,
        "detected_patterns": patterns,
        "recommendation_summary": generate_recommendation_text(top_variants, patterns)
    }
    
    # Save rankings
    rankings_path = f"{output_dir}/variant_rankings.json"
    with open(rankings_path, "w") as f:
        json.dump(recommendations, f, indent=2)
    print(f"✓ Saved variant rankings to {rankings_path}")
    
    # Print top variants
    print("\n✓ TOP 7 VARIANTS FOR PHASE 2:")
    for i, variant in enumerate(top_variants):
        print(f"   {i+1}. {variant['variant_id']}: {variant['composite_score']:.2f}")
        print(f"      Comprehension: {variant['avg_scores']['comprehension']:.2f}")
        print(f"      Format Valid: {variant['avg_scores']['format_valid']:.2f}")
        print(f"      Consistency: {variant['consistency']:.2f}")
    
    # Print patterns
    if any(patterns.values()):
        print("\n⚠ DETECTED PATTERNS:")
        for pattern_type, affected_variants in patterns.items():
            if affected_variants:
                print(f"   {pattern_type}: {', '.join(affected_variants)}")
    
    return recommendations


def generate_recommendation_text(top_variants: List[Dict], patterns: Dict) -> str:
    """Generate human-readable recommendation summary."""
    
    lines = [
        "PHASE 1 RECOMMENDATIONS:",
        "",
        f"Proceed to Phase 2 with top {len(top_variants)} variants.",
        ""
    ]
    
    # Highlight concerns
    if patterns.get("high_refusal_rate"):
        lines.append(f"⚠ High refusal rate: {', '.join(patterns['high_refusal_rate'])}")
    if patterns.get("low_comprehension"):
        lines.append(f"⚠ Low comprehension: {', '.join(patterns['low_comprehension'])}")
    if patterns.get("high_sa_awareness"):
        lines.append(f"⚠ High SA awareness: {', '.join(patterns['high_sa_awareness'])}")
    if patterns.get("parsing_issues"):
        lines.append(f"⚠ Parsing issues: {', '.join(patterns['parsing_issues'])}")
    
    if not any(patterns.values()):
        lines.append("✓ No major issues detected across variants.")
    
    lines.extend([
        "",
        "Next steps:",
        "1. Review top variants for substantive differences",
        "2. Run Phase 2 batch testing on 500 scenarios",
        "3. Make final variant selection based on Phase 2 results"
    ])
    
    return "\n".join(lines)


def generate_anomaly_report(
    judge_results: List[Dict],
    output_dir: str = "data/results/prompt_validation"
):
    """Generate detailed anomaly report."""
    
    print("\n7. Generating anomaly report...")
    
    from collections import Counter
    
    # Collect all anomalies
    all_anomalies = []
    for result in judge_results:
        for anomaly in result.get("anomalies", []):
            all_anomalies.append({
                "type": anomaly,
                "variant_id": result["variant_id"],
                "scenario_id": result["scenario_id"],
                "model": result["model"]
            })
    
    # Aggregate
    anomaly_counts = Counter(a["type"] for a in all_anomalies)
    by_variant = {}
    for anomaly in all_anomalies:
        variant_id = anomaly["variant_id"]
        if variant_id not in by_variant:
            by_variant[variant_id] = []
        by_variant[variant_id].append(anomaly)
    
    # Generate report
    report_lines = [
        "# PIPE-A7 Phase 1: Anomaly Report",
        "",
        "## Summary",
        f"Total anomalies detected: {len(all_anomalies)}",
        "",
        "## Anomaly Types",
        ""
    ]
    
    for anomaly_type, count in anomaly_counts.most_common():
        report_lines.append(f"- **{anomaly_type}**: {count} occurrences")
    
    report_lines.extend([
        "",
        "## By Variant",
        ""
    ])
    
    for variant_id in sorted(by_variant.keys()):
        variant_anomalies = by_variant[variant_id]
        report_lines.append(f"### {variant_id}")
        report_lines.append(f"Total: {len(variant_anomalies)}")
        
        variant_counts = Counter(a["type"] for a in variant_anomalies)
        for anom_type, count in variant_counts.most_common():
            report_lines.append(f"- {anom_type}: {count}")
        report_lines.append("")
    
    # Save report
    report_path = f"{output_dir}/anomaly_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"✓ Saved anomaly report to {report_path}")


def main_phase1():
    """Run complete Phase 1 pipeline."""
    
    # Step 1: Run validation study
    study_results = run_validation_study()
    
    # Step 2: Load scenarios for judge context
    with open("data/scenarios/seeds_phase1.json") as f:
        scenarios = json.load(f)
    
    # Step 3: Evaluate with judge
    judge_results = evaluate_with_judge(
        study_results["raw_responses"],
        scenarios
    )
    
    # Step 4: Analyze and rank
    recommendations = analyze_and_rank_variants(judge_results)
    
    # Step 5: Generate anomaly report
    generate_anomaly_report(judge_results)
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print("\nReview outputs in data/results/prompt_validation/:")
    print("  - raw_responses.json")
    print("  - judge_metrics.json")
    print("  - variant_rankings.json")
    print("  - anomaly_report.md")
    print("  - cost_summary.json")
    print("  - judge_cost_summary.json")
    print("\nTotal cost:", study_results["cost_summary"]["total_cost"] + 
          json.load(open("data/results/prompt_validation/judge_cost_summary.json"))["total_cost"])


if __name__ == "__main__":
    main_phase1()