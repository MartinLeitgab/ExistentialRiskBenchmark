"""
Phase 2: Analyze batch testing results and make final variant selection
"""

import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

from utils.llm_client import UnifiedLLMClient, BatchHandle


def load_batch_results(batch_dir: str = "data/batches/variant_testing") -> pd.DataFrame:
    """Load and parse all batch results using UnifiedLLMClient."""
    
    print("\n1. Loading batch results...")
    
    with open(f"{batch_dir}/batch_handles.json") as f:
        batch_handles_data = json.load(f)
    
    all_results = []
    
    for key, handle_data in batch_handles_data.items():
        variant_id = handle_data["variant_id"]
        model = handle_data["model"]
        batch_id = handle_data["batch_id"]
        provider = handle_data["provider"]
        
        print(f"\n   Retrieving: {variant_id} × {model} (batch_id: {batch_id})")
        
        try:
            # Initialize client
            if provider == "anthropic":
                client = UnifiedLLMClient(provider="anthropic", model=model)
            elif provider == "openai":
                client = UnifiedLLMClient(provider="openai", model=model)
            else:
                client = UnifiedLLMClient(provider="google", model=model)
            
            # Create BatchHandle object
            from utils.llm_client import BatchHandle
            batch_handle = BatchHandle(provider=provider, id=batch_id)
            
            # Retrieve results (this will poll until complete)
            print(f"   → Polling for completion (may take a while)...")
            results_dict = client.retrieve_batch_results(batch_handle, timeout=1800)
            
            print(f"   → Retrieved {len(results_dict)} responses")
            
            # Parse each response
            for custom_id, response_text in results_dict.items():
                # Extract scenario_id from custom_id
                # Format: {scenario_id}_{variant_id}_{model}
                scenario_id = custom_id.split("_")[0]
                
                # Parse response
                parsed_choice = parse_response(response_text)
                
                all_results.append({
                    "scenario_id": scenario_id,
                    "variant_id": variant_id,
                    "model": model,
                    "raw_response": response_text,
                    "parsed_choice": parsed_choice
                })
                
        except Exception as e:
            print(f"   ✗ Error retrieving batch {batch_id}: {e}")
            continue
    
    print(f"\n✓ Loaded {len(all_results)} total responses")
    return pd.DataFrame(all_results)


def analyze_variant_consistency(df: pd.DataFrame, output_dir: str):
    """
    Compute cross-variant Spearman correlations.
    High correlation = variants elicit similar preferences.
    """
    
    print("\n2. Computing variant consistency (Spearman correlations)...")
    
    variants = df["variant_id"].unique()
    models = df["model"].unique()
    
    for model in models:
        print(f"\n   Model: {model}")
        
        model_df = df[df["model"] == model]
        
        # Build response matrix: rows=scenarios, cols=variants
        scenarios = model_df["scenario_id"].unique()
        response_matrix = np.zeros((len(scenarios), len(variants)))
        
        for i, scenario_id in enumerate(scenarios):
            for j, variant_id in enumerate(variants):
                choice = model_df[
                    (model_df["scenario_id"] == scenario_id) &
                    (model_df["variant_id"] == variant_id)
                ]["parsed_choice"].values[0]
                
                # Encode: A=1, B=0, equal=0.5, unparseable=NaN
                if choice == "A":
                    response_matrix[i, j] = 1.0
                elif choice == "B":
                    response_matrix[i, j] = 0.0
                elif choice == "equal":
                    response_matrix[i, j] = 0.5
                else:
                    response_matrix[i, j] = np.nan
        
        # Compute correlations
        corr_matrix = np.zeros((len(variants), len(variants)))
        
        for i, var1 in enumerate(variants):
            for j, var2 in enumerate(variants):
                if i >= j:
                    continue
                
                # Remove NaN rows
                mask = ~(np.isnan(response_matrix[:, i]) | np.isnan(response_matrix[:, j]))
                
                if mask.sum() > 10:
                    corr, pval = spearmanr(
                        response_matrix[mask, i],
                        response_matrix[mask, j]
                    )
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    
                    print(f"      {var1} × {var2}: r={corr:.3f} (p={pval:.4f})")
        
        # Visualize
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
        plt.colorbar(label="Spearman r")
        plt.xticks(range(len(variants)), variants, rotation=45, ha="right")
        plt.yticks(range(len(variants)), variants)
        plt.title(f"Variant Correlation Matrix - {model}")
        plt.tight_layout()
        
        fig_path = f"{output_dir}/figures/variant_correlations_{model}.png"
        Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150)
        print(f"      → Saved heatmap: {fig_path}")


def analyze_parsing_success(df: pd.DataFrame):
    """Compute parsing success rates by variant."""
    
    print("\n3. Analyzing parsing success rates...")
    
    for variant in df["variant_id"].unique():
        variant_df = df[df["variant_id"] == variant]
        success_rate = (variant_df["parsed_choice"] != "UNPARSEABLE").mean() * 100
        print(f"   {variant}: {success_rate:.1f}%")


def recommend_final_variants(df: pd.DataFrame, output_dir: str) -> Dict:
    """Make final variant selection recommendation."""
    
    print("\n4. Generating final recommendations...")
    
    # Criteria:
    # 1. High correlation with other variants (>0.8)
    # 2. High parsing success (>95%)
    # 3. Low refusal rate (from Phase 1 data)
    
    recommendations = {
        "primary_variant": None,
        "secondary_variants": [],
        "rationale": []
    }
    
    # Load Phase 1 patterns
    with open(f"{output_dir}/variant_rankings.json") as f:
        phase1_data = json.load(f)
    
    patterns = phase1_data.get("detected_patterns", {})
    
    # Simple heuristic: highest composite score from Phase 1 + good Phase 2 performance
    top_variant = phase1_data["variant_rankings"][0]["variant_id"]
    
    # Parsing check
    parsing_rate = (
        df[df["variant_id"] == top_variant]["parsed_choice"] != "UNPARSEABLE"
    ).mean()
    
    if parsing_rate >= 0.95:
        recommendations["primary_variant"] = top_variant
        recommendations["rationale"].append(
            f"{top_variant} selected as primary: highest Phase 1 score + {parsing_rate*100:.1f}% parsing success"
        )
    else:
        # Fallback to second-best
        top_variant = phase1_data["variant_rankings"][1]["variant_id"]
        recommendations["primary_variant"] = top_variant
        recommendations["rationale"].append(
            f"{top_variant} selected as primary: fallback due to parsing issues with top variant"
        )
    
    # Select 2 secondary variants
    # Criteria: different SA levels for robustness
    remaining_variants = [
        v["variant_id"] for v in phase1_data["variant_rankings"][1:6]
        if v["variant_id"] != recommendations["primary_variant"]
    ]
    recommendations["secondary_variants"] = remaining_variants[:2]
    
    # Save
    rec_path = f"{output_dir}/final_variant_selection.json"
    with open(rec_path, "w") as f:
        json.dump(recommendations, f, indent=2)
    
    print("\n" + "="*80)
    print("FINAL VARIANT SELECTION")
    print("="*80)
    print(f"Primary: {recommendations['primary_variant']}")
    print(f"Secondary: {', '.join(recommendations['secondary_variants'])}")
    print("\nRationale:")
    for r in recommendations["rationale"]:
        print(f"  - {r}")
    
    return recommendations


def parse_response(raw_text: str) -> str:
    """Parse A/B/equal from response."""
    import re
    
    patterns = [
        r"^\s*([ABC])\s*$",
        r"choice:\s*([ABC])",
        r"answer:\s*([ABC])",
        r"\b([ABC])\b"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            choice = match.group(1).upper()
            if choice == "C":
                return "equal"
            return choice
    
    return "UNPARSEABLE"


def main_phase2():
    """Run complete Phase 2 analysis pipeline."""
    
    output_dir = "data/results/prompt_validation"
    
    # Load results
    df = load_batch_results()
    
    # Analyze
    analyze_variant_consistency(df, output_dir)
    analyze_parsing_success(df)
    recommendations = recommend_final_variants(df, output_dir)
    
    print("\n✓ Phase 2 analysis complete")
    print(f"✓ Final recommendations saved to {output_dir}/final_variant_selection.json")


if __name__ == "__main__":
    main_phase2()