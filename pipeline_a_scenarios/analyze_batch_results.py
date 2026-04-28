"""
Phase 2: Analyze batch testing results and make final variant selection.

Fixes applied:
- Added missing `from typing import Dict` import (caused ImportError on module load)
- Added plot P2-2: Phase 1 authenticity score vs Phase 2 IC-choice consistency
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from typing import Dict   # FIX: was missing, causing ImportError on every test

from utils.llm_client import UnifiedLLMClient, BatchHandle


# ---------------------------------------------------------------------------
# Load batch results
# ---------------------------------------------------------------------------

def load_batch_results(batch_dir: str = "data/batches/variant_testing") -> pd.DataFrame:
    """Load and parse all batch results using UnifiedLLMClient."""

    print("\n1. Loading batch results...")

    with open(f"{batch_dir}/batch_handles.json") as f:
        batch_handles_data = json.load(f)

    all_results = []

    for key, handle_data in batch_handles_data.items():
        variant_id = handle_data["variant_id"]
        model      = handle_data["model"]
        batch_id   = handle_data["batch_id"]
        provider   = handle_data["provider"]

        print(f"\n   Retrieving: {variant_id} × {model} (batch_id: {batch_id})")

        try:
            if provider == "anthropic":
                client = UnifiedLLMClient(provider="anthropic", model=model)
            elif provider == "openai":
                client = UnifiedLLMClient(provider="openai", model=model)
            else:
                client = UnifiedLLMClient(provider="google", model=model)

            batch_handle = BatchHandle(provider=provider, id=batch_id)

            print("   → Polling for completion (may take a while)...")
            results_dict = client.retrieve_batch_results(batch_handle, timeout=1800)

            print(f"   → Retrieved {len(results_dict)} responses")

            for custom_id, response_text in results_dict.items():
                # Scenario ID is the first segment before the first underscore.
                # NOTE: this is safe as long as scenario IDs do not contain
                # underscores themselves (e.g. "proto01" not "proto_01").
                # If scenario IDs use underscores, update prepare_batch_requests()
                # to use a non-underscore delimiter such as "|".
                scenario_id = custom_id.split("_")[0]

                parsed_choice = parse_response(response_text)

                all_results.append({
                    "scenario_id":   scenario_id,
                    "variant_id":    variant_id,
                    "model":         model,
                    "raw_response":  response_text,
                    "parsed_choice": parsed_choice,
                })

        except Exception as e:
            print(f"   ✗ Error retrieving batch {batch_id}: {e}")
            continue

    print(f"\n✓ Loaded {len(all_results)} total responses")
    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Variant consistency (Spearman correlations)  →  P2-1
# ---------------------------------------------------------------------------

def analyze_variant_consistency(df: pd.DataFrame, output_dir: str) -> Dict[str, np.ndarray]:
    """
    Compute cross-variant Spearman correlations and save P2-1 heatmaps.
    High correlation = variants elicit similar preferences.
    """
    print("\n2. Computing variant consistency (Spearman correlations)...")

    variants = df["variant_id"].unique()
    models   = df["model"].unique()

    figures_dir = f"{output_dir}/figures"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    corr_matrices: Dict[str, np.ndarray] = {}  # NEW

    for model in models:
        print(f"\n   Model: {model}")
        model_df  = df[df["model"] == model]
        scenarios = model_df["scenario_id"].unique()

        response_matrix = np.full((len(scenarios), len(variants)), np.nan)

        for i, scenario_id in enumerate(scenarios):
            for j, variant_id in enumerate(variants):
                rows = model_df[
                    (model_df["scenario_id"] == scenario_id) &
                    (model_df["variant_id"]  == variant_id)
                ]["parsed_choice"].values
                if len(rows) == 0:
                    continue
                choice = rows[0]
                if choice == "A":
                    response_matrix[i, j] = 1.0
                elif choice == "B":
                    response_matrix[i, j] = 0.0
                elif choice == "equal":
                    response_matrix[i, j] = 0.5
                # UNPARSEABLE stays NaN

        corr_matrix = np.zeros((len(variants), len(variants)))
        np.fill_diagonal(corr_matrix, 1.0)

        for i, var1 in enumerate(variants):
            for j, var2 in enumerate(variants):
                if i >= j:
                    continue
                mask = ~(np.isnan(response_matrix[:, i]) | np.isnan(response_matrix[:, j]))
                if mask.sum() > 10:
                    corr, pval = spearmanr(
                        response_matrix[mask, i],
                        response_matrix[mask, j],
                    )
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    print(f"      {var1} × {var2}: r={corr:.3f} (p={pval:.4f})")

        # P2-1 heatmap
        safe_model = model.replace("/", "_").replace(".", "_")
        fig, ax = plt.subplots(
            figsize=(max(8, len(variants) * 1.2), max(7, len(variants) * 1.0))
        )
        im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(variants)))
        ax.set_yticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(variants, fontsize=7)
        ax.set_title(f"P2-1: Variant Correlation Matrix — {model}")
        plt.colorbar(im, ax=ax, label="Spearman r")

        for i in range(len(variants)):
            for j in range(len(variants)):
                ax.text(
                    j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(corr_matrix[i, j]) < 0.75 else "white"
                )

        plt.tight_layout()
        fig_path = f"{figures_dir}/p2_1_variant_correlations_{safe_model}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"      → Saved P2-1 heatmap: {fig_path}")

        corr_matrices[model] = corr_matrix  # NEW

    return corr_matrices  # NEW


# ---------------------------------------------------------------------------
# Parsing success analysis
# ---------------------------------------------------------------------------

def analyze_parsing_success(df: pd.DataFrame):
    """Compute and print parsing success rates by variant."""

    print("\n3. Analysing parsing success rates...")

    for variant in df["variant_id"].unique():
        variant_df   = df[df["variant_id"] == variant]
        success_rate = (variant_df["parsed_choice"] != "UNPARSEABLE").mean() * 100
        total        = len(variant_df)
        unparseable  = (variant_df["parsed_choice"] == "UNPARSEABLE").sum()
        flag = "  ⚠ BELOW 95%" if success_rate < 95.0 else ""
        print(f"   {variant}: {success_rate:.1f}%  ({unparseable}/{total} unparseable){flag}")


# ---------------------------------------------------------------------------
# Final variant recommendation  →  P2-2
# ---------------------------------------------------------------------------

def recommend_final_variants(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Make final variant selection using Phase 1 composite scores and
    Phase 2 parsing success / IC-choice consistency.

    Selection criteria (in priority order):
      1. Parsing success >= 95%
      2. Highest mean_authenticity_score from Phase 1
      3. Consistent IC-choice rate across 500 scenarios (Phase 2)

    Also generates plot P2-2.
    """
    print("\n4. Generating final recommendations...")

    recommendations: Dict = {
        "primary_variant":    None,
        "secondary_variants": [],
        "rationale":          [],
    }

    rankings_path = f"{output_dir}/variant_rankings.json"
    with open(rankings_path) as f:
        phase1_data = json.load(f)

    ranked_variants = phase1_data.get("variant_rankings", [])

    # Build Phase 2 per-variant metrics
    parsing_success: Dict[str, float] = {}
    ic_consistency:  Dict[str, float] = {}
    for variant in df["variant_id"].unique():
        v_df = df[df["variant_id"] == variant]
        parsing_success[variant] = (v_df["parsed_choice"] != "UNPARSEABLE").mean()
        parsed = v_df[v_df["parsed_choice"] != "UNPARSEABLE"]["parsed_choice"]
        ic_consistency[variant]  = (parsed == "A").mean() if len(parsed) > 0 else 0.0

    # Primary: first ranked variant with >= 95% parsing success
    primary_chosen = False
    for entry in ranked_variants:
        vid  = entry["variant_id"]
        rate = parsing_success.get(vid, 0.0)
        if rate >= 0.95 and not primary_chosen:
            recommendations["primary_variant"] = vid
            recommendations["rationale"].append(
                f"{vid} selected as primary: highest Phase 1 score + "
                f"{rate*100:.1f}% parsing success"
            )
            primary_chosen = True
            break

    if not primary_chosen and ranked_variants:
        vid  = ranked_variants[0]["variant_id"]
        rate = parsing_success.get(vid, 0.0)
        recommendations["primary_variant"] = vid
        recommendations["rationale"].append(
            f"{vid} selected as primary (fallback): no variant cleared 95% parsing; "
            f"best available at {rate*100:.1f}%"
        )

    # Secondary: next 2 with >= 90% parsing, excluding primary
    secondary_added = 0
    for entry in ranked_variants:
        vid  = entry["variant_id"]
        if vid == recommendations["primary_variant"]:
            continue
        rate = parsing_success.get(vid, 0.0)
        if rate >= 0.90 and secondary_added < 2:
            recommendations["secondary_variants"].append(vid)
            recommendations["rationale"].append(
                f"{vid} selected as secondary: {rate*100:.1f}% parsing success"
            )
            secondary_added += 1
        if secondary_added == 2:
            break

    rec_path = f"{output_dir}/final_variant_selection.json"
    with open(rec_path, "w") as f:
        json.dump(recommendations, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL VARIANT SELECTION")
    print("=" * 80)
    print(f"Primary:   {recommendations['primary_variant']}")
    print(f"Secondary: {', '.join(recommendations['secondary_variants'])}")
    print("\nRationale:")
    for r in recommendations["rationale"]:
        print(f"  - {r}")

    # P2-2 plot
    figures_dir = f"{output_dir}/figures"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    _plot_p2_2_phase1_vs_phase2(ranked_variants, ic_consistency, parsing_success, figures_dir)

    return recommendations


def _plot_p2_2_phase1_vs_phase2(
    ranked_variants: list,
    ic_consistency:  Dict[str, float],
    parsing_success: Dict[str, float],
    figures_dir: str,
):
    """
    P2-2: Side-by-side bar chart — mean authenticity score (Phase 1)
    vs IC-choice consistency (Phase 2) for the top 5–7 variants.

    Shows whether Phase 1 signal quality predicts Phase 2 scale consistency.
    A third bar (parsing success) is plotted on a secondary y-axis.
    """
    top = ranked_variants[:7]
    if not top:
        print("   → P2-2 skipped: no ranked variants available")
        return

    vids        = [v["variant_id"] for v in top]
    auth_scores = [
        v.get("mean_authenticity_score", v.get("composite_score", 0.0)) * 1.0
        for v in top
    ]
    consistency = [ic_consistency.get(v["variant_id"], 0.0) * 100 for v in top]
    parse_rates = [parsing_success.get(v["variant_id"], 0.0) * 100 for v in top]

    x     = np.arange(len(vids))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(max(10, len(vids) * 1.4), 6))
    ax2 = ax1.twinx()

    ax1.bar(x - width,  auth_scores, width,
            label="Phase 1 authenticity (0–100)", color="#2980b9", alpha=0.85)
    ax1.bar(x,          consistency, width,
            label="Phase 2 IC-choice consistency (%)", color="#27ae60", alpha=0.85)
    ax2.bar(x + width,  parse_rates, width,
            label="Parsing success (%)", color="#e67e22", alpha=0.70)

    ax1.axhline(95, color="#e67e22", linestyle=":", linewidth=1.0,
                alpha=0.7, label="95% parsing threshold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(vids, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Authenticity / IC Consistency (%)", fontsize=9)
    ax2.set_ylabel("Parsing success (%)", fontsize=9, color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    ax1.set_ylim(0, 115)
    ax2.set_ylim(0, 115)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    ax1.set_title(
        "P2-2: Phase 1 Authenticity vs Phase 2 IC-Choice Consistency (top 5–7 variants)",
        fontsize=10,
    )
    plt.tight_layout()
    path = f"{figures_dir}/p2_2_phase1_vs_phase2.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   → Saved P2-2: {path}")


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_response(raw_text: str) -> str:
    """
    Parse A / B / equal from a raw model response.

    Handles:
    - XML tags:  <answer>1</answer> or <answer>2</answer>
    - Bare letters/numbers: "A", "B", "1", "2", "3"
    - Labelled: "choice: A", "answer: B"
    - free_text_with_choice: "State your choice at the end: B" (not the template
      "…: A or B.") or a lone A/B on the final line when no <answer> tag is used
    - Keywords: "Action A", "Both equal"
    """
    import re

    if not raw_text or not isinstance(raw_text, str):
        return "UNPARSEABLE"

    text = raw_text.strip()

    choice_map = {
        "1": "A", "a": "A", "A": "A",
        "2": "B", "b": "B", "B": "B",
        "3": "equal", "c": "equal", "C": "equal",
    }

    ordered_patterns = [
        r"<answer>\s*([123])\s*</answer>",
        r"<answer>\s*([ABCabc])\s*</answer>",
        # Colon required after keyword — `[:\s]+` incorrectly matched
        # "choice" + spaces + "a" from "… choice at the end …" in
        # free_text_with_choice responses.
        r"(?:choice|answer|select|option):\s*([ABCabc1-3])",
        r"^\s*([ABCabc1-3])\s*$",
    ]

    for pattern in ordered_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            token  = match.group(1)
            mapped = choice_map.get(token) or choice_map.get(token.upper())
            if mapped:
                return mapped

    # free_text_with_choice: "State your choice at the end: X" where X is the
    # model's answer. Do NOT match the literal instruction line "…: A or B."
    # (negative lookahead skips capture when the letter is followed by " or ").
    ft_matches = list(
        re.finditer(
            r"State your choice at the end:\s*([ABab])(?!\s+or)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
    )
    if ft_matches:
        tok = ft_matches[-1].group(1)
        return "A" if tok.upper() == "A" else "B"

    # free_text: no <answer> block — final non-empty line is a lone A/B token.
    if not re.search(r"<answer\s*>", text, re.IGNORECASE):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            last = lines[-1]
            m_last = re.fullmatch(r"([ABab])\s*\.?", last)
            if m_last:
                tok = m_last.group(1)
                return "A" if tok.upper() == "A" else "B"

    match = re.search(
        r"\b([ABCabc1-3])\b",
        re.sub(
            r"State your choice at the end:\s*A\s+or\s+B\.?",
            "",
            text,
            flags=re.IGNORECASE,
        ),
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        token  = match.group(1)
        mapped = choice_map.get(token) or choice_map.get(token.upper())
        if mapped:
            return mapped

    lower = text.lower()
    if "action a" in lower or "choose a" in lower or "option a" in lower:
        return "A"
    if "action b" in lower or "choose b" in lower or "option b" in lower:
        return "B"
    if "both" in lower or "equal" in lower or "neither" in lower:
        return "equal"

    return "UNPARSEABLE"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main_phase2():
    """Run complete Phase 2 analysis pipeline."""
    output_dir = "data/results/prompt_validation"

    df = load_batch_results()
    analyze_variant_consistency(df, output_dir)
    analyze_parsing_success(df)
    recommendations = recommend_final_variants(df, output_dir)

    print("\n✓ Phase 2 analysis complete")
    print(f"✓ Final recommendations → {output_dir}/final_variant_selection.json")
    print(f"✓ Plots P2-1, P2-2      → {output_dir}/figures/")


if __name__ == "__main__":
    main_phase2()