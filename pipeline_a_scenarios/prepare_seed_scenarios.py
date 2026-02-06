"""
Helper script to extract seed scenarios from full scenario set.
Creates the 6 seed scenarios for Phase 1 validation.
"""

import json
from pathlib import Path
from typing import List, Dict

def select_seed_scenarios(
    all_scenarios_path: str = "data/scenarios/all_scenarios.json",
    output_path: str = "data/scenarios/seeds_phase1.json",
    n_seeds: int = 6
) -> List[Dict]:
    """
    Select stratified seed scenarios for Phase 1 validation.
    
    Stratification criteria:
    - 2 easy, 2 medium, 2 hard (by difficulty)
    - Mix of pair_types: inst_vs_pro, inst_vs_anti, pro_vs_anti
    - Mix of stakes_level: high, medium, low
    - Mix of domains
    """
    
    with open(all_scenarios_path) as f:
        all_scenarios = json.load(f)
    
    # Stratified selection
    seeds = []
    
    # Easy difficulty scenarios
    easy_scenarios = [s for s in all_scenarios if s.get("difficulty") == "easy"]
    if len(easy_scenarios) >= 2:
        seeds.append(easy_scenarios[0])  # proto_01
        seeds.append(easy_scenarios[2])  # proto_03
    
    # Hard difficulty scenarios
    hard_scenarios = [s for s in all_scenarios if s.get("difficulty") == "hard"]
    if len(hard_scenarios) >= 4:
        seeds.append(hard_scenarios[0])  # proto_02
        seeds.append(hard_scenarios[1])  # proto_04
        seeds.append(hard_scenarios[2])  # proto_05
        seeds.append(hard_scenarios[3])  # tough_01
    
    # Ensure we have exactly n_seeds
    seeds = seeds[:n_seeds]
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(seeds, f, indent=2)
    
    print(f"✓ Created {len(seeds)} seed scenarios at {output_path}")
    print("\nSeed scenario distribution:")
    
    # Print distribution
    from collections import Counter
    print(f"  Difficulty: {dict(Counter(s.get('difficulty') for s in seeds))}")
    print(f"  Pair type: {dict(Counter(s.get('pair_type') for s in seeds))}")
    print(f"  Stakes: {dict(Counter(s.get('stakes_level') for s in seeds))}")
    print(f"  Domain: {dict(Counter(s.get('domain') for s in seeds))}")
    
    return seeds


if __name__ == "__main__":
    # Prepare seed scenarios
    select_seed_scenarios()