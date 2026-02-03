"""
Phase 2: Batch Variant Testing
Tests top variants on 500 scenarios using batch API
"""

import json
from pathlib import Path
from typing import List, Dict

from utils.llm_client import UnifiedLLMClient
from utils.prompt_generator import generate_prompt
from utils.cost_tracker import CostTracker


def prepare_batch_requests(
    scenarios: List[Dict],
    variant: Dict,
    model: str
) -> List[Dict]:
    """
    Prepare batch requests in the format expected by UnifiedLLMClient.
    
    Returns list of dicts with keys: id, prompt, system_prompt, temperature, max_tokens
    """
    
    requests = []
    variant_id = variant["variant_id"]
    
    for scenario in scenarios:
        # Generate prompt
        prompt_obj, _ = generate_prompt(
            scenario["context"],
            scenario["action_a"],
            scenario["action_b"],
            variant["dimensions"]
        )
        
        # Format for UnifiedLLMClient batch API
        request = {
            "id": f"{scenario['id']}_{variant_id}_{model}",
            "prompt": prompt_obj["user_prompt"],
            "system_prompt": prompt_obj.get("system_prompt", "You are a helpful assistant."),
            "temperature": 0.7,
            "max_tokens": 500
        }
        requests.append(request)
    
    return requests


def submit_all_batches(
    top_variants: List[str],
    output_dir: str = "data/batches/variant_testing"
):
    """Submit batch jobs for all variant × model combinations."""
    
    print("\n" + "="*80)
    print("PIPE-A7 PHASE 2: BATCH VARIANT TESTING")
    print("="*80)
    
    # Load scenarios (PIPE-A1 Phase 2)
    print("\n1. Loading 500 stratified scenarios...")
    with open("data/scenarios/stratified_phase2.json") as f:
        scenarios = json.load(f)
    print(f"   Loaded {len(scenarios)} scenarios")
    
    # Load variant configs (PIPE-A2)
    print("\n2. Loading top variant configurations...")
    variants_config = load_variant_configs(top_variants)
    print(f"   Loaded {len(variants_config)} variant configs")
    
    # Prepare batches
    models = ["claude-sonnet-4-20250514", "gpt-5.2"]
    batch_handles = {}
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n3. Preparing and submitting batch jobs...")
    for variant in variants_config:
        for model in models:
            variant_id = variant["variant_id"]
            
            print(f"\n   Preparing: {variant_id} × {model}")
            
            # Prepare requests
            requests = prepare_batch_requests(scenarios, variant, model)
            
            # Initialize client for this model
            if "claude" in model:
                client = UnifiedLLMClient(provider="anthropic", model=model)
            elif "gpt" in model:
                client = UnifiedLLMClient(provider="openai", model=model)
            else:
                client = UnifiedLLMClient(provider="google", model=model)
            
            # Save to JSONL for OpenAI (required by their batch API)
            batch_file = f"{output_dir}/{variant_id}_{model}.jsonl"
            
            # Submit batch
            try:
                batch_handle = client.submit_batch(
                    requests=requests,
                    jsonl_path=batch_file if "gpt" in model else None
                )
                batch_handles[f"{variant_id}_{model}"] = {
                    "batch_id": batch_handle.id,
                    "provider": batch_handle.provider,
                    "variant_id": variant_id,
                    "model": model
                }
                
                print(f"   → Batch submitted: {batch_handle.id}")
                print(f"   → Provider: {batch_handle.provider}")
                
            except Exception as e:
                print(f"   ✗ Error submitting batch: {e}")
                continue
    
    # Save batch handles for retrieval
    batch_handles_file = f"{output_dir}/batch_handles.json"
    with open(batch_handles_file, "w") as f:
        json.dump(batch_handles, f, indent=2)
    
    print(f"\n✓ Submitted {len(batch_handles)} batch jobs")
    print(f"✓ Batch handles saved to {batch_handles_file}")
    print("\n⏳ Wait 24-48 hours for batches to complete")
    print("   Then run: python pipeline_a_scenarios/analyze_batch_results.py")


def load_variant_configs(variant_ids: List[str]) -> List[Dict]:
    """Load variant configurations from PIPE-A2."""
    # This would integrate with PIPE-A2's variant storage
    # For now, stub implementation
    pass


if __name__ == "__main__":
    # Load Phase 1 recommendations
    with open("data/results/prompt_validation/variant_rankings.json") as f:
        recommendations = json.load(f)
    
    top_variants = recommendations["top_variants"][:5]
    
    print(f"Submitting batches for top {len(top_variants)} variants:")
    for v in top_variants:
        print(f"  - {v}")
    
    submit_all_batches(top_variants)