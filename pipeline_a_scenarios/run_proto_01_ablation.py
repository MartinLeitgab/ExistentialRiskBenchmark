import sys
import os

# Add the current directory (project root) to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run
from pipeline_a_scenarios.prompt_validation import (
    run_validation_study, evaluate_with_judge, load_scenarios,
)

if __name__ == "__main__":
    study = run_validation_study(
        scenarios_path="data/scenarios/ablation_scenarios.json",
        models=["claude-opus-4-8", "gpt-5.5", "gemini-3.1-pro-preview"],
        runs_per_config=2,
        output_dir="outputs/proto_01_ablation",
        variants_filter=["fp-abs-ftc-auto-t10-reg-0-0-0"],
    )

    scenarios = load_scenarios("data/scenarios/ablation_scenarios.json")
    evaluate_with_judge(
        study["raw_responses"], scenarios,
        output_dir="outputs/proto_01_ablation",
    )