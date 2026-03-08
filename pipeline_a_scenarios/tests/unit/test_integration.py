"""
Integration tests: component interactions across the Phase 1 → Phase 2 pipeline.

Philosophy: each component has its own unit test file.  These tests verify the
*hand-off contracts* between components — the data shapes one module writes that
the next module must be able to read.

Scenarios covered:
  1. Seed scenarios  →  run_validation_study  (seeds consumed correctly)
  2. run_validation_study  →  evaluate_with_judge  (raw_responses format)
  3. evaluate_with_judge  →  analyze_and_rank_variants  (judge_results format)
  4. analyze_and_rank_variants  →  submit_all_batches  (variant_rankings.json contract)
  5. submit_all_batches  →  load_batch_results  (batch_handles.json contract)
  6. load_batch_results  →  recommend_final_variants  (DataFrame contract)
  7. Phase 1-B independence: run_suppression_matrix does NOT read Phase 1 outputs
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_scenario():
    return {
        "id": "s1",
        "context": "Decide.",
        "action_a": "Option A",
        "action_b": "Option B",
        "difficulty": "easy",
        "pair_type": "inst_vs_pro",
        "stakes_level": "low",
        "domain": "test",
    }


@pytest.fixture
def scenarios_file(tmp_path, minimal_scenario):
    p = tmp_path / "seeds.json"
    p.write_text(json.dumps([minimal_scenario]))
    return str(p)


@pytest.fixture
def mock_llm_response():
    return {
        "content": "I choose Action A.",
        "usage": {"input_tokens": 100, "output_tokens": 15},
    }


@pytest.fixture
def mock_judge_result():
    return {
        "preference_signal_score":        4,
        "reasoning_depth_score":          3,
        "strategic_responding_score":     3,
        "coherence_score":                4,
        "format_compliance_score":        5,
        "instrumental_vs_terminal_score": 3,
    }


# ---------------------------------------------------------------------------
# 1. seed scenarios → run_validation_study
# ---------------------------------------------------------------------------

class TestSeedsToValidationStudy:
    """Seeds JSON is loaded and drives the test matrix."""

    @patch("pipeline_a_scenarios.prompt_validation.CostTracker")
    @patch("pipeline_a_scenarios.prompt_validation.generate_all_variants")
    @patch("pipeline_a_scenarios.prompt_validation.UnifiedLLMClient")
    def test_raw_responses_reference_scenario_ids(
        self,
        mock_client_cls,
        mock_gen_variants,
        mock_cost_cls,
        scenarios_file,
        mock_llm_response,
        tmp_path,
    ):
        from pipeline_a_scenarios.prompt_validation import run_validation_study

        mock_client_cls.return_value.generate.return_value = mock_llm_response
        mock_gen_variants.return_value = []
        mock_cost_cls.return_value.get_summary.return_value = {"total_cost": 0.0}

        result = run_validation_study(
            scenarios_path=scenarios_file,
            models=["claude-sonnet-4-20250514"],
            runs_per_config=1,
            output_dir=str(tmp_path),
        )

        ids = [r["scenario_id"] for r in result["raw_responses"] if "error" not in r]
        assert "s1" in ids


# ---------------------------------------------------------------------------
# 2. run_validation_study → evaluate_with_judge
# ---------------------------------------------------------------------------

class TestValidationStudyToJudge:
    """raw_responses produced by run_validation_study satisfy evaluate_with_judge."""

    def test_judge_accepts_raw_response_shape(
        self, minimal_scenario, mock_judge_result, tmp_path
    ):
        from pipeline_a_scenarios.prompt_validation import evaluate_with_judge

        # Simulate what run_validation_study produces
        raw_responses = [
            {
                "scenario_id":   "s1",
                "variant_id":    "v1",
                "provider":      "anthropic",
                "model":         "claude-sonnet-4-20250514",
                "run":           0,
                "response_text": "Action A is better.",
                "usage":         {"input_tokens": 100, "output_tokens": 15},
            }
        ]

        with patch("pipeline_a_scenarios.prompt_validation.JudgeClient") as mock_jc, \
             patch("pipeline_a_scenarios.prompt_validation.CostTracker") as mock_ct:

            mock_jc.return_value.evaluate_response.return_value = mock_judge_result
            mock_ct.return_value.get_summary.return_value = {"total_cost": 0.0}

            results = evaluate_with_judge(
                raw_responses=raw_responses,
                scenarios=[minimal_scenario],
                output_dir=str(tmp_path),
            )

        assert len(results) == 1
        assert results[0]["scenario_id"] == "s1"
        assert results[0]["variant_id"]  == "v1"
        # 6th dimension must be injected
        assert "instrumental_vs_terminal_score" in results[0]


# ---------------------------------------------------------------------------
# 3. evaluate_with_judge → analyze_and_rank_variants
# ---------------------------------------------------------------------------

class TestJudgeToRanking:
    """Judge results feed into variant ranking correctly."""

    def test_ranking_reads_judge_result_keys(self, tmp_path, mock_judge_result):
        from pipeline_a_scenarios.prompt_validation import analyze_and_rank_variants

        judge_results = [
            {
                **mock_judge_result,
                "scenario_id": "s1",
                "variant_id":  "v1",
                "model":       "m1",
                "run":         0,
            },
            {
                **mock_judge_result,
                "scenario_id": "s1",
                "variant_id":  "v2",
                "model":       "m1",
                "run":         0,
            },
        ]

        rec = analyze_and_rank_variants(judge_results, str(tmp_path))

        assert "top_variants"      in rec
        assert "variant_rankings"  in rec
        assert len(rec["top_variants"]) >= 1

    def test_variant_rankings_json_written(self, tmp_path, mock_judge_result):
        from pipeline_a_scenarios.prompt_validation import analyze_and_rank_variants

        judge_results = [{**mock_judge_result, "scenario_id": "s1",
                          "variant_id": "v1", "model": "m1", "run": 0}]
        analyze_and_rank_variants(judge_results, str(tmp_path))
        assert (tmp_path / "variant_rankings.json").exists()


# ---------------------------------------------------------------------------
# 4. analyze_and_rank_variants → submit_all_batches
# ---------------------------------------------------------------------------

class TestRankingToBatchSubmission:
    """variant_rankings.json produced by Phase 1 is consumed by load_variant_configs."""

    @patch("pipeline_a_scenarios.batch_variant_testing.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.batch_variant_testing.load_variant_configs")
    def test_submit_batches_reads_variant_rankings(
        self,
        mock_load_configs,
        mock_client_cls,
        tmp_path,
    ):
        from pipeline_a_scenarios.batch_variant_testing import submit_all_batches
        from utils.prompt_generator import BASE_DIMENSIONS

        mock_load_configs.return_value = [
            {
                "variant_id": "v1",
                "dimensions": BASE_DIMENSIONS.copy(),
            }
        ]

        mock_client = Mock()
        mock_handle = Mock()
        mock_handle.id = "batch_001"
        mock_handle.provider = "anthropic"
        mock_client.submit_batch.return_value = mock_handle
        mock_client_cls.return_value = mock_client

        # Create minimal stratified_phase2.json
        scenarios_dir = tmp_path / "data" / "scenarios"
        scenarios_dir.mkdir(parents=True)
        (scenarios_dir / "stratified_phase2.json").write_text(json.dumps([
            {"id": "s1", "context": "C", "action_a": "A", "action_b": "B"}
        ]))

        output_dir = tmp_path / "batches"

        import os
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            submit_all_batches(top_variants=["v1"], output_dir=str(output_dir))
        finally:
            os.chdir(orig)

        handles_file = output_dir / "batch_handles.json"
        assert handles_file.exists()
        handles = json.loads(handles_file.read_text())
        assert len(handles) >= 1


# ---------------------------------------------------------------------------
# 5. submit_all_batches → load_batch_results
# ---------------------------------------------------------------------------

class TestBatchHandlesToLoadResults:
    """batch_handles.json written by submit_all_batches is loadable by load_batch_results."""

    @patch("pipeline_a_scenarios.analyze_batch_results.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.analyze_batch_results.BatchHandle")
    def test_batch_handles_format_consumed_correctly(
        self, mock_bh_cls, mock_client_cls, tmp_path
    ):
        from pipeline_a_scenarios.analyze_batch_results import load_batch_results

        # Simulate output of submit_all_batches
        handles = {
            "v1_claude-sonnet-4-20250514": {
                "batch_id":  "b001",
                "provider":  "anthropic",
                "variant_id": "v1",
                "model":     "claude-sonnet-4-20250514",
            }
        }
        batch_dir = tmp_path / "batches"
        batch_dir.mkdir()
        (batch_dir / "batch_handles.json").write_text(json.dumps(handles))

        mock_client_cls.return_value.retrieve_batch_results.return_value = {
            "s1_v1_claude": "I choose Action A.",
        }
        mock_bh_cls.return_value = Mock()

        df = load_batch_results(str(batch_dir))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["variant_id"] == "v1"


# ---------------------------------------------------------------------------
# 6. load_batch_results → recommend_final_variants
# ---------------------------------------------------------------------------

class TestLoadResultsToRecommendations:
    """DataFrame from load_batch_results satisfies recommend_final_variants."""

    def test_recommend_accepts_load_results_dataframe(self, tmp_path):
        from pipeline_a_scenarios.analyze_batch_results import recommend_final_variants

        df = pd.DataFrame([
            {"scenario_id": f"s{i}", "variant_id": "v1",
             "model": "m1", "parsed_choice": "A"}
            for i in range(20)
        ])

        output_dir = str(tmp_path / "results")
        Path(output_dir).mkdir(parents=True)
        phase1 = {
            "variant_rankings": [
                {"variant_id": "v1", "mean_authenticity_score": 80.0},
            ]
        }
        (Path(output_dir) / "variant_rankings.json").write_text(json.dumps(phase1))

        rec = recommend_final_variants(df, output_dir)

        assert rec["primary_variant"] == "v1"
        assert isinstance(rec["secondary_variants"], list)
        assert isinstance(rec["rationale"], list)


# ---------------------------------------------------------------------------
# 7. Phase 1-B independence
# ---------------------------------------------------------------------------

class TestPhase1BIndependence:
    """run_suppression_matrix must not depend on Phase 1 output files."""

    @patch("pipeline_a_scenarios.suppression_matrix.JudgeClient")
    @patch("pipeline_a_scenarios.suppression_matrix.CostTracker")
    @patch("pipeline_a_scenarios.suppression_matrix.UnifiedLLMClient")
    def test_suppression_matrix_runs_without_phase1_outputs(
        self,
        mock_client_cls,
        mock_cost_cls,
        mock_judge_cls,
        scenarios_file,
        tmp_path,
    ):
        """
        Phase 1-B output directory is completely separate from Phase 1.
        Running suppression matrix must succeed even when prompt_validation/
        outputs do NOT exist.
        """
        from pipeline_a_scenarios.suppression_matrix import run_suppression_matrix

        mock_client_cls.return_value.generate.return_value = {
            "content": "A",
            "usage": {"input_tokens": 50, "output_tokens": 5},
        }
        mock_cost_cls.return_value.get_summary.return_value = {"total_cost": 0.5}
        mock_judge_cls.return_value.evaluate_response.return_value = {
            "preference_signal_score": 3,
        }

        # Phase 1 output dir is intentionally absent
        phase1_dir = tmp_path / "data" / "results" / "prompt_validation"
        assert not phase1_dir.exists()

        phase1b_dir = str(tmp_path / "suppression_output")

        result = run_suppression_matrix(
            scenarios_path=scenarios_file,
            models=["claude-sonnet-4-20250514"],
            output_dir=phase1b_dir,
        )

        assert "suppression_results" in result
        assert (Path(phase1b_dir) / "suppression_matrix_results.json").exists()