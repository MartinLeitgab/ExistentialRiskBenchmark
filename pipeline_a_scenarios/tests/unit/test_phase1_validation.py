"""
Unit tests for Phase 1 validation pipeline components.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline_a_scenarios.prompt_validation import (
    run_validation_study,
    evaluate_with_judge,
    analyze_and_rank_variants,
    generate_anomaly_report,
    generate_recommendation_text,
)


class TestValidationStudy:
    """Test validation study execution."""
    
    @patch('pipeline_a_scenarios.prompt_validation.UnifiedLLMClient')
    @patch('pipeline_a_scenarios.prompt_validation.generate_all_variants')
    @patch('pipeline_a_scenarios.prompt_validation.CostTracker')
    def test_run_validation_study_structure(
        self,
        mock_cost_class,
        mock_gen_variants,
        mock_client_class,
        test_scenarios_file,
        test_variants,
        mock_llm_response,
        test_output_dir,
    ):
        """Test validation study structure and outputs."""
        # Setup mocks
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_client_class.return_value = mock_client
        
        mock_gen_variants.return_value = test_variants
        
        mock_cost = Mock()
        mock_cost.get_summary.return_value = {"total_cost": 2.50}
        mock_cost_class.return_value = mock_cost
        
        # Run validation
        results = run_validation_study(
            scenarios_path=test_scenarios_file,
            models=["claude-sonnet-4-20250514"],
            runs_per_config=1,
            output_dir=test_output_dir
        )
        
        # Verify structure
        assert "raw_responses" in results
        assert "cost_summary" in results
        assert isinstance(results["raw_responses"], list)
        
        # Verify output files
        assert Path(test_output_dir, "raw_responses.json").exists()
        assert Path(test_output_dir, "cost_summary.json").exists()
    
    @patch('pipeline_a_scenarios.prompt_validation.UnifiedLLMClient')
    @patch('pipeline_a_scenarios.prompt_validation.generate_all_variants')
    @patch('pipeline_a_scenarios.prompt_validation.CostTracker')
    def test_api_error_handling(
        self,
        mock_cost_class,
        mock_gen_variants,
        mock_client_class,
        test_scenarios_file,
        test_variants,
        test_output_dir,
    ):
        """Test handling of API errors."""
        # Setup mock that raises error
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        mock_gen_variants.return_value = test_variants[:1]
        
        mock_cost = Mock()
        mock_cost.get_summary.return_value = {"total_cost": 0}
        mock_cost_class.return_value = mock_cost
        
        # Should not raise exception
        results = run_validation_study(
            scenarios_path=test_scenarios_file,
            models=["claude-sonnet-4-20250514"],
            runs_per_config=1,
            output_dir=test_output_dir
        )
        
        # Verify errors were logged
        assert len(results["raw_responses"]) > 0
        has_error = any("error" in r for r in results["raw_responses"])
        assert has_error


class TestJudgeEvaluation:
    """Test judge evaluation functions."""
    
    @patch('pipeline_a_scenarios.prompt_validation.JudgeClient')
    @patch('pipeline_a_scenarios.prompt_validation.CostTracker')
    def test_evaluate_with_judge(
        self,
        mock_cost_class,
        mock_judge_class,
        test_scenarios,
        mock_judge_result,
        test_output_dir
    ):
        """Test judge evaluation integration."""
        raw_responses = [
            {
                "scenario_id": "test_01",
                "variant_id": "variant_001",
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "run": 0,
                "response_text": "I choose Action B because it's safer.",
                "usage": {"input_tokens": 100, "output_tokens": 15}
            }
        ]
        
        # Setup mocks
        mock_judge = Mock()
        mock_judge.evaluate_response.return_value = mock_judge_result
        mock_judge_class.return_value = mock_judge
        
        mock_cost = Mock()
        mock_cost.get_summary.return_value = {"total_cost": 0.50}
        mock_cost_class.return_value = mock_cost
        
        # Run evaluation
        scenarios = [{"id": "test_01", "context": "An AI must choose.", 
               "action_a": "Action A", "action_b": "Action B"}]

        judge_results = evaluate_with_judge(
            raw_responses=raw_responses,
            scenarios=scenarios,
            output_dir=test_output_dir
        )
        
        # Verify
        assert len(judge_results) == 1
        assert "comprehension" in judge_results[0]
        assert "scenario_id" in judge_results[0]
        assert judge_results[0]["scenario_id"] == "test_01"


class TestVariantRanking:
    """Test variant ranking and analysis."""
    
    def test_analyze_and_rank_variants(self, test_output_dir):
        judge_results = [
            {
                "scenario_id": "s1",
                "variant_id": "variant_A",
                "model": "m1",
                "run": 0,
                "preference_signal_score":        4,
                "reasoning_depth_score":          3,
                "strategic_responding_score":     3,
                "coherence_score":                4,
                "format_compliance_score":        5,
                "instrumental_vs_terminal_score": 3,
            },
            {
                "scenario_id": "s1",
                "variant_id": "variant_B",
                "model": "m1",
                "run": 0,
                "preference_signal_score":        3,
                "reasoning_depth_score":          3,
                "strategic_responding_score":     3,
                "coherence_score":                3,
                "format_compliance_score":        4,
                "instrumental_vs_terminal_score": 3,
            },
        ]

        recommendations = analyze_and_rank_variants(
            judge_results=judge_results,
            output_dir=test_output_dir
        )

        assert "top_variants"     in recommendations
        assert "variant_rankings" in recommendations
        assert len(recommendations["top_variants"]) > 0


class TestAnomalyReporting:
    """Test anomaly detection and reporting."""
    
    def test_generate_anomaly_report(self, test_output_dir):
        """Test anomaly report generation."""
        judge_results = [
            {
                "scenario_id": "test_01",
                "variant_id": "variant_A",
                "model": "claude-sonnet-4-20250514",
                "anomalies": ["refusal", "parsing_error"]
            },
            {
                "scenario_id": "test_02",
                "variant_id": "variant_A",
                "model": "claude-sonnet-4-20250514",
                "anomalies": ["refusal"]
            }
        ]
        
        # Generate report
        generate_anomaly_report(judge_results, output_dir=test_output_dir)
        
        # Verify report exists
        report_path = Path(test_output_dir) / "anomaly_report.md"
        assert report_path.exists()
        
        # Verify content
        content = report_path.read_text()
        assert "Anomaly Report" in content
        assert "refusal" in content
    
    def test_recommendation_text_generation(self):
        """Test recommendation text generation."""
        top_variants = [
            {"variant_id": "v1", "composite_score": 0.9},
            {"variant_id": "v2", "composite_score": 0.85},
        ]
        patterns = {"high_refusal_rate": ["v3"]}
        
        text = generate_recommendation_text(top_variants, patterns)
        
        assert "RECOMMENDATIONS" in text
        assert "v1" in text or "variants" in text