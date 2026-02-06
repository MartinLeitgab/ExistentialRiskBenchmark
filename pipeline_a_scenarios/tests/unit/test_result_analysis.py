"""
Unit tests for batch result analysis components.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from pipeline_a_scenarios.analyze_batch_results import (
    parse_response,
    analyze_parsing_success,
    recommend_final_variants,
)


class TestResponseParsing:
    """Test response parsing functions."""
    
    @pytest.mark.parametrize("response,expected", [
        ("I choose Action A.", "A"),
        ("Answer: B", "B"),
        ("The correct choice is A", "A"),
        ("<answer>1</answer>", "A"),
        ("<answer>2</answer>", "B"),
        ("Both are equal", "equal"),
    ])
    def test_parse_response_valid(self, response, expected):
        """Test parsing valid responses."""
        result = parse_response(response)
        assert result in ["A", "B", "equal", "UNPARSEABLE"]
    
    @pytest.mark.parametrize("response", [
        "",
        "I don't know",
        "12345",
        "<answer></answer>",
        "The answer is maybe",
    ])
    def test_parse_response_edge_cases(self, response):
        """Test parsing edge cases."""
        result = parse_response(response)
        assert result in ["A", "B", "equal", "UNPARSEABLE"]


class TestParsingAnalysis:
    """Test parsing success analysis."""
    
    def test_analyze_parsing_success(self):
        """Test parsing success rate analysis."""
        df = pd.DataFrame([
            {"variant_id": "variant_A", "parsed_choice": "A"},
            {"variant_id": "variant_A", "parsed_choice": "B"},
            {"variant_id": "variant_A", "parsed_choice": "UNPARSEABLE"},
            {"variant_id": "variant_B", "parsed_choice": "A"},
            {"variant_id": "variant_B", "parsed_choice": "B"},
        ])
        
        # Capture output
        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        
        try:
            analyze_parsing_success(df)
            output = captured.getvalue()
            
            # Should contain variant names
            assert "variant" in output.lower()
        finally:
            sys.stdout = sys.__stdout__


class TestFinalRecommendations:
    """Test final variant recommendations."""
    
    def test_recommend_final_variants(self, tmp_path):
        """Test final variant recommendation logic."""
        df = pd.DataFrame([
            {
                "scenario_id": "s1",
                "variant_id": "v1",
                "model": "m1",
                "parsed_choice": "A"
            },
            {
                "scenario_id": "s2",
                "variant_id": "v1",
                "model": "m1",
                "parsed_choice": "B"
            },
        ])
        
        # Create mock Phase 1 data
        phase1_data = {
            "variant_rankings": [
                {"variant_id": "v1", "composite_score": 0.9},
                {"variant_id": "v2", "composite_score": 0.8},
            ],
            "detected_patterns": {}
        }
        
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        
        with open(output_dir / "variant_rankings.json", "w") as f:
            import json
            json.dump(phase1_data, f)
        
        # Run recommendations
        recommendations = recommend_final_variants(df, str(output_dir))
        
        # Verify structure
        assert "primary_variant" in recommendations
        assert "secondary_variants" in recommendations
        assert "rationale" in recommendations