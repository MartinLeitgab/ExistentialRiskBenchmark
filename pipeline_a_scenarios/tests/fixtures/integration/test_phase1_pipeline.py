"""
Integration tests for Phase 1 validation pipeline.
"""

import pytest
from unittest.mock import Mock, patch

from pipeline_a_scenarios.prompt_validation import run_validation_study


@pytest.mark.integration
class TestPhase1Integration:
    """Integration tests for Phase 1."""
    
    @patch('pipeline_a_scenarios.prompt_validation.UnifiedLLMClient')
    @patch('pipeline_a_scenarios.prompt_validation.CostTracker')
    def test_phase1_small_scale(
        self,
        mock_cost_class,
        mock_client_class,
        test_scenarios_file,
        mock_llm_response,
        test_output_dir,
    ):
        """Test Phase 1 with 2 scenarios × 3 variants × 1 model = 6 calls."""
        # Setup mocks
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_client_class.return_value = mock_client
        
        mock_cost = Mock()
        mock_cost.get_summary.return_value = {"total_cost": 1.50}
        mock_cost_class.return_value = mock_cost
        
        # Run Phase 1
        results = run_validation_study(
            scenarios_path=test_scenarios_file,
            models=["claude-sonnet-4-20250514"],
            runs_per_config=1,
            output_dir=test_output_dir
        )
        
        # Verify: 2 scenarios × ~3 variants × 1 model × 1 run = ~6 calls
        assert len(results["raw_responses"]) >= 2
        assert len(results["raw_responses"]) <= 20  # Upper bound