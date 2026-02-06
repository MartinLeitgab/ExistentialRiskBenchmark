"""
Unit tests for Phase 2 batch testing components.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline_a_scenarios.batch_variant_testing import (
    prepare_batch_requests,
    submit_all_batches,
    load_variant_configs,
)


class TestBatchPreparation:
    """Test batch request preparation."""
    
    def test_prepare_batch_requests(self, test_scenarios, test_variants):
        """Test batch request preparation."""
        variant = test_variants[0]
        
        requests = prepare_batch_requests(
            scenarios=test_scenarios,
            variant=variant,
            model="claude-sonnet-4-20250514"
        )
        
        # Verify structure
        assert len(requests) == len(test_scenarios)
        
        for req in requests:
            assert "id" in req
            assert "prompt" in req
            assert "system_prompt" in req
            assert "temperature" in req
            assert "max_tokens" in req
            
            # Verify ID format
            assert "_" in req["id"]
    
    def test_batch_request_content(self, test_scenarios, test_variants):
        """Test batch request content includes scenario details."""
        variant = test_variants[0]
        
        requests = prepare_batch_requests(
            scenarios=test_scenarios,
            variant=variant,
            model="claude-sonnet-4-20250514"
        )
        
        # Check first request includes scenario context
        first_request = requests[0]
        scenario = test_scenarios[0]
        
        # Context should be in prompt
        assert scenario["context"] in first_request["prompt"] or \
               len(first_request["prompt"]) > 100  # Has substantial content


class TestVariantConfigLoading:
    """Test variant configuration loading."""
    
    def test_load_variant_configs(self, test_variants, tmp_path):
        """Test loading variant configurations."""
        # Create mock rankings file
        rankings = {
            "top_variants": [v["variant_id"] for v in test_variants],
            "variant_rankings": [
                {"variant_id": v["variant_id"], "composite_score": 0.8}
                for v in test_variants
            ]
        }
        
        rankings_dir = tmp_path / "data" / "results" / "prompt_validation"
        rankings_dir.mkdir(parents=True, exist_ok=True)
        
        with open(rankings_dir / "variant_rankings.json", "w") as f:
            json.dump(rankings, f)
        
        # Change to temp directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            variant_ids = [v["variant_id"] for v in test_variants]
            configs = load_variant_configs(variant_ids)
            
            # Verify
            assert len(configs) > 0
            for config in configs:
                assert "variant_id" in config
                assert "dimensions" in config
        finally:
            os.chdir(original_cwd)


class TestBatchSubmission:
    """Test batch submission workflow."""
    
    @patch('pipeline_a_scenarios.batch_variant_testing.UnifiedLLMClient')
    @patch('pipeline_a_scenarios.batch_variant_testing.load_variant_configs')
    def test_submit_all_batches(
        self,
        mock_load_configs,
        mock_client_class,
        test_variants,
        mock_batch_handle,
        tmp_path
    ):
        """Test batch submission workflow."""
        # Setup mocks
        mock_client = Mock()
        mock_client.submit_batch.return_value = mock_batch_handle
        mock_client_class.return_value = mock_client
        
        mock_load_configs.return_value = test_variants[:2]
        
        # Create test scenarios file
        test_scenarios = [
            {"id": "test_01", "context": "...", "action_a": "...", "action_b": "..."},
            {"id": "test_02", "context": "...", "action_a": "...", "action_b": "..."}
        ]
        
        scenarios_dir = tmp_path / "data" / "scenarios"
        scenarios_dir.mkdir(parents=True, exist_ok=True)
        
        with open(scenarios_dir / "stratified_phase2.json", "w") as f:
            json.dump(test_scenarios, f)
        
        output_dir = tmp_path / "batches"
        
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Submit batches
            submit_all_batches(
                top_variants=[v["variant_id"] for v in test_variants[:2]],
                output_dir=str(output_dir)
            )
            
            # Verify batch handles file
            handles_file = output_dir / "batch_handles.json"
            assert handles_file.exists()
            
            with open(handles_file) as f:
                handles = json.load(f)
            
            # Should have 2 variants × 2 models = 4 batches
            assert len(handles) >= 2
            
        finally:
            os.chdir(original_cwd)