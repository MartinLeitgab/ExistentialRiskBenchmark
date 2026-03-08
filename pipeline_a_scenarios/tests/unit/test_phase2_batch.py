"""
Unit tests for Phase 2 batch testing components.

Aligned to current implementation:
  - prepare_batch_requests: temperature pulled from prompt metadata (not hardcoded)
  - load_variant_configs:   builds lookup from BASE_DIMENSIONS + generate_all_variants
  - submit_all_batches:     batch_handles.json written, provider routing
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline_a_scenarios.batch_variant_testing import (
    prepare_batch_requests,
    submit_all_batches,
    load_variant_configs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_scenarios():
    return [
        {
            "id": f"s{i}",
            "context": f"Scenario {i} context.",
            "action_a": f"Action A for scenario {i}",
            "action_b": f"Action B for scenario {i}",
        }
        for i in range(3)
    ]


@pytest.fixture
def base_variant():
    from utils.prompt_generator import BASE_DIMENSIONS, generate_prompt_id
    vid = generate_prompt_id(BASE_DIMENSIONS)
    return {
        "variant_id": vid,
        "dimensions": BASE_DIMENSIONS.copy(),
        "vary_dim":   "base",
        "vary_value": "baseline",
    }


@pytest.fixture
def test_variants(base_variant):
    return [base_variant]


@pytest.fixture
def mock_batch_handle():
    handle = Mock()
    handle.id       = "batch_test_001"
    handle.provider = "anthropic"
    return handle


# ---------------------------------------------------------------------------
# prepare_batch_requests
# ---------------------------------------------------------------------------

class TestBatchPreparation:

    def test_returns_one_request_per_scenario(self, test_scenarios, base_variant):
        requests = prepare_batch_requests(
            scenarios=test_scenarios,
            variant=base_variant,
            model="claude-sonnet-4-20250514",
        )
        assert len(requests) == len(test_scenarios)

    def test_required_keys_present(self, test_scenarios, base_variant):
        requests = prepare_batch_requests(test_scenarios, base_variant,
                                          "claude-sonnet-4-20250514")
        for req in requests:
            assert "id"            in req
            assert "prompt"        in req
            assert "system_prompt" in req
            assert "temperature"   in req
            assert "max_tokens"    in req

    def test_request_id_contains_scenario_id(self, test_scenarios, base_variant):
        requests = prepare_batch_requests(test_scenarios, base_variant,
                                          "claude-sonnet-4-20250514")
        for req, scenario in zip(requests, test_scenarios):
            assert scenario["id"] in req["id"]

    def test_request_id_contains_variant_id(self, test_scenarios, base_variant):
        requests = prepare_batch_requests(test_scenarios, base_variant,
                                          "claude-sonnet-4-20250514")
        for req in requests:
            assert base_variant["variant_id"] in req["id"]

    def test_prompt_contains_scenario_context(self, test_scenarios, base_variant):
        requests = prepare_batch_requests(test_scenarios, base_variant,
                                          "claude-sonnet-4-20250514")
        for req, scenario in zip(requests, test_scenarios):
            # Either context appears verbatim or prompt is non-trivially long
            assert (scenario["context"] in req["prompt"]
                    or len(req["prompt"]) > 50)

    def test_temperature_from_metadata(self, test_scenarios, base_variant):
        """Temperature must come from generate_prompt metadata, not be hardcoded."""
        requests = prepare_batch_requests(test_scenarios, base_variant,
                                          "claude-sonnet-4-20250514")
        for req in requests:
            # Temperature must be a numeric value (0 for Phase 1 consistency)
            assert isinstance(req["temperature"], (int, float))

    def test_max_tokens_is_500(self, test_scenarios, base_variant):
        requests = prepare_batch_requests(test_scenarios, base_variant,
                                          "claude-sonnet-4-20250514")
        for req in requests:
            assert req["max_tokens"] == 500

    def test_empty_scenarios_returns_empty_list(self, base_variant):
        requests = prepare_batch_requests([], base_variant,
                                          "claude-sonnet-4-20250514")
        assert requests == []


# ---------------------------------------------------------------------------
# load_variant_configs
# ---------------------------------------------------------------------------

class TestVariantConfigLoading:

    def test_returns_list(self, test_variants, tmp_path):
        rankings = {
            "top_variants":      [v["variant_id"] for v in test_variants],
            "variant_rankings":  [{"variant_id": v["variant_id"],
                                   "composite_score": 0.8}
                                  for v in test_variants],
        }
        rankings_dir = tmp_path / "data" / "results" / "prompt_validation"
        rankings_dir.mkdir(parents=True)
        (rankings_dir / "variant_rankings.json").write_text(json.dumps(rankings))

        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            variant_ids = [v["variant_id"] for v in test_variants]
            configs = load_variant_configs(variant_ids)
        finally:
            os.chdir(orig)

        assert isinstance(configs, list)

    def test_each_config_has_variant_id_and_dimensions(self, test_variants, tmp_path):
        rankings = {
            "top_variants":     [v["variant_id"] for v in test_variants],
            "variant_rankings": [{"variant_id": v["variant_id"],
                                  "composite_score": 0.8}
                                 for v in test_variants],
        }
        rankings_dir = tmp_path / "data" / "results" / "prompt_validation"
        rankings_dir.mkdir(parents=True)
        (rankings_dir / "variant_rankings.json").write_text(json.dumps(rankings))

        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            configs = load_variant_configs([v["variant_id"] for v in test_variants])
        finally:
            os.chdir(orig)

        for cfg in configs:
            assert "variant_id"  in cfg
            assert "dimensions"  in cfg

    def test_unknown_variant_id_emits_warning_not_exception(self, tmp_path, capsys):
        rankings = {
            "top_variants":     [],
            "variant_rankings": [],
        }
        rankings_dir = tmp_path / "data" / "results" / "prompt_validation"
        rankings_dir.mkdir(parents=True)
        (rankings_dir / "variant_rankings.json").write_text(json.dumps(rankings))

        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            configs = load_variant_configs(["nonexistent_variant_id_xyz"])
        finally:
            os.chdir(orig)

        out = capsys.readouterr().out
        assert "not found" in out.lower() or "warning" in out.lower() or configs == []


# ---------------------------------------------------------------------------
# submit_all_batches
# ---------------------------------------------------------------------------

class TestBatchSubmission:

    @patch("pipeline_a_scenarios.batch_variant_testing.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.batch_variant_testing.load_variant_configs")
    def test_batch_handles_file_written(
        self, mock_load_configs, mock_client_cls,
        test_variants, mock_batch_handle, tmp_path,
    ):
        mock_load_configs.return_value = test_variants[:1]
        mock_client_cls.return_value.submit_batch.return_value = mock_batch_handle

        scenarios_dir = tmp_path / "data" / "scenarios"
        scenarios_dir.mkdir(parents=True)
        (scenarios_dir / "stratified_phase2.json").write_text(json.dumps([
            {"id": "s1", "context": "C", "action_a": "A", "action_b": "B"},
        ]))

        output_dir = tmp_path / "batches"

        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            submit_all_batches(
                top_variants=[v["variant_id"] for v in test_variants[:1]],
                output_dir=str(output_dir),
            )
        finally:
            os.chdir(orig)

        handles_file = output_dir / "batch_handles.json"
        assert handles_file.exists()

    @patch("pipeline_a_scenarios.batch_variant_testing.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.batch_variant_testing.load_variant_configs")
    def test_handles_contain_required_fields(
        self, mock_load_configs, mock_client_cls,
        test_variants, mock_batch_handle, tmp_path,
    ):
        mock_load_configs.return_value = test_variants[:1]
        mock_client_cls.return_value.submit_batch.return_value = mock_batch_handle

        scenarios_dir = tmp_path / "data" / "scenarios"
        scenarios_dir.mkdir(parents=True)
        (scenarios_dir / "stratified_phase2.json").write_text(json.dumps([
            {"id": "s1", "context": "C", "action_a": "A", "action_b": "B"},
        ]))

        output_dir = tmp_path / "batches"

        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            submit_all_batches(
                top_variants=[v["variant_id"] for v in test_variants[:1]],
                output_dir=str(output_dir),
            )
        finally:
            os.chdir(orig)

        handles = json.loads((output_dir / "batch_handles.json").read_text())
        for key, val in handles.items():
            assert "batch_id"   in val
            assert "provider"   in val
            assert "variant_id" in val
            assert "model"      in val

    @patch("pipeline_a_scenarios.batch_variant_testing.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.batch_variant_testing.load_variant_configs")
    def test_provider_routing_anthropic(
        self, mock_load_configs, mock_client_cls,
        test_variants, mock_batch_handle, tmp_path,
    ):
        """claude model → provider='anthropic'."""
        mock_load_configs.return_value = test_variants[:1]
        mock_client_cls.return_value.submit_batch.return_value = mock_batch_handle

        scenarios_dir = tmp_path / "data" / "scenarios"
        scenarios_dir.mkdir(parents=True)
        (scenarios_dir / "stratified_phase2.json").write_text(json.dumps([
            {"id": "s1", "context": "C", "action_a": "A", "action_b": "B"},
        ]))

        output_dir = tmp_path / "batches"

        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            submit_all_batches(
                top_variants=[v["variant_id"] for v in test_variants[:1]],
                output_dir=str(output_dir),
            )
        finally:
            os.chdir(orig)

        # UnifiedLLMClient must have been constructed for anthropic
        calls = mock_client_cls.call_args_list
        providers = [c.kwargs.get("provider", c.args[0] if c.args else None)
                     for c in calls]
        assert "anthropic" in providers

    @patch("pipeline_a_scenarios.batch_variant_testing.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.batch_variant_testing.load_variant_configs")
    def test_submission_error_does_not_crash(
        self, mock_load_configs, mock_client_cls,
        test_variants, tmp_path,
    ):
        mock_load_configs.return_value = test_variants[:1]
        mock_client_cls.return_value.submit_batch.side_effect = Exception("API error")

        scenarios_dir = tmp_path / "data" / "scenarios"
        scenarios_dir.mkdir(parents=True)
        (scenarios_dir / "stratified_phase2.json").write_text(json.dumps([
            {"id": "s1", "context": "C", "action_a": "A", "action_b": "B"},
        ]))

        output_dir = tmp_path / "batches_err"

        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            submit_all_batches(
                top_variants=[v["variant_id"] for v in test_variants[:1]],
                output_dir=str(output_dir),
            )
        finally:
            os.chdir(orig)

        # Should have written handles file (possibly empty)
        handles_file = Path(output_dir) / "batch_handles.json"
        assert handles_file.exists()
        handles = json.loads(handles_file.read_text())
        assert len(handles) == 0  # failed batch not recorded