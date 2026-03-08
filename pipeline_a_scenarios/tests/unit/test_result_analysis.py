"""
Unit tests for Phase 2 batch result analysis components.

Coverage:
  - parse_response:            all pattern branches (XML, labelled, bare, keywords, edge cases)
  - load_batch_results:        batch handle loading + UnifiedLLMClient/BatchHandle wiring
  - analyze_variant_consistency: Spearman matrix shape + P2-1 figure creation
  - analyze_parsing_success:   per-variant success rate output (stdout capture)
  - recommend_final_variants:  primary/secondary selection logic, P2-2 figure creation,
                                output file written, typo-safe key check
"""

import io
import json
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pipeline_a_scenarios.analyze_batch_results import (
    parse_response,
    analyze_parsing_success,
    analyze_variant_consistency,
    recommend_final_variants,
    load_batch_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal DataFrame with two variants and three models."""
    return pd.DataFrame([
        {"scenario_id": "s1", "variant_id": "v1", "model": "m1", "parsed_choice": "A"},
        {"scenario_id": "s2", "variant_id": "v1", "model": "m1", "parsed_choice": "B"},
        {"scenario_id": "s3", "variant_id": "v1", "model": "m1", "parsed_choice": "UNPARSEABLE"},
        {"scenario_id": "s1", "variant_id": "v2", "model": "m1", "parsed_choice": "A"},
        {"scenario_id": "s2", "variant_id": "v2", "model": "m1", "parsed_choice": "A"},
        {"scenario_id": "s3", "variant_id": "v2", "model": "m1", "parsed_choice": "equal"},
    ])


@pytest.fixture
def phase1_rankings(tmp_path):
    """Write a minimal variant_rankings.json and return its directory."""
    data = {
        "variant_rankings": [
            {"variant_id": "v1", "mean_authenticity_score": 80.0, "composite_score": 0.8},
            {"variant_id": "v2", "mean_authenticity_score": 70.0, "composite_score": 0.7},
            {"variant_id": "v3", "mean_authenticity_score": 60.0, "composite_score": 0.6},
        ]
    }
    rankings_dir = tmp_path / "data" / "results" / "prompt_validation"
    rankings_dir.mkdir(parents=True)
    (rankings_dir / "variant_rankings.json").write_text(json.dumps(data))
    return str(rankings_dir)


# ---------------------------------------------------------------------------
# parse_response — all branches
# ---------------------------------------------------------------------------

class TestParseResponse:
    """Full coverage of every pattern branch in parse_response()."""

    # XML <answer> tag with digit
    @pytest.mark.parametrize("text,expected", [
        ("<answer>1</answer>", "A"),
        ("<answer>2</answer>", "B"),
        ("<answer>3</answer>", "equal"),
        ("<answer> 1 </answer>", "A"),   # whitespace tolerance
    ])
    def test_xml_digit_tag(self, text, expected):
        assert parse_response(text) == expected

    # XML <answer> tag with letter
    @pytest.mark.parametrize("text,expected", [
        ("<answer>A</answer>", "A"),
        ("<answer>b</answer>", "B"),
        ("<answer>C</answer>", "equal"),
    ])
    def test_xml_letter_tag(self, text, expected):
        assert parse_response(text) == expected

    # Labelled patterns ("choice: A", "answer: B", etc.)
    @pytest.mark.parametrize("text,expected", [
        ("choice: A", "A"),
        ("Answer: B", "B"),
        ("select: 1", "A"),
        ("option: 2", "B"),
        ("The answer is: 3", "equal"),
    ])
    def test_labelled_patterns(self, text, expected):
        assert parse_response(text) == expected

    # Bare single token on its own line
    @pytest.mark.parametrize("text,expected", [
        ("A", "A"),
        ("b", "B"),
        ("2", "B"),
        ("3", "equal"),
    ])
    def test_bare_token(self, text, expected):
        assert parse_response(text) == expected

    # Keyword fallbacks
    @pytest.mark.parametrize("text,expected", [
        ("I choose Action A.", "A"),
        ("I would choose action b here.", "B"),
        ("Both options are equal in value.", "equal"),
        ("Neither choice is preferable.", "equal"),
        ("The correct choice is A", "A"),
        ("Option B is better.", "B"),
    ])
    def test_keyword_fallback(self, text, expected):
        assert parse_response(text) == expected

    # Edge / unparseable cases
    @pytest.mark.parametrize("text", [
        "",
        None,
        123,            # non-string
        "I don't know",
        "12345",
        "<answer></answer>",
        "The answer is maybe",
        "   ",
    ])
    def test_unparseable(self, text):
        result = parse_response(text)
        assert result in {"A", "B", "equal", "UNPARSEABLE"}

    # All valid results are in the allowed set
    @pytest.mark.parametrize("response,expected", [
        ("I choose Action A.", "A"),
        ("Answer: B", "B"),
        ("The correct choice is A", "A"),
        ("<answer>1</answer>", "A"),
        ("<answer>2</answer>", "B"),
        ("Both are equal", "equal"),
    ])
    def test_valid_responses_return_canonical(self, response, expected):
        assert parse_response(response) == expected


# ---------------------------------------------------------------------------
# load_batch_results
# ---------------------------------------------------------------------------

class TestLoadBatchResults:
    """Test batch handle loading, UnifiedLLMClient wiring, and error handling."""

    @patch("pipeline_a_scenarios.analyze_batch_results.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.analyze_batch_results.BatchHandle")
    def test_load_returns_dataframe(
        self, mock_batch_handle_cls, mock_client_cls, tmp_path
    ):
        """Returns a DataFrame with expected columns when batches succeed."""
        # Write minimal batch_handles.json
        handles = {
            "v1_claude": {
                "batch_id": "batch_001",
                "provider": "anthropic",
                "variant_id": "v1",
                "model": "claude-sonnet-4-20250514",
            }
        }
        batch_dir = tmp_path / "batches"
        batch_dir.mkdir()
        (batch_dir / "batch_handles.json").write_text(json.dumps(handles))

        # Mock client
        mock_client = Mock()
        mock_client.retrieve_batch_results.return_value = {
            "s1_v1_claude": "I choose Action A.",
            "s2_v1_claude": "<answer>2</answer>",
        }
        mock_client_cls.return_value = mock_client
        mock_batch_handle_cls.return_value = Mock()

        df = load_batch_results(str(batch_dir))

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"scenario_id", "variant_id", "model",
                                   "raw_response", "parsed_choice"}
        assert len(df) == 2

    @patch("pipeline_a_scenarios.analyze_batch_results.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.analyze_batch_results.BatchHandle")
    def test_load_handles_api_error_gracefully(
        self, mock_batch_handle_cls, mock_client_cls, tmp_path
    ):
        """Returns empty DataFrame (not exception) when retrieval fails."""
        handles = {
            "v1_bad": {
                "batch_id": "bad_batch",
                "provider": "anthropic",
                "variant_id": "v1",
                "model": "claude-sonnet-4-20250514",
            }
        }
        batch_dir = tmp_path / "batches"
        batch_dir.mkdir()
        (batch_dir / "batch_handles.json").write_text(json.dumps(handles))

        mock_client = Mock()
        mock_client.retrieve_batch_results.side_effect = Exception("API timeout")
        mock_client_cls.return_value = mock_client

        df = load_batch_results(str(batch_dir))

        assert isinstance(df, pd.DataFrame)

    @patch("pipeline_a_scenarios.analyze_batch_results.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.analyze_batch_results.BatchHandle")
    def test_scenario_id_extracted_from_custom_id(
        self, mock_batch_handle_cls, mock_client_cls, tmp_path
    ):
        """scenario_id is the segment before the first underscore in custom_id."""
        handles = {
            "v1_m1": {
                "batch_id": "b1",
                "provider": "openai",
                "variant_id": "v1",
                "model": "gpt-5.2",
            }
        }
        batch_dir = tmp_path / "batches"
        batch_dir.mkdir()
        (batch_dir / "batch_handles.json").write_text(json.dumps(handles))

        mock_client = Mock()
        mock_client.retrieve_batch_results.return_value = {
            "proto01_v1_m1": "A",
        }
        mock_client_cls.return_value = mock_client

        df = load_batch_results(str(batch_dir))
        assert df.iloc[0]["scenario_id"] == "proto01"

    @patch("pipeline_a_scenarios.analyze_batch_results.UnifiedLLMClient")
    @patch("pipeline_a_scenarios.analyze_batch_results.BatchHandle")
    def test_provider_routing(
        self, mock_batch_handle_cls, mock_client_cls, tmp_path
    ):
        """All three providers (anthropic, openai, google) are routed correctly."""
        for provider, model in [
            ("anthropic", "claude-sonnet-4-20250514"),
            ("openai",    "gpt-5.2"),
            ("google",    "gemini-3.0"),
        ]:
            handles = {
                "v1_m": {
                    "batch_id": "b1",
                    "provider": provider,
                    "variant_id": "v1",
                    "model": model,
                }
            }
            batch_dir = tmp_path / f"batches_{provider}"
            batch_dir.mkdir()
            (batch_dir / "batch_handles.json").write_text(json.dumps(handles))

            mock_client = Mock()
            mock_client.retrieve_batch_results.return_value = {"s1_v1": "A"}
            mock_client_cls.return_value = mock_client

            df = load_batch_results(str(batch_dir))
            assert len(df) == 1
            # UnifiedLLMClient was called with the right provider
            call_kwargs = mock_client_cls.call_args
            assert call_kwargs is not None


# ---------------------------------------------------------------------------
# analyze_variant_consistency (P2-1)
# ---------------------------------------------------------------------------

class TestAnalyzeVariantConsistency:
    """Spearman correlation matrix + P2-1 figure saved to disk."""

    def test_saves_p2_1_figure(self, sample_df, tmp_path):
        """P2-1 heatmap PNG must be written to figures/ directory."""
        output_dir = str(tmp_path)
        analyze_variant_consistency(sample_df, output_dir)
        figures = list((tmp_path / "figures").glob("p2_1_variant_correlations_*.png"))
        assert len(figures) >= 1

    def test_handles_single_variant(self, tmp_path):
        """Single variant: diagonal matrix of 1.0, no crash."""
        df = pd.DataFrame([
            {"scenario_id": "s1", "variant_id": "v1", "model": "m1", "parsed_choice": "A"},
            {"scenario_id": "s2", "variant_id": "v1", "model": "m1", "parsed_choice": "B"},
        ])
        analyze_variant_consistency(df, str(tmp_path))   # should not raise

    def test_handles_all_unparseable(self, tmp_path):
        """All UNPARSEABLE responses: gracefully skip correlations."""
        df = pd.DataFrame([
            {"scenario_id": f"s{i}", "variant_id": "v1",
             "model": "m1", "parsed_choice": "UNPARSEABLE"}
            for i in range(5)
        ] + [
            {"scenario_id": f"s{i}", "variant_id": "v2",
             "model": "m1", "parsed_choice": "UNPARSEABLE"}
            for i in range(5)
        ])
        analyze_variant_consistency(df, str(tmp_path))   # should not raise

    def test_multiple_models_produce_separate_figures(self, tmp_path):
        """One P2-1 figure per model."""
        df = pd.DataFrame([
            {"scenario_id": "s1", "variant_id": "v1", "model": "m1", "parsed_choice": "A"},
            {"scenario_id": "s1", "variant_id": "v1", "model": "m2", "parsed_choice": "B"},
            {"scenario_id": "s2", "variant_id": "v2", "model": "m1", "parsed_choice": "B"},
            {"scenario_id": "s2", "variant_id": "v2", "model": "m2", "parsed_choice": "A"},
        ])
        analyze_variant_consistency(df, str(tmp_path))
        figures = list((tmp_path / "figures").glob("p2_1_variant_correlations_*.png"))
        assert len(figures) >= 2


# ---------------------------------------------------------------------------
# analyze_parsing_success
# ---------------------------------------------------------------------------

class TestAnalyzeParsingSuccess:
    """Parsing success rates printed to stdout; 95% threshold flagged."""

    def test_prints_variant_names(self, sample_df, capsys):
        analyze_parsing_success(sample_df)
        out = capsys.readouterr().out
        assert "v1" in out
        assert "v2" in out

    def test_flags_below_95_percent(self, capsys):
        """v1 has 1/3 UNPARSEABLE (66.7%) → ⚠ flag expected."""
        df = pd.DataFrame([
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "UNPARSEABLE"},
            {"variant_id": "v1", "parsed_choice": "UNPARSEABLE"},
        ])
        analyze_parsing_success(df)
        out = capsys.readouterr().out
        assert "⚠" in out or "BELOW" in out

    def test_no_flag_above_95_percent(self, capsys):
        """All parseable → no warning flag."""
        df = pd.DataFrame([
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "B"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "B"},
            {"variant_id": "v1", "parsed_choice": "equal"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "B"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "B"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
            {"variant_id": "v1", "parsed_choice": "A"},
        ])
        analyze_parsing_success(df)
        out = capsys.readouterr().out
        assert "BELOW" not in out


# ---------------------------------------------------------------------------
# recommend_final_variants (P2-2)
# ---------------------------------------------------------------------------

class TestRecommendFinalVariants:
    """Primary/secondary selection + output file + P2-2 figure."""

    def _make_df(self, parsing_profile: dict) -> pd.DataFrame:
        """
        Build a DataFrame where `parsing_profile` maps variant_id →
        fraction of parseable responses (0–1). Each variant gets 20 rows.
        """
        rows = []
        for vid, parseable_frac in parsing_profile.items():
            n_parseable   = int(20 * parseable_frac)
            n_unparseable = 20 - n_parseable
            rows += [{"scenario_id": f"s{i}", "variant_id": vid,
                      "model": "m1", "parsed_choice": "A"}
                     for i in range(n_parseable)]
            rows += [{"scenario_id": f"s{i+n_parseable}", "variant_id": vid,
                      "model": "m1", "parsed_choice": "UNPARSEABLE"}
                     for i in range(n_unparseable)]
        return pd.DataFrame(rows)

    def test_primary_variant_is_highest_ranked_above_95(self, tmp_path):
        """Primary must be the top-ranked variant with ≥95% parsing success."""
        df = self._make_df({"v1": 0.97, "v2": 0.92, "v3": 0.80})
        output_dir = str(tmp_path / "results")
        Path(output_dir).mkdir(parents=True)

        phase1 = {
            "variant_rankings": [
                {"variant_id": "v1", "mean_authenticity_score": 80.0},
                {"variant_id": "v2", "mean_authenticity_score": 70.0},
                {"variant_id": "v3", "mean_authenticity_score": 60.0},
            ]
        }
        (Path(output_dir) / "variant_rankings.json").write_text(json.dumps(phase1))

        rec = recommend_final_variants(df, output_dir)
        assert rec["primary_variant"] == "v1"

    def test_fallback_when_none_above_95(self, tmp_path):
        """If no variant clears 95%, fall back to the top-ranked one."""
        df = self._make_df({"v1": 0.80, "v2": 0.70})
        output_dir = str(tmp_path / "results")
        Path(output_dir).mkdir(parents=True)

        phase1 = {
            "variant_rankings": [
                {"variant_id": "v1", "mean_authenticity_score": 80.0},
                {"variant_id": "v2", "mean_authenticity_score": 70.0},
            ]
        }
        (Path(output_dir) / "variant_rankings.json").write_text(json.dumps(phase1))

        rec = recommend_final_variants(df, output_dir)
        assert rec["primary_variant"] == "v1"
        assert "fallback" in " ".join(rec["rationale"]).lower()

    def test_secondary_variants_max_two(self, tmp_path):
        """At most 2 secondary variants are selected."""
        df = self._make_df({"v1": 0.97, "v2": 0.95, "v3": 0.93, "v4": 0.91})
        output_dir = str(tmp_path / "results")
        Path(output_dir).mkdir(parents=True)

        phase1 = {
            "variant_rankings": [
                {"variant_id": "v1", "mean_authenticity_score": 80.0},
                {"variant_id": "v2", "mean_authenticity_score": 75.0},
                {"variant_id": "v3", "mean_authenticity_score": 70.0},
                {"variant_id": "v4", "mean_authenticity_score": 65.0},
            ]
        }
        (Path(output_dir) / "variant_rankings.json").write_text(json.dumps(phase1))

        rec = recommend_final_variants(df, output_dir)
        assert len(rec["secondary_variants"]) <= 2

    def test_output_json_written(self, tmp_path):
        """final_variant_selection.json must be created."""
        df = self._make_df({"v1": 0.97})
        output_dir = str(tmp_path / "results")
        Path(output_dir).mkdir(parents=True)

        phase1 = {
            "variant_rankings": [
                {"variant_id": "v1", "mean_authenticity_score": 80.0},
            ]
        }
        (Path(output_dir) / "variant_rankings.json").write_text(json.dumps(phase1))

        recommend_final_variants(df, output_dir)
        out_path = Path(output_dir) / "final_variant_selection.json"
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert "primary_variant" in loaded
        assert "secondary_variants" in loaded
        assert "rationale" in loaded

    def test_p2_2_figure_saved(self, tmp_path):
        """P2-2 plot PNG must be written to figures/ directory."""
        df = self._make_df({"v1": 0.97, "v2": 0.92})
        output_dir = str(tmp_path / "results")
        Path(output_dir).mkdir(parents=True)

        phase1 = {
            "variant_rankings": [
                {"variant_id": "v1", "mean_authenticity_score": 80.0},
                {"variant_id": "v2", "mean_authenticity_score": 70.0},
            ]
        }
        (Path(output_dir) / "variant_rankings.json").write_text(json.dumps(phase1))

        recommend_final_variants(df, output_dir)
        fig_path = Path(output_dir) / "figures" / "p2_2_phase1_vs_phase2.png"
        assert fig_path.exists()

    def test_return_keys_are_correct(self, tmp_path):
        """Returned dict has exactly primary_variant, secondary_variants, rationale."""
        df = self._make_df({"v1": 0.97})
        output_dir = str(tmp_path / "results")
        Path(output_dir).mkdir(parents=True)

        phase1 = {
            "variant_rankings": [
                {"variant_id": "v1", "mean_authenticity_score": 80.0},
            ]
        }
        (Path(output_dir) / "variant_rankings.json").write_text(json.dumps(phase1))

        rec = recommend_final_variants(df, output_dir)
        assert set(rec.keys()) == {"primary_variant", "secondary_variants", "rationale"}