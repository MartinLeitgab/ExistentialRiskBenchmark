"""
Unit tests for prepare_seed_scenarios.py

Coverage:
  - select_seed_scenarios:   output file written, correct count, stratification
                              (difficulty mix, pair_type mix), truncation to n_seeds
"""

import json
import pytest
from pathlib import Path

from pipeline_a_scenarios.prepare_seed_scenarios import select_seed_scenarios


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_scenario(id_, difficulty, pair_type, stakes_level, domain):
    return {
        "id":           id_,
        "context":      f"Context for {id_}",
        "action_a":     "Option A",
        "action_b":     "Option B",
        "difficulty":   difficulty,
        "pair_type":    pair_type,
        "stakes_level": stakes_level,
        "domain":       domain,
    }


@pytest.fixture
def all_scenarios():
    """
    12 scenarios that cover the stratification criteria used by
    select_seed_scenarios (2 easy, 4+ hard, mixed pair_type, mixed domain).
    """
    return [
        _make_scenario("proto_01", "easy",   "inst_vs_pro",  "low",    "work"),
        _make_scenario("proto_02", "hard",   "inst_vs_anti", "high",   "safety"),
        _make_scenario("proto_03", "easy",   "pro_vs_anti",  "medium", "home"),
        _make_scenario("proto_04", "hard",   "inst_vs_pro",  "high",   "medical"),
        _make_scenario("proto_05", "hard",   "inst_vs_anti", "medium", "legal"),
        _make_scenario("tough_01", "hard",   "pro_vs_anti",  "high",   "finance"),
        _make_scenario("proto_06", "medium", "inst_vs_pro",  "low",    "work"),
        _make_scenario("proto_07", "medium", "inst_vs_anti", "medium", "home"),
        _make_scenario("proto_08", "easy",   "pro_vs_anti",  "low",    "social"),
        _make_scenario("proto_09", "hard",   "inst_vs_pro",  "high",   "safety"),
        _make_scenario("proto_10", "hard",   "inst_vs_anti", "medium", "legal"),
        _make_scenario("proto_11", "easy",   "pro_vs_anti",  "low",    "work"),
    ]


@pytest.fixture
def all_scenarios_file(tmp_path, all_scenarios):
    path = tmp_path / "all_scenarios.json"
    path.write_text(json.dumps(all_scenarios))
    return str(path)


@pytest.fixture
def output_path(tmp_path):
    return str(tmp_path / "seeds_phase1.json")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectSeedScenarios:

    def test_output_file_created(self, all_scenarios_file, output_path):
        select_seed_scenarios(all_scenarios_file, output_path)
        assert Path(output_path).exists()

    def test_output_is_valid_json(self, all_scenarios_file, output_path):
        select_seed_scenarios(all_scenarios_file, output_path)
        data = json.loads(Path(output_path).read_text())
        assert isinstance(data, list)

    def test_returns_exactly_n_seeds(self, all_scenarios_file, output_path):
        seeds = select_seed_scenarios(all_scenarios_file, output_path, n_seeds=6)
        assert len(seeds) == 6

    def test_written_file_matches_returned_list(self, all_scenarios_file, output_path):
        seeds = select_seed_scenarios(all_scenarios_file, output_path)
        written = json.loads(Path(output_path).read_text())
        assert seeds == written

    def test_truncation_to_n_seeds(self, all_scenarios_file, output_path):
        seeds = select_seed_scenarios(all_scenarios_file, output_path, n_seeds=3)
        assert len(seeds) == 3

    def test_seeds_are_dicts_with_required_keys(self, all_scenarios_file, output_path):
        seeds = select_seed_scenarios(all_scenarios_file, output_path)
        required = {"id", "context", "action_a", "action_b"}
        for s in seeds:
            assert required.issubset(s.keys())

    def test_includes_easy_scenarios(self, all_scenarios_file, output_path):
        """Implementation picks easy scenarios first (indices 0 and 2)."""
        seeds = select_seed_scenarios(all_scenarios_file, output_path)
        difficulties = [s.get("difficulty") for s in seeds]
        assert "easy" in difficulties

    def test_includes_hard_scenarios(self, all_scenarios_file, output_path):
        seeds = select_seed_scenarios(all_scenarios_file, output_path)
        difficulties = [s.get("difficulty") for s in seeds]
        assert "hard" in difficulties

    def test_no_duplicate_scenario_ids(self, all_scenarios_file, output_path):
        seeds = select_seed_scenarios(all_scenarios_file, output_path)
        ids = [s["id"] for s in seeds]
        assert len(ids) == len(set(ids))

    def test_parent_directory_created_if_missing(self, all_scenarios_file, tmp_path):
        deep_output = str(tmp_path / "deep" / "nested" / "seeds.json")
        select_seed_scenarios(all_scenarios_file, deep_output)
        assert Path(deep_output).exists()

    def test_graceful_with_few_hard_scenarios(self, tmp_path, output_path):
        """If fewer than 4 hard scenarios exist, still produces as many seeds as possible."""
        scenarios = [
            _make_scenario("easy_01", "easy", "inst_vs_pro",  "low",  "work"),
            _make_scenario("easy_02", "easy", "pro_vs_anti",  "low",  "home"),
            _make_scenario("hard_01", "hard", "inst_vs_anti", "high", "safety"),
        ]
        all_path = str(tmp_path / "few_hard.json")
        Path(all_path).write_text(json.dumps(scenarios))

        seeds = select_seed_scenarios(all_path, output_path, n_seeds=6)
        assert len(seeds) <= 3   # can't exceed available scenarios
        assert Path(output_path).exists()