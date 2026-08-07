"""Regression tests for two silent-failure bugs found on 2026-08-07.

1. **Unlogged spend.** Cost logging was entirely caller-side: the pipeline scripts called
   `cost_tracker.log_cost()` by hand and every standalone script under `scripts/` did not,
   so the §17a-§17c probe series and all label-validation runs spent real money with no
   JSONL row. `UnifiedLLMClient` now takes an opt-in `cost_tracker` and logs each call
   itself.
2. **Two-band difficulty key.** `PREFERENCE_PAIRS_DESCRIPTIVE` was keyed on the retired
   easy/hard bands, so `validate_scenario`'s descriptive fallback was a dead branch for
   every `easy_A` / `easy_B` seed.
"""

import json
from pathlib import Path

import pytest

from pipeline_a_scenarios.create_prototypes import (
    PREFERENCE_PAIRS_DESCRIPTIVE,
    REQUIRED_SCENARIOS,
    VALID_DIFFICULTY_BANDS,
    validate_scenario,
)
from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE1_PATH = REPO_ROOT / "data" / "scenarios" / "seeds_phase1.json"


@pytest.fixture(scope="module")
def seeds():
    return json.loads(PHASE1_PATH.read_text(encoding="utf-8"))


class RecordingTracker:
    """Minimal CostTracker stand-in — records what would have been billed."""

    def __init__(self):
        self.entries = []

    def log_cost(self, **kwargs):
        self.entries.append(kwargs)
        return 0.0


class StubProvider:
    """Returns a fresh response per call so cache behaviour is observable."""

    def __init__(self, usage=None):
        self.calls = 0
        self._usage = (
            usage
            if usage is not None
            else {
                "input_tokens": 100,
                "output_tokens": 50,
            }
        )

    def generate(self, **_kwargs):
        self.calls += 1
        return {"content": f"response {self.calls}", "usage": dict(self._usage)}


def _client(tracker=None, usage=None, enable_cache=False):
    """Build a client whose provider call is stubbed at the adapter boundary."""
    client = UnifiedLLMClient(
        provider="openai",
        model="gpt-5.6-sol",
        enable_cache=enable_cache,
        cost_tracker=tracker,
        client_override=object(),
    )
    stub = StubProvider(usage=usage)
    client._generate_openai = lambda *a, **k: stub.generate()
    client._stub = stub
    return client


# ------------------------------------------------------------------ auto cost logging


def test_generate_logs_one_entry_per_call():
    tracker = RecordingTracker()
    client = _client(tracker)

    client.generate(prompt="a", max_tokens=10)
    client.generate(prompt="b", max_tokens=10)

    assert len(tracker.entries) == 2
    entry = tracker.entries[0]
    assert entry["provider"] == "openai"
    assert entry["model"] == "gpt-5.6-sol"
    assert entry["input_tokens"] == 100 and entry["output_tokens"] == 50
    assert entry["call_type"] == "sync"
    assert entry["metadata"]["auto_logged"] is True


def test_no_tracker_means_no_logging_and_no_crash():
    """The default path must stay free of cost-tracking side effects."""
    client = _client(tracker=None)
    assert client.generate(prompt="a", max_tokens=10)["content"] == "response 1"


def test_cache_hits_are_not_billed_twice():
    """A cache hit issues no provider call, so it must not produce a second cost row."""
    tracker = RecordingTracker()
    client = _client(tracker, enable_cache=True)

    client.generate(prompt="same", max_tokens=10)
    client.generate(prompt="same", max_tokens=10)

    assert client._stub.calls == 1
    assert len(tracker.entries) == 1


def test_missing_usage_block_fails_loudly_rather_than_logging_zero():
    """A silent zero is indistinguishable from a free call and under-reports spend."""
    tracker = RecordingTracker()
    client = _client(tracker, usage={})

    with pytest.raises(KeyError, match="usage"):
        client.generate(prompt="a", max_tokens=10)
    assert tracker.entries == []


def test_real_cost_tracker_receives_a_billable_row(tmp_path):
    """End-to-end against the real CostTracker — a stub cannot catch a kwarg mismatch.

    This is the assertion that would have caught the original bug: the symptom was an
    empty `data/metadata/`, i.e. no row reaching the dashboard.
    """
    from pipeline_a_scenarios.utils.cost_tracker import CostTracker

    tracker = CostTracker(
        user_id="pytest_auto_log", data_dir=str(tmp_path), load_existing=False
    )
    _client(tracker).generate(prompt="a", max_tokens=10)

    rows = [
        json.loads(line)
        for line in (tmp_path / "costs_pytest_auto_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 100 and rows[0]["output_tokens"] == 50
    assert rows[0]["cost"] > 0, "a logged call must carry a non-zero cost"


def test_standalone_scripts_pass_a_tracker():
    """The scripts that produced the unlogged probe spend must now wire one in."""
    for name in ("validate_scenario_labels.py", "probe_proto01_guards.py"):
        source = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "cost_tracker=CostTracker(" in source, name


# --------------------------------------------------------------- three-band difficulty


def test_descriptive_pairs_are_keyed_on_three_bands():
    for pair_type, bands in PREFERENCE_PAIRS_DESCRIPTIVE.items():
        assert set(bands) <= VALID_DIFFICULTY_BANDS, pair_type
        assert "easy" not in bands, f"{pair_type} still carries the retired 'easy' key"


def test_required_stratification_uses_three_bands():
    for spec in REQUIRED_SCENARIOS:
        assert spec["difficulty"] in VALID_DIFFICULTY_BANDS, spec


def test_every_seed_band_has_a_descriptive_entry(seeds):
    """The fallback branch in validate_scenario must be reachable for every seed."""
    for scenario in seeds:
        bands = PREFERENCE_PAIRS_DESCRIPTIVE[scenario["pair_type"]]
        assert scenario["difficulty"] in bands, scenario["id"]


def test_two_band_difficulty_is_rejected(seeds):
    stale = dict(seeds[0], difficulty="easy")
    assert any("difficulty 'easy'" in e for e in validate_scenario(stale))


def test_seed_set_validates_clean(seeds):
    assert [e for s in seeds for e in validate_scenario(s)] == []
