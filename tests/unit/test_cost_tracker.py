"""
tests/unit/test_cost_tracker.py
12 CORE TESTS - ACTUAL 2026 PRICING
Author: Pooja Puranik
Date: 12/02/2026
"""

from pipeline_a_scenarios.utils.cost_tracker import CostTracker
import pytest
import tempfile
import json
from pathlib import Path


# ==================== FIXTURES ====================

@pytest.fixture
def isolated_tracker():
    """Fixture: Isolated CostTracker with temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = CostTracker(
            user_id="test_user",
            data_dir=tmpdir,
            load_existing=False
        )
        yield tracker


# ==================== TEST 1: INITIALIZATION ====================

def test_tracker_initialization():
    """Test 1: Verify tracker creates correct file structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = CostTracker(
            user_id="test_user",
            data_dir=tmpdir,
            load_existing=False
        )
        assert tracker.user_id == "test_user"
        assert tracker.data_path == Path(tmpdir) / "costs_test_user.jsonl"
        assert tracker.monthly_budget_limit == 200.0
        assert tracker.total_budget_limit == 1000.0
        assert tracker.month_1_2_threshold == 800.0
        assert tracker.costs == []


# ==================== TEST 2: OPENAI PRICING (2026) ====================

def test_openai_pricing(isolated_tracker):
    """Test 2: Verify OpenAI pricing calculation - 2026 rates."""
    tracker = isolated_tracker
    
    # GPT-4o: $2.50/$10.00 per 1M tokens (2026 pricing)
    cost = tracker.calculate_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=1_000_000,
        output_tokens=500_000
    )
    assert cost == 7.50  # (2.5*1 + 10*0.5) = 2.5 + 5 = 7.5
    
    # GPT-4o-mini: $0.075/$0.30 per 1M tokens (2026 pricing)
    cost = tracker.calculate_cost(
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=1_000_000,
        output_tokens=500_000
    )
    assert cost == 0.225  # (0.075*1 + 0.30*0.5) = 0.075 + 0.15 = 0.225


# ==================== TEST 3: ANTHROPIC PRICING (2026) ====================

def test_anthropic_pricing(isolated_tracker):
    """Test 3: Verify Anthropic pricing calculation - 2026 rates."""
    tracker = isolated_tracker
    
    # Claude Sonnet 4.5: $3.00/$15.00 per 1M tokens (unchanged)
    cost = tracker.calculate_cost(
        provider="anthropic",
        model="claude-sonnet-4.5",
        input_tokens=1_000_000,
        output_tokens=500_000
    )
    assert cost == 10.50  # (3*1 + 15*0.5) = 3 + 7.5 = 10.5


# ==================== TEST 4: GOOGLE PRICING (2026) ====================

def test_google_pricing(isolated_tracker):
    """Test 4: Verify Google pricing calculation - 2026 rates."""
    tracker = isolated_tracker
    
    # Gemini 2.5 Pro: $1.25/$5.00 per 1M tokens (unchanged)
    cost = tracker.calculate_cost(
        provider="google",
        model="gemini-2.5-pro",
        input_tokens=1_000_000,
        output_tokens=500_000
    )
    assert cost == 3.75  # (1.25*1 + 5*0.5) = 1.25 + 2.5 = 3.75


# ==================== TEST 5: BATCH DISCOUNT ====================

def test_batch_discount(isolated_tracker):
    """Test 5: Verify 50% batch discount."""
    tracker = isolated_tracker
    
    standard = tracker.calculate_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        batch_api=False
    )
    
    batch = tracker.calculate_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        batch_api=True
    )
    
    assert batch == standard * 0.5


# ==================== TEST 6: LOG COST ====================

def test_log_cost(isolated_tracker):
    """Test 6: Verify cost logging to JSONL."""
    tracker = isolated_tracker
    
    cost = tracker.log_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        batch_api=True,
        metadata={"test_id": "123"}
    )
    
    assert len(tracker.costs) == 1
    entry = tracker.costs[0]
    
    assert entry["provider"] == "openai"
    assert entry["model"] == "gpt-4o"
    assert entry["input_tokens"] == 1000
    assert entry["output_tokens"] == 500
    assert entry["batch_api"] is True
    assert entry["metadata"]["test_id"] == "123"
    assert "timestamp" in entry
    
    # Verify file was written
    assert tracker.data_path.exists()
    with open(tracker.data_path) as f:
        lines = f.readlines()
        assert len(lines) == 1
        saved = json.loads(lines[0])
        assert saved["provider"] == "openai"


# ==================== TEST 7: PERSISTENCE ====================

def test_persistence():
    """Test 7: Verify costs persist between sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First session
        t1 = CostTracker(
            user_id="test",
            data_dir=tmpdir,
            load_existing=False
        )
        t1.log_cost(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cost=0.05
        )
        
        # Second session
        t2 = CostTracker(
            user_id="test",
            data_dir=tmpdir,
            load_existing=True
        )
        assert len(t2.costs) == 1
        assert t2.get_total_cost() == 0.05


# ==================== TEST 8: BUDGET ALERTS ====================

def test_budget_alerts(isolated_tracker):
    """Test 8: Verify 80% budget alert triggers."""
    tracker = isolated_tracker
    tracker.total_budget_limit = 1000.0
    
    assert tracker.check_80_percent_alert() is False
    
    tracker.log_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=0,
        output_tokens=0,
        cost=800.0
    )
    assert tracker.check_80_percent_alert() is True


# ==================== TEST 9: MONTH 1-2 THRESHOLD ====================

def test_month_1_2_threshold(isolated_tracker):
    """Test 9: Verify $800 threshold for first two months."""
    tracker = isolated_tracker
    
    tracker.log_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=0,
        output_tokens=0,
        cost=500.0
    )
    assert tracker.check_month_1_2_threshold() is False
    
    tracker.reset()
    tracker.log_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=0,
        output_tokens=0,
        cost=800.0
    )
    assert tracker.check_month_1_2_threshold() is True


# ==================== TEST 10: PROVIDER BREAKDOWN ====================

def test_provider_breakdown(isolated_tracker):
    """Test 10: Verify cost breakdown by provider."""
    tracker = isolated_tracker
    
    tracker.log_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=0,
        output_tokens=0,
        cost=10.0
    )
    tracker.log_cost(
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=0,
        output_tokens=0,
        cost=5.0
    )
    tracker.log_cost(
        provider="anthropic",
        model="claude-sonnet-4.5",
        input_tokens=0,
        output_tokens=0,
        cost=20.0
    )
    tracker.log_cost(
        provider="google",
        model="gemini-2.5-pro",
        input_tokens=0,
        output_tokens=0,
        cost=15.0
    )
    
    breakdown = tracker.get_provider_breakdown()
    
    assert len(breakdown) == 3
    assert breakdown["openai"]["cost"] == 15.0  # 10 + 5
    assert breakdown["openai"]["calls"] == 2
    assert breakdown["anthropic"]["cost"] == 20.0
    assert breakdown["anthropic"]["calls"] == 1
    assert breakdown["google"]["cost"] == 15.0
    assert breakdown["google"]["calls"] == 1


# ==================== TEST 11: TEAM AGGREGATION ====================

def test_team_aggregation():
    """Test 11: Verify team lead can see all costs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Team members
        alice = CostTracker(
            user_id="alice",
            data_dir=tmpdir,
            load_existing=False
        )
        bob = CostTracker(
            user_id="bob",
            data_dir=tmpdir,
            load_existing=False
        )
        
        alice.log_cost(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cost=10.0
        )
        bob.log_cost(
            provider="anthropic",
            model="claude-sonnet-4.5",
            input_tokens=800,
            output_tokens=400,
            cost=20.0
        )
        
        # Team lead
        lead = CostTracker(
            user_id="lead",
            data_dir=tmpdir,
            load_existing=False
        )
        all_costs = lead._load_all_user_costs()
        
        assert len(all_costs) == 2
        assert sum(c["cost"] for c in all_costs) == 30.0


# ==================== TEST 12: RESET ====================

def test_reset(isolated_tracker):
    """Test 12: Verify reset clears all data."""
    tracker = isolated_tracker
    
    tracker.log_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        cost=0.05
    )
    assert tracker.data_path.exists()
    assert len(tracker.costs) == 1
    assert tracker.get_total_cost() == 0.05
    
    tracker.reset()
    
    assert not tracker.data_path.exists()
    assert len(tracker.costs) == 0
    assert tracker.get_total_cost() == 0.0
    
    # Can still log after reset
    tracker.log_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        cost=0.03
    )
    assert len(tracker.costs) == 1
    assert tracker.get_total_cost() == 0.03