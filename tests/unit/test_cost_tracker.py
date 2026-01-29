"""
tests/unit/test_cost_tracker.py
UNIT TESTS FOR COST TRACKER
Tests the CostTracker class for tracking LLM API costs, budgets, alerts, and
analytics.
Author: Pooja Puranik
version: 2.0.0
date: 26/01/2026

10 Critical Infrastructure Tests Focused on:
1. JSONL append-only persistence 
2. Per-user file isolation 
3. Team aggregate dashboard 
4. Budget tracking configuration 
5. Data persistence across sessions
6. Reset functionality
7. Auto-log integration with LLM client
8. CSV export functionality
9. Singleton pattern
10. Error handling
"""
from pipeline_a_scenarios.utils.cost_tracker import CostTracker
import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

# Fix imports for tests/unit/ directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pipeline_a_scenarios.utils.cost_tracker import CostTracker, get_tracker


# ==================== FIXTURES ====================
@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for isolated tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def fresh_tracker(temp_data_dir):
    """Create fresh CostTracker with default settings."""
    return CostTracker(
        user_id="test_user",
        data_dir=temp_data_dir,
        load_existing=False,
        monthly_budget_limit=200.0,
        total_budget_limit=1000.0,
    )


@pytest.fixture
def multi_user_data(temp_data_dir):
    """Create test data for multiple users (team aggregation)."""
    # Create user files
    users = [
        ("alice", "openai", 0.15),
        ("bob", "anthropic", 0.25),
        ("charlie", "google", 0.10),
    ]
    
    for user, provider, cost in users:
        jsonl_path = Path(temp_data_dir) / f"costs_{user}.jsonl"
        with open(jsonl_path, "w") as f:
            entry = {
                "timestamp": "2026-01-15T10:30:00",
                "provider": provider,
                "model": f"{provider}-model",
                "input_tokens": 1000,
                "output_tokens": 500,
                "cost": cost,
                "batch_api": False,
                "metadata": {}
            }
            json.dump(entry, f)
            f.write("\n")
    
    return temp_data_dir


# ==================== TEST 1: JSONL APPEND-ONLY PERSISTENCE ====================
def test_jsonl_append_only_persistence(fresh_tracker, temp_data_dir):
    """Test JSONL append-only format (TL requirement #1)."""
    tracker = fresh_tracker
    
    # Log 3 costs
    for i in range(3):
        tracker.log_cost(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            metadata={"call_id": i}
        )
    
    # Verify JSONL file
    jsonl_path = Path(temp_data_dir) / "costs_test_user.jsonl"
    assert jsonl_path.exists()
    
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 3  # Each append adds new line
        
        # Verify each line is valid JSON
        for i, line in enumerate(lines):
            entry = json.loads(line.strip())
            assert entry["provider"] == "openai"
            assert entry["metadata"]["call_id"] == i


# ==================== TEST 2: PER-USER FILE ISOLATION ====================
def test_per_user_file_isolation(temp_data_dir):
    """Test per-user cost files (TL requirement #2)."""
    # Create separate trackers for different users
    tracker1 = CostTracker(user_id="alice", data_dir=temp_data_dir)
    tracker2 = CostTracker(user_id="bob", data_dir=temp_data_dir)
    
    # Each logs to their own file
    tracker1.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    tracker2.log_cost("anthropic", "claude-sonnet-4.5", 800, 400, cost=0.05)
    
    # Verify separate files
    alice_file = Path(temp_data_dir) / "costs_alice.jsonl"
    bob_file = Path(temp_data_dir) / "costs_bob.jsonl"
    
    assert alice_file.exists()
    assert bob_file.exists()
    
    # Verify isolation
    with open(alice_file, "r") as f:
        alice_data = json.loads(f.readline().strip())
        assert alice_data["provider"] == "openai"
    
    with open(bob_file, "r") as f:
        bob_data = json.loads(f.readline().strip())
        assert bob_data["provider"] == "anthropic"


# ==================== TEST 3: TEAM AGGREGATE DASHBOARD ====================
def test_team_aggregate_dashboard(multi_user_data):
    """Test aggregate dashboard loads all user files (TL requirement #3)."""
    tracker = CostTracker(user_id="admin", data_dir=multi_user_data)
    
    # Load all user costs
    all_costs = tracker._load_all_user_costs()
    
    # Verify aggregation
    assert len(all_costs) == 3
    total_cost = sum(entry["cost"] for entry in all_costs)
    assert total_cost == pytest.approx(0.50, rel=1e-9)  # 0.15 + 0.25 + 0.10
    
    # Verify all users represented
    providers = [entry["provider"] for entry in all_costs]
    assert "openai" in providers
    assert "anthropic" in providers
    assert "google" in providers


# ==================== TEST 4: BUDGET TRACKING CONFIGURATION ====================
def test_budget_tracking_configuration():
    """Test budget configuration (TL requirement #4)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = CostTracker(
            user_id="test",
            data_dir=temp_dir,
            monthly_budget_limit=500.0,
            total_budget_limit=2000.0,
        )
        
        assert tracker.monthly_budget_limit == 500.0
        assert tracker.total_budget_limit == 2000.0
        assert tracker.month_1_2_threshold == 800.0  # Fixed threshold


# ==================== TEST 5: DATA PERSISTENCE ACROSS SESSIONS ====================
def test_data_persistence_across_sessions(temp_data_dir):
    """Test costs are saved and loaded correctly."""
    # Session 1: Create and log
    tracker1 = CostTracker(user_id="persist_test", data_dir=temp_data_dir)
    tracker1.log_cost("openai", "gpt-4o", 1234, 567, cost=0.123456)
    
    # Session 2: Load same data
    tracker2 = CostTracker(user_id="persist_test", data_dir=temp_data_dir, load_existing=True)
    
    # Verify data preserved
    assert len(tracker2.costs) == 1
    entry = tracker2.costs[0]
    assert entry["provider"] == "openai"
    assert entry["cost"] == pytest.approx(0.123456, rel=1e-9)


# ==================== TEST 6: RESET FUNCTIONALITY ====================
def test_reset_functionality_clears_data(temp_data_dir):
    """Test reset deletes user file and clears memory."""
    tracker = CostTracker(user_id="reset_test", data_dir=temp_data_dir)
    
    # Add data
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    jsonl_path = Path(temp_data_dir) / "costs_reset_test.jsonl"
    
    # Verify file exists
    assert jsonl_path.exists()
    assert len(tracker.costs) == 1
    
    # Mock user confirmation
    with patch('builtins.input', return_value='y'):
        tracker.reset()
    
    # Verify cleanup
    assert not jsonl_path.exists()
    assert len(tracker.costs) == 0


# ==================== TEST 7: AUTO-LOG INTEGRATION ====================
def test_auto_log_integration_with_llm_client(fresh_tracker):
    """Test automatic logging from LLM client responses."""
    # Mock LLM response (Anthropic format)
    mock_response = {
        "id": "msg_123",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
        },
    }
    
    # Auto-log cost
    cost = fresh_tracker.auto_log_from_llm_client(
        provider="anthropic",
        model="claude-sonnet-4.5",
        response=mock_response,
    )
    
    # Verify auto-logging
    assert cost > 0
    assert len(fresh_tracker.costs) == 1
    
    entry = fresh_tracker.costs[0]
    assert entry["input_tokens"] == 1000
    assert entry["output_tokens"] == 500
    assert entry["metadata"]["auto_logged"] is True
    assert entry["metadata"]["response_id"] == "msg_123"


# ==================== TEST 8: CSV EXPORT FUNCTIONALITY ====================
def test_csv_export_works(temp_data_dir):
    """Test CSV export creates valid file with data."""
    tracker = CostTracker(user_id="export_test", data_dir=temp_data_dir)
    
    # Add test data
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.15)
    tracker.log_cost("anthropic", "claude-sonnet-4.5", 800, 400, cost=0.25)
    
    # Export
    csv_path = Path(temp_data_dir) / "export.csv"
    tracker.export_csv(str(csv_path))
    
    # Verify
    assert csv_path.exists()
    
    with open(csv_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 3  # Header + 2 data rows
        assert "timestamp" in lines[0]
        assert "provider" in lines[0]
        assert "cost" in lines[0]


# ==================== TEST 9: SINGLETON PATTERN ====================
def test_singleton_pattern_ensures_single_instance():
    """Test get_tracker() returns same instance."""
    # Clear singleton
    import pipeline_a_scenarios.utils.cost_tracker as ct_module
    ct_module._default_tracker = None
    
    # First call creates
    tracker1 = get_tracker(user_id="test_user")
    assert tracker1 is not None
    
    # Second call returns same instance
    tracker2 = get_tracker()
    assert tracker2 is tracker1
    
    # Third call with different user_id still returns same
    tracker3 = get_tracker(user_id="different_user")
    assert tracker3 is tracker1


# ==================== TEST 10: ERROR HANDLING ====================
def test_error_handling_for_corrupted_data(temp_data_dir):
    """Test graceful handling of corrupted JSONL data."""
    # Create corrupted JSONL file
    jsonl_path = Path(temp_data_dir) / "costs_test.jsonl"
    with open(jsonl_path, "w") as f:
        f.write('{"valid": "json"}\n')
        f.write('{ invalid json }\n')  # Corrupted line
        f.write('{"another": "valid"}\n')
    
    # Should not crash, should load valid lines
    tracker = CostTracker(user_id="test", data_dir=temp_data_dir, load_existing=True)
    
    # Should load 2 valid entries, skip corrupted
    assert len(tracker.costs) == 2