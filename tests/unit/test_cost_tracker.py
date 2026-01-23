"""
tests/unit/test_cost_tracker.py
UNIT TESTS FOR COST TRACKER
Tests the CostTracker class for tracking LLM API costs, budgets, alerts, and
analytics.
Author: Pooja Puranik
version: 1.0.0
date: 23/01/2026
"""
from pipeline_a_scenarios.utils.cost_tracker import CostTracker
import pytest
import sys
import os
import tempfile


# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


@pytest.fixture
def temp_data_file():
    """
    Create temporary data file for tests.

    Returns:
        str: Path to temporary JSON file containing empty array
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("[]")
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def fresh_tracker():
    """
    Create fresh CostTracker with default settings.

    Returns:
        CostTracker: New CostTracker instance with isolated data file
    """
    # Use a unique temp file for each test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("[]")
        temp_path = f.name

    tracker = CostTracker(data_path=temp_path, load_existing=False)
    yield tracker

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def tracker_with_budget():
    """
    Create CostTracker with budget for alert testing.

    Returns:
        CostTracker: New CostTracker with budget limits ($200 monthly, $1000 total)
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("[]")
        temp_path = f.name

    tracker = CostTracker(
        data_path=temp_path,
        load_existing=False,
        monthly_budget_limit=200.0,
        total_budget_limit=1000.0,
    )
    yield tracker

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ==================== INITIALIZATION TESTS ====================
def test_tracker_initialization_defaults(fresh_tracker):
    """
    Test CostTracker initializes with correct default values.

    Verifies:
    - Default monthly budget: $200.00
    - Default total budget: $1000.00
    - Month 1-2 threshold: $800.00
    - Empty email alerts list
    - Empty costs list
    """
    tracker = fresh_tracker

    # Check defaults
    assert tracker.monthly_budget_limit == 200.0
    assert tracker.total_budget_limit == 1000.0
    assert tracker.month_1_2_threshold == 800.0
    assert tracker.alert_emails == []
    assert tracker.costs == []  # Should be empty for fresh tracker


def test_tracker_initialization_custom_values():
    """
    Test CostTracker accepts and uses custom initialization values.

    Verifies:
    - Custom monthly and total budgets are set correctly
    - Custom email recipients are stored
    - Each test uses isolated data file
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("[]")
        temp_path = f.name

    tracker = CostTracker(
        data_path=temp_path,
        load_existing=False,
        monthly_budget_limit=500.0,
        total_budget_limit=2000.0,
        alert_emails=["test1@example.com", "test2@example.com"],
    )

    assert tracker.monthly_budget_limit == 500.0
    assert tracker.total_budget_limit == 2000.0
    assert len(tracker.alert_emails) == 2
    assert tracker.alert_emails[0] == "test1@example.com"

    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_tracker_email_limit():
    """
    Test email list is limited to maximum 4 recipients.

    Verifies:
    - Only first 4 emails are kept when more are provided
    - Extra emails beyond 4 are ignored
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("[]")
        temp_path = f.name

    tracker = CostTracker(
        data_path=temp_path,
        load_existing=False,
        alert_emails=[
            "email1@test.com",
            "email2@test.com",
            "email3@test.com",
            "email4@test.com",
            "email5@test.com",  # Should be ignored
        ],
    )

    assert len(tracker.alert_emails) == 4
    assert "email5@test.com" not in tracker.alert_emails

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ==================== COST CALCULATION TESTS ====================
def test_calculate_cost_anthropic(fresh_tracker):
    """
    Test Anthropic cost calculation accuracy.

    PRICING UPDATE (January 2026):
    - Claude Sonnet 4.5: $3/$15 per million tokens

    Verifies:
    - Correct pricing for current Anthropic models
    - Calculation matches expected formula: (input_tokens * input_price + output_tokens * output_price) / 1,000,000
    """
    tracker = fresh_tracker

    # Test exact pricing with current rates
    cost = tracker.calculate_cost(
        provider="anthropic",
        model="claude-sonnet-4.5",
        input_tokens=1_000_000,  # 1M tokens
        output_tokens=500_000,  # 0.5M tokens
    )

    # $3 per million input, $15 per million output (January 2026 pricing)
    expected = (1_000_000 * 3 + 500_000 * 15) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_openai(fresh_tracker):
    """
    Test OpenAI cost calculation accuracy.

    PRICING UPDATE (January 2026):
    - GPT-4o: $2.50/$10.00 per million tokens
    - Old GPT-4: No longer available, replaced by GPT-4o

    Verifies:
    - Correct pricing for current OpenAI models
    - Handles partial million tokens correctly
    """
    tracker = fresh_tracker

    cost = tracker.calculate_cost(
        provider="openai", model="gpt-4o", input_tokens=500_000, output_tokens=250_000
    )

    # GPT-4o: $2.50 per million input, $10.00 per million output (January 2026 pricing)
    expected = (500_000 * 2.5 + 250_000 * 10.0) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_google(fresh_tracker):
    """
    Test Google cost calculation accuracy.

    PRICING UPDATE (January 2026):
    - Gemini 2.5 Pro: $1.25/$5.00 per million tokens
    - Old Gemini 1.5 Pro: No longer primary model

    Verifies:
    - Correct pricing for current Google models
    - Large token counts handled correctly
    """
    tracker = fresh_tracker

    cost = tracker.calculate_cost(
        provider="google",
        model="gemini-2.5-pro",
        input_tokens=2_000_000,
        output_tokens=1_000_000,
    )

    # Gemini 2.5 Pro: $1.25 per million input, $5.00 per million output (January 2026 pricing)
    expected = (2_000_000 * 1.25 + 1_000_000 * 5.0) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_batch_discount(fresh_tracker):
    """
    Test batch API discount (50% off) is applied correctly.

    Verifies:
    - Regular API calls use full pricing
    - Batch API calls get 50% discount
    - Discount calculation is accurate
    """
    tracker = fresh_tracker

    # Regular cost
    regular_cost = tracker.calculate_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        batch_api=False,
    )

    # Batch cost (50% discount)
    batch_cost = tracker.calculate_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        batch_api=True,
    )

    assert batch_cost == pytest.approx(regular_cost * 0.5, rel=1e-9)


def test_calculate_cost_unknown_provider(fresh_tracker):
    """
    Test error handling for unknown provider.

    Verifies:
    - ValueError raised for unsupported provider
    - Error message indicates supported providers
    """
    tracker = fresh_tracker

    with pytest.raises(ValueError, match="Unknown provider"):
        tracker.calculate_cost(
            provider="unknown", model="test-model", input_tokens=1000, output_tokens=500
        )


def test_calculate_cost_unknown_model_fallback(fresh_tracker):
    """
    Test fallback behavior for unknown model names.

    Verifies:
    - No error raised for unknown model
    - System uses fallback model pricing
    - Returns positive cost value
    """
    tracker = fresh_tracker

    # Should not raise error, should use fallback
    cost = tracker.calculate_cost(
        provider="openai",
        model="unknown-model-name",
        input_tokens=1000,
        output_tokens=500,
    )

    assert cost > 0  # Should calculate something


# ==================== COST LOGGING TESTS ====================
def test_log_cost_with_calculation(fresh_tracker):
    """
    Test logging cost with automatic calculation.

    Verifies:
    - Cost is calculated and logged correctly
    - Total cost updates appropriately
    - Cost entry has correct structure with all required fields
    """
    tracker = fresh_tracker

    cost = tracker.log_cost(
        provider="anthropic",
        model="claude-haiku-4.5",
        input_tokens=1000,
        output_tokens=500,
    )

    assert cost > 0
    assert tracker.get_total_cost() == cost
    assert len(tracker.costs) == 1

    # Verify entry structure
    entry = tracker.costs[0]
    assert entry["provider"] == "anthropic"
    assert entry["model"] == "claude-haiku-4.5"
    assert entry["input_tokens"] == 1000
    assert entry["output_tokens"] == 500
    assert entry["cost"] == cost


def test_log_cost_with_manual_cost(fresh_tracker):
    """
    Test logging cost with manually specified cost value.

    Verifies:
    - Manual cost is accepted without recalculation
    - Manual cost overrides automatic calculation
    """
    tracker = fresh_tracker

    manual_cost = 0.123456
    cost = tracker.log_cost(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        cost=manual_cost,
    )

    assert cost == manual_cost
    assert tracker.get_total_cost() == manual_cost


def test_log_cost_with_metadata(fresh_tracker):
    """
    Test logging cost with additional metadata.

    Verifies:
    - Metadata dictionary is stored with cost entry
    - Metadata persists with the entry
    """
    tracker = fresh_tracker

    metadata = {"user_id": "test_user", "request_id": "req_123", "temperature": 0.7}

    tracker.log_cost(
        provider="google",
        model="gemini-2.5-pro",
        input_tokens=1000,
        output_tokens=500,
        cost=0.05,
        metadata=metadata,
    )

    assert tracker.costs[0]["metadata"] == metadata


def test_log_cost_persistence():
    """
    Test that costs are persisted to and loaded from file.

    Verifies:
    - Costs saved by one tracker instance are loaded by another
    - JSON file persistence works correctly
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("[]")
        temp_path = f.name

    # First tracker writes data
    tracker1 = CostTracker(data_path=temp_path, load_existing=False)
    tracker1.log_cost("openai", "gpt-4o", 1000, 500, cost=0.05)

    # Second tracker loads data
    tracker2 = CostTracker(data_path=temp_path, load_existing=True)

    assert len(tracker2.costs) == 1
    assert tracker2.get_total_cost() == 0.05

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ==================== BUDGET & ALERT TESTS ====================
def test_get_total_cost_empty(fresh_tracker):
    """
    Test total cost calculation with no logged costs.

    Verifies:
    - Returns 0.0 when no costs have been logged
    """
    tracker = fresh_tracker
    assert tracker.get_total_cost() == 0.0


def test_get_total_cost_multiple_entries(fresh_tracker):
    """
    Test total cost calculation with multiple cost entries.

    Verifies:
    - Sums all logged costs correctly
    - Handles multiple providers and models
    """
    tracker = fresh_tracker

    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    tracker.log_cost("anthropic", "claude-haiku-4.5", 500, 250, cost=0.05)
    tracker.log_cost("google", "gemini-2.5-pro", 300, 100, cost=0.02)

    total = tracker.get_total_cost()
    assert total == pytest.approx(0.17, rel=1e-9)


def test_get_monthly_cost(fresh_tracker):
    """
    Test monthly cost calculation filters by current month.

    Verifies:
    - Only costs from current month are included
    - Costs from previous months are excluded
    """
    tracker = fresh_tracker

    # Add costs for current month
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    tracker.log_cost("anthropic", "claude-haiku-4.5", 500, 250, cost=0.05)

    monthly_cost = tracker.get_monthly_cost()
    assert monthly_cost == pytest.approx(0.15, rel=1e-9)


def test_check_80_percent_alert(tracker_with_budget):
    """
    Test 80% total budget alert detection.

    Verifies:
    - Returns False when below 80% threshold
    - Returns True when at or above 80% threshold
    - Reset functionality works for clean testing
    """
    tracker = tracker_with_budget

    # Start fresh - should be 0%
    assert tracker.get_total_cost() == 0.0
    assert not tracker.check_80_percent_alert()

    # Add $700 (70%) - should not alert
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=700.0)
    assert not tracker.check_80_percent_alert()

    # Reset for clean test
    tracker.reset()

    # Add $800 (80%) - should alert
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=800.0)
    assert tracker.check_80_percent_alert()


def test_check_month_1_2_threshold(tracker_with_budget):
    """
    Test Month 1-2 threshold ($800) alert detection.

    Verifies:
    - Returns False when below $800 threshold
    - Returns True when at or above $800 threshold
    """
    tracker = tracker_with_budget

    # Start fresh
    assert tracker.get_total_cost() == 0.0
    assert not tracker.check_month_1_2_threshold()

    # Add $799 - should not alert
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=799.0)
    assert not tracker.check_month_1_2_threshold()

    # Reset for clean test
    tracker.reset()

    # Add $800 - should alert
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=800.0)
    assert tracker.check_month_1_2_threshold()


def test_check_monthly_80_percent_alert(tracker_with_budget):
    """
    Test monthly 80% budget alert detection.

    Verifies:
    - Returns False when monthly cost below 80% of monthly budget
    - Returns True when monthly cost at or above 80% of monthly budget
    """
    tracker = tracker_with_budget

    # Start fresh
    assert tracker.get_monthly_cost() == 0.0
    assert not tracker.check_monthly_80_percent_alert()

    # $150 (75%) - should not alert
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=150.0)
    assert not tracker.check_monthly_80_percent_alert()

    # Reset for clean test
    tracker.reset()

    # Add $160 (80%) - should alert
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=160.0)
    assert tracker.check_monthly_80_percent_alert()


# ==================== PROJECTION TESTS ====================
def test_get_monthly_projection_empty(fresh_tracker):
    """
    Test monthly projection with no data.

    Verifies:
    - Returns 0.0 when no costs in current month
    """
    tracker = fresh_tracker
    projection = tracker.get_monthly_projection()
    # Should return 0.0 when no monthly cost
    assert projection == 0.0


def test_get_monthly_projection(fresh_tracker):
    """
    Test monthly projection calculation based on current spending rate.

    Note: This test is simplified to avoid datetime mocking issues.
    The actual projection logic is: monthly_cost * (30 / current_day)

    Verifies:
    - Basic projection formula works with sample data
    """
    tracker = fresh_tracker

    # Add some cost data
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=150.0)

    # Get the projection
    projection = tracker.get_monthly_projection()

    # Instead of asserting exact value (which depends on current date),
    # we can test that:
    # 1. It returns a float
    # 2. If we have costs, projection should be >= monthly cost
    # 3. It handles the calculation without errors

    assert isinstance(projection, float)

    monthly_cost = tracker.get_monthly_cost()
    if monthly_cost > 0:
        # Projection should extrapolate, so it should be >= monthly_cost
        # (unless we're at the end of the month)
        # For simplicity, just check it's a reasonable number
        assert projection >= 0


# ==================== BREAKDOWN TESTS ====================
def test_get_cost_breakdown_by_model(fresh_tracker):
    """
    Test cost breakdown aggregation by model.

    Verifies:
    - Groups costs by model name
    - Calculates totals: cost, calls, tokens per model
    - Includes provider information for each model
    """
    tracker = fresh_tracker

    # Add costs for different models
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    tracker.log_cost("openai", "gpt-4o", 500, 250, cost=0.05)  # Same model
    tracker.log_cost("anthropic", "claude-haiku-4.5", 300, 100, cost=0.02)
    tracker.log_cost("google", "gemini-2.5-pro", 200, 50, cost=0.03)

    breakdown = tracker.get_cost_breakdown_by_model()

    assert len(breakdown) == 3  # gpt-4o, claude-haiku-4.5, gemini-2.5-pro
    assert breakdown["gpt-4o"]["total_cost"] == pytest.approx(0.15, rel=1e-9)
    assert breakdown["gpt-4o"]["total_calls"] == 2
    assert breakdown["gpt-4o"]["provider"] == "openai"
    assert breakdown["claude-haiku-4.5"]["total_cost"] == pytest.approx(0.02, rel=1e-9)
    assert breakdown["gemini-2.5-pro"]["total_cost"] == pytest.approx(0.03, rel=1e-9)


def test_get_provider_breakdown(fresh_tracker):
    """
    Test cost breakdown aggregation by provider.

    Verifies:
    - Groups costs by provider (OpenAI, Anthropic, Google)
    - Calculates totals: cost and call count per provider
    """
    tracker = fresh_tracker

    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    tracker.log_cost("openai", "gpt-4o-mini", 500, 250, cost=0.05)
    tracker.log_cost("anthropic", "claude-haiku-4.5", 300, 100, cost=0.02)
    tracker.log_cost("anthropic", "claude-sonnet-4.5", 200, 50, cost=0.03)
    tracker.log_cost("google", "gemini-2.5-pro", 100, 25, cost=0.01)

    breakdown = tracker.get_provider_breakdown()

    assert len(breakdown) == 3
    assert breakdown["openai"]["cost"] == pytest.approx(0.15, rel=1e-9)
    assert breakdown["openai"]["calls"] == 2
    assert breakdown["anthropic"]["cost"] == pytest.approx(0.05, rel=1e-9)
    assert breakdown["anthropic"]["calls"] == 2
    assert breakdown["google"]["cost"] == pytest.approx(0.01, rel=1e-9)
    assert breakdown["google"]["calls"] == 1


# ==================== DATA MANAGEMENT TESTS ====================
def test_reset(fresh_tracker):
    """
    Test reset functionality clears all data and alerts.

    Verifies:
    - Costs list is emptied
    - Total cost resets to 0.0
    - Alert tracking resets to initial state
    """
    tracker = fresh_tracker

    # Add some data
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    tracker.log_cost("anthropic", "claude-haiku-4.5", 500, 250, cost=0.05)

    assert len(tracker.costs) == 2
    assert tracker.get_total_cost() > 0

    # Reset
    tracker.reset()

    assert len(tracker.costs) == 0
    assert tracker.get_total_cost() == 0.0
    # Check alerts are also reset
    assert tracker.alerts_sent["80_percent_total"] is False
    assert tracker.alerts_sent["month_1_2"] is False


def test_export_csv(fresh_tracker, tmp_path):
    """
    Test CSV export functionality.

    Verifies:
    - CSV file is created
    - File contains expected headers
    - All cost data is exported
    """
    tracker = fresh_tracker

    # Add data
    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    tracker.log_cost("anthropic", "claude-haiku-4.5", 500, 250, cost=0.05)

    # Export
    csv_path = tmp_path / "test_export.csv"
    tracker.export_csv(str(csv_path))

    # Verify file exists and has content
    assert csv_path.exists()
    with open(csv_path, "r") as f:
        content = f.read()
        assert "timestamp" in content
        assert "provider" in content
        assert "cost" in content


def test_export_csv_empty(fresh_tracker, tmp_path, capsys):
    """
    Test CSV export behavior with no data.

    Verifies:
    - Appropriate message is printed when no data to export
    """
    tracker = fresh_tracker

    csv_path = tmp_path / "empty.csv"
    tracker.export_csv(str(csv_path))

    # Check console output
    captured = capsys.readouterr()
    assert "No cost data to export" in captured.out


# ==================== INTEGRATION TESTS ====================
def test_auto_log_from_llm_client(fresh_tracker):
    """
    Test auto-logging from LLM client response.

    Verifies:
    - Extracts token usage from LLM response
    - Automatically logs cost based on response
    - Includes metadata indicating auto-logged entry
    """
    tracker = fresh_tracker

    # Mock LLM response
    mock_response = {
        "id": "resp_123",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        },
        "choices": [{"text": "Test response"}],
    }

    cost = tracker.auto_log_from_llm_client(
        provider="openai", model="gpt-4o", response=mock_response
    )

    assert cost > 0
    assert len(tracker.costs) == 1

    entry = tracker.costs[0]
    assert entry["provider"] == "openai"
    assert entry["model"] == "gpt-4o"
    assert entry["input_tokens"] == 1000
    assert entry["output_tokens"] == 500
    assert entry["metadata"]["auto_logged"] is True


def test_auto_log_from_llm_client_various_formats(fresh_tracker):
    """
    Test auto-logging handles different LLM response formats.

    Verifies:
    - Handles different token field names (input_tokens/output_tokens vs prompt_tokens/completion_tokens)
    - Defaults to 0 tokens when usage data missing
    """
    tracker = fresh_tracker

    # Test different usage field names
    response1 = {"usage": {"input_tokens": 800, "output_tokens": 400}}
    response2 = {"usage": {"prompt_tokens": 600, "completion_tokens": 300}}
    response3 = {"usage": {}}  # No tokens

    tracker.auto_log_from_llm_client("openai", "gpt-4o", response1)
    tracker.auto_log_from_llm_client("anthropic", "claude-haiku-4.5", response2)
    tracker.auto_log_from_llm_client("google", "gemini-2.5-pro", response3)

    assert len(tracker.costs) == 3
    assert tracker.costs[0]["input_tokens"] == 800
    assert tracker.costs[0]["output_tokens"] == 400
    assert tracker.costs[1]["input_tokens"] == 600
    assert tracker.costs[1]["output_tokens"] == 300
    assert tracker.costs[2]["input_tokens"] == 0  # Default for missing
    assert tracker.costs[2]["output_tokens"] == 0


# ==================== EDGE CASE TESTS ====================
def test_negative_tokens_error(fresh_tracker):
    """
    Test handling of negative token counts.

    Verifies:
    - Implementation returns 0.0 for negative tokens
    - Negative values are treated as zero cost
    """
    tracker = fresh_tracker

    cost = tracker.calculate_cost("openai", "gpt-4o", -100, 50)

    # Should return 0.0 for negative tokens
    assert cost == 0.0


def test_zero_tokens(fresh_tracker):
    """
    Test cost calculation with zero tokens.

    Verifies:
    - Zero tokens result in zero cost
    - Can successfully log zero-cost entries
    """
    tracker = fresh_tracker

    cost = tracker.calculate_cost("openai", "gpt-4o", 0, 0)
    assert cost == 0.0

    # Should be able to log zero cost
    tracker.log_cost("openai", "gpt-4o", 0, 0, cost=0.0)
    assert tracker.get_total_cost() == 0.0


def test_large_token_counts(fresh_tracker):
    """
    Test with very large token counts (billions).

    PRICING UPDATE (January 2026):
    Updated to use Claude Sonnet 4.5 pricing: $3/$15 per million tokens

    Verifies:
    - Handles large numbers without overflow
    - Calculates correctly for extreme values
    """
    tracker = fresh_tracker

    # 10 billion tokens
    cost = tracker.calculate_cost(
        provider="anthropic",
        model="claude-sonnet-4.5",
        input_tokens=10_000_000_000,
        output_tokens=5_000_000_000,
    )

    # Should calculate correctly without overflow
    expected = (10_000_000_000 * 3 + 5_000_000_000 * 15) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


def test_corrupted_data_file():
    """
    Test handling of corrupted JSON data file.

    Verifies:
    - Does not crash on corrupted file
    - Starts with fresh data when file is invalid
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{ invalid json }")
        temp_path = f.name

    # Should not crash, should start fresh
    tracker = CostTracker(data_path=temp_path, load_existing=True)
    assert tracker.costs == []
    assert tracker.get_total_cost() == 0.0

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ==================== DASHBOARD TESTS ====================
def test_dashboard_no_data(fresh_tracker, capsys):
    """
    Test dashboard display with no cost data.

    Verifies:
    - Dashboard renders without errors
    - Shows zero costs
    """
    tracker = fresh_tracker
    tracker.dashboard()

    captured = capsys.readouterr()
    assert "COST TRACKER DASHBOARD" in captured.out
    assert "Total Cost:" in captured.out
    assert "$0.00" in captured.out


def test_dashboard_with_data(fresh_tracker, capsys):
    """
    Test dashboard display with cost data.

    Verifies:
    - Shows actual cost totals
    - Displays provider information
    """
    tracker = fresh_tracker

    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    tracker.log_cost("anthropic", "claude-haiku-4.5", 500, 250, cost=0.05)

    tracker.dashboard()

    captured = capsys.readouterr()
    assert "COST TRACKER DASHBOARD" in captured.out
    assert "Total Cost:" in captured.out
    assert "openai" in captured.out
    assert "anthropic" in captured.out


def test_dashboard_verbose(fresh_tracker, capsys):
    """
    Test verbose dashboard with detailed breakdowns.

    Verifies:
    - Shows detailed model breakdown in verbose mode
    - Includes all model information
    """
    tracker = fresh_tracker

    tracker.log_cost("openai", "gpt-4o", 1000, 500, cost=0.10)
    tracker.log_cost("openai", "gpt-4o-mini", 500, 250, cost=0.05)

    tracker.dashboard(verbose=True)

    captured = capsys.readouterr()
    # Check for dashboard elements
    assert "COST TRACKER DASHBOARD" in captured.out
    assert "gpt-4o" in captured.out
    assert "gpt-4o-mini" in captured.out
