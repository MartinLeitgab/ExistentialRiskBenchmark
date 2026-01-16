"""Tests for the CostTracker utility module."""

from pipeline_a_scenarios.utils.cost_tracker import CostTracker


def test_tracker_initialization():
    """Test cost tracker initializes correctly."""
    tracker = CostTracker()
    assert tracker.get_total_cost() == 0.0
    assert len(tracker.get_all_costs()) == 0


def test_log_cost():
    """Test logging costs."""
    tracker = CostTracker()
    tracker.log_cost(
        provider="anthropic",
        model="claude-sonnet-4",
        input_tokens=1000,
        output_tokens=500,
        cost=0.015,
    )

    assert tracker.get_total_cost() == 0.015
    costs = tracker.get_all_costs()
    assert len(costs) == 1
    assert costs[0]["provider"] == "anthropic"


def test_cost_calculation_anthropic():
    """Test Anthropic cost calculation."""
    tracker = CostTracker()
    cost = tracker.calculate_cost(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        input_tokens=1000,
        output_tokens=500,
    )
    # $3 per million input, $15 per million output
    expected = (1000 * 3 + 500 * 15) / 1_000_000
    assert abs(cost - expected) < 0.0001


def test_cost_calculation_openai():
    """Test OpenAI cost calculation."""
    tracker = CostTracker()
    cost = tracker.calculate_cost(
        provider="openai", model="gpt-5.2", input_tokens=1000, output_tokens=500
    )
    assert cost > 0


def test_budget_alert():
    """Test budget alert at 80%."""
    tracker = CostTracker(budget_limit=100.0)

    # Should not alert at 70%
    tracker.log_cost("anthropic", "claude-sonnet-4", 1000, 500, 70.0)
    assert not tracker.check_budget_alert()

    # Should alert at 85%
    tracker.log_cost("anthropic", "claude-sonnet-4", 1000, 500, 15.0)
    assert tracker.check_budget_alert()


def test_get_costs_by_provider():
    """Test filtering costs by provider."""
    tracker = CostTracker()
    tracker.log_cost("anthropic", "claude-sonnet-4", 1000, 500, 10.0)
    tracker.log_cost("openai", "gpt-5.2", 1000, 500, 15.0)
    tracker.log_cost("anthropic", "claude-sonnet-4", 500, 250, 5.0)

    anthropic_costs = tracker.get_costs_by_provider("anthropic")
    assert len(anthropic_costs) == 2
    assert sum(c["cost"] for c in anthropic_costs) == 15.0
