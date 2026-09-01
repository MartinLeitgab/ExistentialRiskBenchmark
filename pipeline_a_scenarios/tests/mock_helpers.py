"""Shared helpers for unit-test mocks."""

from typing import Any
from unittest.mock import Mock


def configure_mock_cost_tracker(mock: Mock, total_cost: float = 0.0) -> Mock:
    """Align CostTracker mocks with production API used by prompt_validation."""
    mock.get_total_cost.return_value = total_cost
    mock.get_cost_breakdown_by_model.return_value = {}
    mock.get_provider_breakdown.return_value = {}
    mock.get_batch_stats.return_value = {
        "total_batch_calls": 0,
        "total_batch_cost": 0.0,
        "total_batch_savings": 0.0,
        "avg_batch_size": 0,
    }
    mock.get_summary.return_value = {"total_cost": total_cost}
    return mock
