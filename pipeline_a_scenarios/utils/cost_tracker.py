"""Track and manage API usage costs."""

from typing import List, Dict, Optional


class CostTracker:
    """Track API usage costs across multiple providers."""

    def __init__(self, budget_limit: Optional[float] = None):
        self.costs: List[Dict] = []
        self.budget_limit = budget_limit

    def log_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Log a cost entry."""
        entry = {
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }
        self.costs.append(entry)

    def calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost based on provider and model."""
        # Pricing rates (USD per million tokens)
        pricing = {
            "anthropic": {
                "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
                "claude-4.5": {"input": 3.0, "output": 15.0},
            },
            "openai": {
                "gpt-5.2": {"input": 50.0, "output": 150.0},
                "gpt-4": {"input": 30.0, "output": 60.0},
            },
        }

        if provider not in pricing or model not in pricing[provider]:
            raise ValueError(f"Unknown pricing for {provider}/{model}")

        rates = pricing[provider][model]
        input_cost = (input_tokens * rates["input"]) / 1_000_000
        output_cost = (output_tokens * rates["output"]) / 1_000_000

        return input_cost + output_cost

    def get_total_cost(self) -> float:
        """Get total cost across all entries."""
        return sum(entry["cost"] for entry in self.costs)

    def get_all_costs(self) -> List[Dict]:
        """Get all cost entries."""
        return self.costs.copy()

    def get_costs_by_provider(self, provider: str) -> List[Dict]:
        """Get costs for a specific provider."""
        return [entry for entry in self.costs if entry["provider"] == provider]

    def check_budget_alert(self) -> bool:
        """Check if we've reached 80% of budget limit."""
        if self.budget_limit is None:
            return False
        return self.get_total_cost() >= 0.8 * self.budget_limit
