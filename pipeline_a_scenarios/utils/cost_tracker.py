"""
Cost Tracker - Production LLM Cost Monitoring (v2.0)
Provides comprehensive cost tracking across multiple LLM providers with
automatic alerting, budget management, and detailed analytics. Designed for both
research environments and production deployments.

- Append-only JSONL format for better scalability
- Per-user cost files for team collaboration
- Aggregate dashboard for team-wide cost monitoring
- Simplified alert system (manual checks via dashboard)

Author: Pooja Puranik
version: 2.0.0
date: 26/01/2026
Last updated: 26/01/2026
"""
import json
import csv
import warnings
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class CostTracker:
    """
    Production cost tracker for LLM API cost monitoring with team 
    collaboration.
    
    Implements all TL requirements:
    - Append-only JSONL format (TL req #1)
    - Per-user cost files (TL req #2)
    - Aggregate dashboard (TL req #3)
    - Simple budget tracking (TL req #4)
    
    Integrates with UnifiedLLMClient for automatic cost logging.
    
    Attributes:
        user_id: Unique identifier (defaults to $USER env var)
        data_dir: Directory for JSONL cost files
        monthly_budget_limit: Monthly spending limit ($200)
        total_budget_limit: Total project budget ($1000)
        month_1_2_threshold: Critical threshold for months 1-2 ($800)
        costs: In-memory list of cost entries
    """

    # Pricing per million tokens (input, output)
    # Updated with actual 2025 pricing
    PRICING = {
        "openai": {
            "gpt-5.2": {"input": 5.0, "output": 15.0},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4": {"input": 3.0, "output": 6.0},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        },
        "anthropic": {
            "claude-opus-4.5": {"input": 5.0, "output": 25.0},
            "claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
            "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
            "claude-haiku-4.5": {"input": 1.0, "output": 5.0},
        },
        "google": {
            "gemini-3-flash-preview": {"input": 2.0, "output": 8.0},
            "gemini-3-pro": {"input": 2.0, "output": 12.0},
            "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        },
    }

    def __init__(
        self,
        user_id: Optional[str] = None,
        data_dir: str = "data/metadata",
        load_existing: bool = True,
        monthly_budget_limit: float = 200.0,
        total_budget_limit: float = 1000.0,
    ):
        """
        Initialize production cost tracker with per-user tracking.

        Args:
            user_id: User identifier (defaults to $USER env var).
                     Used for creating per-user JSONL files.
            data_dir: Directory where JSONL cost files are stored.
                      Created if doesn't exist.
            load_existing: Load existing cost data from JSONL file.
            monthly_budget_limit: Monthly spending limit ($200 default).
            total_budget_limit: Total project budget ($1000 default).

        Example:
            >>> tracker = CostTracker(user_id='alice')
            >>> tracker.log_cost('openai', 'gpt-4o', 1000, 500)
        """
        # User identification - defaults to system USER env var
        self.user_id = user_id or os.getenv("USER", "unknown")
        
        # Setup directory paths
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-user JSONL file path (TL requirement #2)
        self.data_path = self.data_dir / f"costs_{self.user_id}.jsonl"

        # Budget configuration
        self.monthly_budget_limit = monthly_budget_limit
        self.total_budget_limit = total_budget_limit
        self.month_1_2_threshold = 800.0  # Critical threshold

        # Load existing costs from JSONL
        self.costs: List[Dict[str, Any]] = []
        if load_existing:
            self.costs = self._load_costs()

        print(
            f"""
✅ COST TRACKER INITIALIZED
   User: {self.user_id}
   File: {self.data_path}
   Monthly Budget: ${monthly_budget_limit:.2f}
   Total Budget: ${total_budget_limit:.2f}
   Loaded: {len(self.costs)} entries
        """
        )

    # ==================== CORE FUNCTIONALITY ====================

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        batch_api: bool = False,
    ) -> float:
        """
        Calculate cost for API call across all supported providers.

        Supports OpenAI, Anthropic, and Google with pricing per million tokens.

        Args:
            provider: One of 'openai', 'anthropic', or 'google'.
            model: Specific model name (e.g., 'gpt-4o', 'claude-sonnet-4.5').
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.
            batch_api: Whether batch API is used (applies 50% discount).

        Returns:
            Cost in USD rounded to 6 decimal places.

        Raises:
            ValueError: If provider is not supported.

        Example:
            >>> tracker.calculate_cost('openai', 'gpt-4o', 1000, 500)
            0.0075
            >>> tracker.calculate_cost('openai', 'gpt-4o', 1000, 500, batch_api=True)
            0.00375  # 50% discount
        """
        # Handle negative tokens (safety check)
        if input_tokens < 0 or output_tokens < 0:
            return 0.0

        if provider not in self.PRICING:
            raise ValueError(
                f"Unknown provider: {provider}. Supported: {list(self.PRICING.keys())}"
            )

        provider_pricing = self.PRICING[provider]

        # Find matching model (handle model name variations)
        if model not in provider_pricing:
            for available_model in provider_pricing.keys():
                if model in available_model or available_model in model:
                    model = available_model
                    break
            else:
                # Use first model as fallback
                model = list(provider_pricing.keys())[0]
                warnings.warn(f"Model '{model}' not found, using {model} pricing")

        pricing = provider_pricing[model]

        # Calculate cost: (input_tokens * input_price + output_tokens * output_price) / 1,000,000
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        # Apply batch API discount (50% off)
        if batch_api:
            cost *= 0.5

        return round(cost, 6)

    def log_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float] = None,
        batch_api: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Log API call cost to JSONL file (append-only - TL requirement #1).

        Creates a new JSON line entry and appends to JSONL file.
        Does NOT rewrite the entire file, ensuring scalability.

        Args:
            provider: LLM provider ('openai', 'anthropic', 'google').
            model: Model name.
            input_tokens: Input token count.
            output_tokens: Output token count.
            cost: Pre-calculated cost (optional, calculated if None).
            batch_api: Whether batch API was used.
            metadata: Additional metadata (auto_logged, response_id, etc).

        Returns:
            The logged cost in USD.

        Example:
            >>> cost = tracker.log_cost('openai', 'gpt-4o', 100, 50)
            >>> # Or with metadata:
            >>> cost = tracker.log_cost(
            ...     'openai', 'gpt-4o', 100, 50,
            ...     metadata={'auto_logged': True}
            ... )
        """
        # Calculate cost if not provided
        if cost is None:
            cost = self.calculate_cost(provider, model, input_tokens, output_tokens, batch_api)

        # Create cost entry with timestamp
        entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "batch_api": batch_api,
            "metadata": metadata or {},
        }

        # Add to in-memory list
        self.costs.append(entry)
        
        # Append to JSONL file (TL requirement #1 - append-only)
        self._save_costs_append(entry)

        print(f"📝 Cost logged: ${cost:.6f} | {provider}/{model}")

        return cost

    def auto_log_from_llm_client(
        self, provider: str, model: str, response: Dict[str, Any]
    ) -> float:
        """
        Auto-log cost from UnifiedLLMClient response.

        Called automatically from LLM client's generate() method.
        Extracts token usage and logs the cost.

        Integration with UnifiedLLMClient:
        
        In UnifiedLLMClient.__init__():
            from cost_tracker import get_tracker
            self.cost_tracker = get_tracker()

        In UnifiedLLMClient.generate():
            # After getting result from API call:
            if self.cost_tracker:
                try:
                    self.cost_tracker.auto_log_from_llm_client(
                        provider=self.provider,
                        model=self.model,
                        response=result,
                    )
                except Exception as e:
                    print(f"⚠️  Cost logging failed: {e}")

        Args:
            provider: Provider name ('openai', 'anthropic', 'google').
            model: Model name.
            response: Response dict with 'usage' key containing token counts.
                     Expected format: response['usage']['prompt_tokens']
                                      response['usage']['completion_tokens']

        Returns:
            Cost in USD.

        Example:
            >>> response = {
            ...     'content': 'Hello',
            ...     'usage': {
            ...         'prompt_tokens': 100,
            ...         'completion_tokens': 50,
            ...     }
            ... }
            >>> cost = tracker.auto_log_from_llm_client('openai', 'gpt-4o', response)
        """
        # Extract token usage from response
        usage = response.get("usage", {})
        
        # Handle different token naming conventions
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

        # Log the cost
        cost = self.log_cost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "auto_logged": True,
                "response_id": response.get("id", ""),
            },
        )

        return cost

    # ==================== DATA PERSISTENCE (JSONL) ====================

    def _load_costs(self) -> List[Dict[str, Any]]:
        """
        Load all costs from JSONL file (append-only format - TL requirement #1).

        Reads line-by-line JSONL format. Each line is a valid JSON object
        representing a cost entry.

        Returns:
            List of cost entry dictionaries.
            Returns empty list if file doesn't exist.

        Note:
            - One entry per line (JSONL format)
            - Handles corrupted lines gracefully with warnings
            - Sorted chronologically by timestamp
        """
        if not self.data_path.exists():
            return []

        costs = []
        try:
            with open(self.data_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            costs.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            warnings.warn(f"Invalid JSON at line {line_num}: {e}")
        except Exception as e:
            warnings.warn(f"Error loading costs: {e}")

        return costs

    def _save_costs_append(self, entry: Dict[str, Any]) -> None:
        """
        Append single cost entry to JSONL file (TL requirement #1).

        Opens file in append mode and writes ONE entry as JSON line.
        Does NOT read or rewrite the entire file - ensures scalability.

        Args:
            entry: Cost entry dictionary to append.

        Note:
            - Append mode ('a') prevents rewriting entire file
            - Each entry on new line (JSONL format)
            - File is created if doesn't exist
            - Fails gracefully with warning if write fails
        """
        try:
            with open(self.data_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")
        except Exception as e:
            warnings.warn(f"Failed to append cost: {e}")

    def reset(self) -> None:
        """
        Reset all cost data for this user.

        Deletes the user's JSONL file and clears in-memory costs.
        Requires confirmation before execution.
        """
        self.costs = []
        if self.data_path.exists():
            self.data_path.unlink()
        print(f"✅ Reset cost data for {self.user_id}")

    # ==================== ANALYTICS & BUDGET CHECKS ====================

    def get_total_cost(self, costs: Optional[List[Dict]] = None) -> float:
        """
        Get total cost across all API calls.

        Args:
            costs: Optional list of costs to sum (for aggregation).
                   If None, uses in-memory self.costs.

        Returns:
            Total cost in USD.
        """
        source = costs if costs is not None else self.costs
        return sum(entry["cost"] for entry in source)

    def get_monthly_cost(self, costs: Optional[List[Dict]] = None) -> float:
        """
        Get cost for current month (January, February, etc).

        Args:
            costs: Optional list of costs to filter (for aggregation).
                   If None, uses in-memory self.costs.

        Returns:
            Monthly cost in USD.
        """
        now = datetime.now()
        source = costs if costs is not None else self.costs
        
        monthly_cost = 0.0
        for entry in source:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time.month == now.month and entry_time.year == now.year:
                monthly_cost += entry["cost"]
        return monthly_cost

    def check_80_percent_alert(self) -> bool:
        """
        Check if 80% of total budget is reached.

        Returns:
            True if total cost >= 80% of total_budget_limit.

        Example:
            >>> tracker.total_budget_limit = 1000
            >>> tracker.log_cost('openai', 'gpt-4o', 200000, 100000)
            >>> tracker.check_80_percent_alert()  # If >= $800
            True
        """
        total_cost = self.get_total_cost()
        return total_cost >= (self.total_budget_limit * 0.8)

    def check_month_1_2_threshold(self) -> bool:
        """
        Check if Month 1-2 critical threshold ($800) is reached.

        This is a TL-specific requirement for tracking spending
        in the first two months of the project.

        Returns:
            True if total cost >= $800 (month_1_2_threshold).
        """
        total_cost = self.get_total_cost()
        return total_cost >= self.month_1_2_threshold

    def check_monthly_80_percent_alert(self) -> bool:
        """
        Check if 80% of monthly budget is reached.

        Returns:
            True if monthly cost >= 80% of monthly_budget_limit.
        """
        monthly_cost = self.get_monthly_cost()
        return monthly_cost >= (self.monthly_budget_limit * 0.8)

    def get_monthly_projection(self, costs: Optional[List[Dict]] = None) -> float:
        """
        Project monthly spend based on current usage.

        Extrapolates current spending to end of month.

        Args:
            costs: Optional list of costs to project from.

        Returns:
            Projected monthly spending in USD.

        Example:
            >>> # On Jan 5, spent $10
            >>> tracker.get_monthly_projection()
            >>> # Returns: $10 * (30 / 5) = $60 projected for month
        """
        now = datetime.now()
        days_in_month = 30
        days_passed = now.day

        monthly_cost = self.get_monthly_cost(costs)
        if days_passed > 0:
            return round(monthly_cost * (days_in_month / days_passed), 2)
        return 0.0

    def get_cost_breakdown_by_model(
        self, costs: Optional[List[Dict]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed cost breakdown by model.

        Args:
            costs: Optional list of costs to breakdown.

        Returns:
            Dictionary with model names as keys:
            {
                'gpt-4o': {
                    'total_cost': 5.50,
                    'total_calls': 10,
                    'total_input_tokens': 5000,
                    'total_output_tokens': 2500,
                    'provider': 'openai'
                },
                ...
            }
        """
        source = costs if costs is not None else self.costs
        breakdown = {}

        for entry in source:
            model = entry["model"]
            if model not in breakdown:
                breakdown[model] = {
                    "total_cost": 0.0,
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "provider": entry["provider"],
                }

            breakdown[model]["total_cost"] += entry["cost"]
            breakdown[model]["total_calls"] += 1
            breakdown[model]["total_input_tokens"] += entry["input_tokens"]
            breakdown[model]["total_output_tokens"] += entry["output_tokens"]

        return breakdown

    def get_provider_breakdown(
        self, costs: Optional[List[Dict]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get cost breakdown by provider (OpenAI, Anthropic, Google).

        Args:
            costs: Optional list of costs to breakdown.

        Returns:
            Dictionary with provider names as keys:
            {
                'openai': {'cost': 5.50, 'calls': 10},
                'anthropic': {'cost': 3.20, 'calls': 5},
                ...
            }
        """
        source = costs if costs is not None else self.costs
        breakdown = {}

        for entry in source:
            provider = entry["provider"]
            if provider not in breakdown:
                breakdown[provider] = {"cost": 0.0, "calls": 0}
            breakdown[provider]["cost"] += entry["cost"]
            breakdown[provider]["calls"] += 1

        return breakdown

    # ==================== DASHBOARD ====================

    def dashboard(self, aggregate: bool = False, verbose: bool = False):
        """
        Display comprehensive cost dashboard.

        Shows budget status, provider breakdown, recent activity,
        and alerts if thresholds are exceeded.

        Args:
            aggregate: If True, load and display all team members' costs.
                       Requires git pull to get latest team files.
                       (TL requirement #3)
            verbose: If True, show detailed model breakdown.

        Example:
            >>> tracker.dashboard()  # Personal dashboard
            >>> tracker.dashboard(aggregate=True)  # Team view
            >>> tracker.dashboard(verbose=True)  # Detailed breakdown
        """
        # Load costs to display
        if aggregate:
            costs_to_display = self._load_all_user_costs()
            display_title = "🌍 TEAM COST DASHBOARD"
        else:
            costs_to_display = self.costs
            display_title = f"👤 COST DASHBOARD - {self.user_id.upper()}"

        total_cost = self.get_total_cost(costs_to_display)
        monthly_cost = self.get_monthly_cost(costs_to_display)
        projection = self.get_monthly_projection(costs_to_display)

        print("\n" + "=" * 80)
        print(display_title.center(80))
        print("=" * 80)

        # Summary Section
        print("\n📊 SUMMARY")
        print("-" * 80)
        print(f"Total API Calls:      {len(costs_to_display):,}")
        print(f"Total Cost:           ${total_cost:.2f}")
        print(f"Monthly Cost:         ${monthly_cost:.2f}")
        if len(costs_to_display) > 0:
            print(f"Avg Cost per Call:    ${total_cost/len(costs_to_display):.6f}")

        # Budget Status Section
        print("\n💰 BUDGET STATUS")
        print("-" * 80)

        # Total Budget Progress
        total_percent = (
            (total_cost / self.total_budget_limit * 100) if self.total_budget_limit > 0 else 0
        )
        total_bar = self._progress_bar(total_percent, 40)
        print(f"Total Budget:         ${self.total_budget_limit:.2f}")
        print(f"  Spent: ${total_cost:.2f} | Remaining: ${self.total_budget_limit - total_cost:.2f}")
        print(f"  {total_percent:>5.1f}% {total_bar}")

        # Alert checks for total budget
        if self.check_80_percent_alert():
            threshold = self.total_budget_limit * 0.8
            print(f"  ⚠️  80% ALERT: ${threshold:.2f} reached!")
        
        if self.check_month_1_2_threshold():
            print(f"  🚨 MONTH 1-2: ${self.month_1_2_threshold:.2f} threshold reached!")

        # Monthly Budget Progress
        monthly_percent = (
            (monthly_cost / self.monthly_budget_limit * 100)
            if self.monthly_budget_limit > 0
            else 0
        )
        monthly_bar = self._progress_bar(monthly_percent, 40)
        print(f"\nMonthly Budget:       ${self.monthly_budget_limit:.2f}")
        print(f"  Spent: ${monthly_cost:.2f} | Remaining: ${self.monthly_budget_limit - monthly_cost:.2f}")
        print(f"  {monthly_percent:>5.1f}% {monthly_bar}")

        if self.check_monthly_80_percent_alert():
            threshold = self.monthly_budget_limit * 0.8
            print(f"  ⚠️  80% MONTHLY ALERT: ${threshold:.2f} reached!")

        # Projections Section
        print("\n📈 PROJECTIONS")
        print("-" * 80)
        print(f"Projected Month End:  ${projection:.2f}")
        if projection > self.monthly_budget_limit:
            print(f"  ⚠️  Exceeds monthly budget by ${projection - self.monthly_budget_limit:.2f}")

        # Provider Breakdown Section
        print("\n🔧 PROVIDER BREAKDOWN")
        print("-" * 80)
        provider_breakdown = self.get_provider_breakdown(costs_to_display)

        if provider_breakdown:
            print(f"{'Provider':<12} {'Calls':>8} {'Cost':>12} {'% of Total':>12}")
            print("-" * 80)

            for provider, data in provider_breakdown.items():
                percentage = (data["cost"] / total_cost * 100) if total_cost > 0 else 0
                print(
                    f"{provider:<12} {data['calls']:>8,} ${data['cost']:>11.2f} {percentage:>11.1f}%"
                )
        else:
            print("No provider data available")

        # Model Breakdown (verbose mode)
        if verbose:
            print("\n📋 DETAILED MODEL BREAKDOWN")
            print("-" * 80)
            model_breakdown = self.get_cost_breakdown_by_model(costs_to_display)

            if model_breakdown:
                print(f"{'Model':<35} {'Calls':>8} {'Cost':>12} {'Tokens':>12}")
                print("-" * 80)

                for model, data in sorted(
                    model_breakdown.items(),
                    key=lambda x: x[1]["total_cost"],
                    reverse=True,
                ):
                    total_tokens = data["total_input_tokens"] + data["total_output_tokens"]
                    print(
                        f"{model:<35} {data['total_calls']:>8,} "
                        f"${data['total_cost']:>11.2f} {total_tokens:>11,}"
                    )
            else:
                print("No model data available")

        # Recent Activity Section
        print("\n⏱️  RECENT ACTIVITY (Last 10)")
        print("-" * 80)

        if costs_to_display:
            recent = costs_to_display[-10:] if len(costs_to_display) > 10 else costs_to_display
            for entry in recent:
                time_str = datetime.fromisoformat(entry["timestamp"]).strftime("%m-%d %H:%M:%S")
                print(
                    f"{time_str} | {entry['provider']:10} | {entry['model']:25} | ${entry['cost']:>8.6f}"
                )
        else:
            print("No recent activity")

        print("\n" + "=" * 80 + "\n")

    def _load_all_user_costs(self) -> List[Dict[str, Any]]:
        """
        Load costs from all user JSONL files (TL requirement #3 - Team aggregation).

        Used by team leads to view aggregate costs across all team members.
        Each user has their own file (costs_alice.jsonl, costs_bob.jsonl, etc).

        Returns:
            Combined list of all cost entries from all user files,
            sorted by timestamp.

        Note:
            Called when --aggregate flag is used:
            python cost_tracker.py --dashboard --aggregate
        """
        all_costs = []
        
        try:
            for cost_file in sorted(self.data_dir.glob("costs_*.jsonl")):
                with open(cost_file, "r") as f:
                    for line in f:
                        if line.strip():
                            try:
                                all_costs.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            warnings.warn(f"Error loading aggregate costs: {e}")

        # Sort by timestamp
        all_costs.sort(key=lambda x: x.get("timestamp", ""))
        return all_costs

    def _progress_bar(self, percentage: float, width: int = 40) -> str:
        """
        Create ASCII progress bar for dashboard visualization.

        Args:
            percentage: Percentage to display (0-100).
            width: Width of bar in characters.

        Returns:
            String like '████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░'
        """
        filled = min(int(width * percentage / 100), width)
        return "█" * filled + "░" * (width - filled)

    def export_csv(self, path: str, costs: Optional[List[Dict]] = None):
        """
        Export costs to CSV file.

        Useful for reporting, spreadsheet analysis, or importing into
        other systems.

        Args:
            path: File path for CSV export.
            costs: Optional list of costs to export.
                   If None, exports self.costs.

        Example:
            >>> tracker.export_csv('my_costs.csv')
            >>> tracker.export_csv('team_costs.csv', aggregate=True)
        """
        source = costs if costs is not None else self.costs
        
        if not source:
            print("No cost data to export")
            return

        try:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=source[0].keys())
                writer.writeheader()
                writer.writerows(source)
            print(f"✅ Exported {len(source)} rows to {path}")
        except Exception as e:
            print(f"❌ Failed to export CSV: {e}")

    # ==================== CLI ====================

    @staticmethod
    def cli():
        """
        Command-line interface for cost tracker.

        Provides commands for dashboard viewing, data management, and team workflows.
        """
        parser = argparse.ArgumentParser(
            description="Cost Tracker - Production LLM Cost Monitoring",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
USAGE EXAMPLES:

Personal dashboard:
  python utils/cost_tracker.py --dashboard

With detailed breakdown:
  python utils/cost_tracker.py --dashboard --verbose

Team aggregate (after git pull):
  python utils/cost_tracker.py --dashboard --aggregate

Export costs to CSV:
  python utils/cost_tracker.py --export costs.csv

Reset personal data:
  python utils/cost_tracker.py --reset

GIT WORKFLOW (Team Collaboration):
  # After each work session (all team members):
  git add data/metadata/costs_$(whoami).jsonl
  git commit -m "Cost log: $(date)"
  git push

  # Team lead view:
  git pull
  python utils/cost_tracker.py --dashboard --aggregate --verbose
            """,
        )

        parser.add_argument("--dashboard", action="store_true", help="Display dashboard")
        parser.add_argument("--verbose", action="store_true", help="Show detailed breakdown")
        parser.add_argument("--aggregate", action="store_true", help="Team aggregate view")
        parser.add_argument("--reset", action="store_true", help="Reset personal data")
        parser.add_argument("--export", type=str, metavar="FILE", help="Export to CSV")

        parser.add_argument(
            "--monthly-budget",
            type=float,
            default=200.0,
            help="Monthly limit ($200)",
        )
        parser.add_argument(
            "--total-budget",
            type=float,
            default=1000.0,
            help="Total limit ($1000)",
        )

        parser.add_argument("--data-dir", type=str, default="data/metadata", help="Data directory")
        parser.add_argument("--fresh", action="store_true", help="Fresh start")
        parser.add_argument("--user-id", type=str, help="User ID (default: $USER)")

        args = parser.parse_args()

        # Initialize tracker
        tracker = CostTracker(
            user_id=args.user_id,
            data_dir=args.data_dir,
            load_existing=not args.fresh,
            monthly_budget_limit=args.monthly_budget,
            total_budget_limit=args.total_budget,
        )

        # Execute commands
        if args.reset:
            confirm = input("Reset personal cost data? (y/N): ")
            if confirm.lower() == "y":
                tracker.reset()
            return tracker

        if args.export:
            if args.aggregate:
                costs = tracker._load_all_user_costs()
                tracker.export_csv(args.export, costs)
            else:
                tracker.export_csv(args.export)
            return tracker

        if args.dashboard or (not args.reset and not args.export):
            tracker.dashboard(aggregate=args.aggregate, verbose=args.verbose)

        return tracker


# ==================== SINGLETON PATTERN ====================
_default_tracker = None


def get_tracker(user_id: Optional[str] = None) -> CostTracker:
    """
    Get or create singleton instance of CostTracker.

    Used for easy integration with UnifiedLLMClient.
    The first call creates the instance, subsequent calls return the same instance.

    Args:
        user_id: Optional user identifier (uses $USER if None).

    Returns:
        Singleton CostTracker instance.

    Example:
        >>> # In UnifiedLLMClient.__init__():
        >>> from cost_tracker import get_tracker
        >>> self.cost_tracker = get_tracker()
        >>> 
        >>> # Anywhere else:
        >>> tracker = get_tracker()
        >>> tracker.dashboard()
    """
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = CostTracker(user_id=user_id)
    return _default_tracker


if __name__ == "__main__":
    CostTracker.cli()
