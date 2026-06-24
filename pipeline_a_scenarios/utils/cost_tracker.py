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
version: 2.0.1
date: 26/01/2026
Last updated: 08/02/2026

Pipeline B note — Step 5 (~USD 235–300 class runs): totals from this tracker are
still approximate. We use published per-1M standard text-token list prices only;
actual bills differ with batch vs standard API, cached input, long-context tiers
(e.g. Gemini prompts >200k tokens, OpenAI >272k), regional multipliers (Anthropic),
reasoning/thinking token billing, and preview SKU changes. Reconcile against
provider invoices for authoritative spend.
"""

import json
import csv
import warnings
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

CallType = Literal["sync", "batch"]


def _build_batch_pricing(
    sync_pricing: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Derive batch API per-1M rates (50% of standard) from sync pricing."""
    return {
        provider: {
            model: {
                "input": rates["input"] / 2,
                "output": rates["output"] / 2,
            }
            for model, rates in models.items()
        }
        for provider, models in sync_pricing.items()
    }


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

    # Pricing per million tokens (input, output), standard sync API text generation.
    # Sources (checked 2026-02): OpenAI platform pricing table; Anthropic Claude
    # pricing docs; Google Gemini API pricing (Developer API).
    # PRICING_BATCH holds batch-API rates (typically 50% of sync); pick via call_type
    # in calculate_cost / log_cost — do not apply an extra multiplier on batch paths.
    PRICING_SYNC = {
        "openai": {
            # gpt-5.5: canonical production judge / core OpenAI model (<272k tier).
            # gpt-5.4 / gpt-5.2: retained for historical JSONL replay (distinct tiers).
            "gpt-5.5": {"input": 5.0, "output": 30.0},
            "gpt-5.4": {"input": 2.5, "output": 15.0},
            "gpt-5.2": {"input": 1.75, "output": 14.0},
            "gpt-4o": {"input": 2.5, "output": 10.0},
            "gpt-4o-mini": {"input": 0.075, "output": 0.30},
            "gpt-4": {"input": 3.0, "output": 6.0},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        },
        "anthropic": {
            # Claude Opus 4.5 / 4.6 / 4.7 / 4.8: $5 / $25 per 1M input/output
            # (Anthropic pricing docs). Hyphen vs dot model ids both appear in
            # logs. 4-8 pricing assumed equal to 4-7; confirm vs Anthropic
            # pricing docs at run time.
            "claude-opus-4-7": {"input": 5.0, "output": 25.0},
            "claude-opus-4.7": {"input": 5.0, "output": 25.0},
            "claude-opus-4-8": {"input": 5.0, "output": 25.0},
            "claude-opus-4.8": {"input": 5.0, "output": 25.0},
            "claude-opus-4.5": {"input": 5.0, "output": 25.0},
            # Sonnet 4.6: $3 / $15 per 1M (Anthropic pricing docs, Feb 2026).
            "claude-sonnet-4.6": {"input": 3.0, "output": 15.0},
            "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
            # Legacy JSONL replay aliases
            "claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
            "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
            "claude-haiku-4.5": {"input": 1.0, "output": 5.0},
        },
        "google": {
            # gemini-3.1-pro-preview paid standard text: $2 / $12 per 1M for prompts
            # <=200k tokens; $4 / $18 when prompt exceeds 200k (Gemini API pricing).
            # This table uses the <=200k tier only — see module docstring.
            "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0},
            "gemini-3-flash-preview": {"input": 0.5, "output": 3.0},
            "gemini-3-pro": {"input": 2.0, "output": 12.0},
            "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        },
    }

    PRICING_BATCH = _build_batch_pricing(PRICING_SYNC)

    # Backward-compatible alias for callers that still reference PRICING.
    PRICING = PRICING_SYNC

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

        from dotenv import load_dotenv

        load_dotenv()
        # User identification - defaults to system USER env var
        self.user_id = (
            user_id
            or os.getenv("USER", "unknown")
            or os.getenv("USER")
            or os.getenv("USERNAME", "unknown")
        )

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

    def _resolve_call_type(
        self,
        call_type: Optional[CallType] = None,
        batch_api: bool = False,
    ) -> CallType:
        if call_type is not None:
            return call_type
        return "batch" if batch_api else "sync"

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        call_type: Optional[CallType] = None,
        batch_api: bool = False,
    ) -> float:
        """
        Calculate cost for API call across all supported providers.

        Supports OpenAI, Anthropic, and Google with pricing per million tokens.
        Uses PRICING_SYNC or PRICING_BATCH based on call_type (no extra multiplier).

        Args:
            provider: One of 'openai', 'anthropic', or 'google'.
            model: Specific model name (e.g., 'gpt-4o', 'claude-sonnet-4-6').
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.
            call_type: 'sync' (standard API) or 'batch' (batch API rates).
            batch_api: Deprecated; if call_type is omitted, True selects 'batch'.

        Returns:
            Cost in USD rounded to 6 decimal places.

        Raises:
            ValueError: If provider is not supported.

        Example:
            >>> tracker.calculate_cost('openai', 'gpt-4o', 1000, 500)
            0.0075
            >>> tracker.calculate_cost('openai', 'gpt-4o', 1000, 500, call_type='batch')
            0.00375
        """
        if input_tokens < 0 or output_tokens < 0:
            return 0.0

        resolved_call_type = self._resolve_call_type(call_type, batch_api)
        pricing_tables = {
            "sync": self.PRICING_SYNC,
            "batch": self.PRICING_BATCH,
        }
        all_pricing = pricing_tables[resolved_call_type]

        if provider not in all_pricing:
            raise ValueError(
                f"Unknown provider: {provider}. Supported: {list(all_pricing.keys())}"
            )

        provider_pricing = all_pricing[provider]

        if model not in provider_pricing:
            for available_model in provider_pricing.keys():
                if model in available_model or available_model in model:
                    model = available_model
                    break
            else:
                model = list(provider_pricing.keys())[0]
                warnings.warn(f"Model '{model}' not found, using {model} pricing")

        pricing = provider_pricing[model]
        cost = (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        ) / 1_000_000
        return round(cost, 6)

    def log_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float] = None,
        call_type: Optional[CallType] = None,
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
            call_type: 'sync' or 'batch' — selects PRICING_SYNC vs PRICING_BATCH.
            batch_api: Deprecated; if call_type is omitted, True selects 'batch'.
            metadata: Additional metadata (auto_logged, response_id, etc).

        Returns:
            The logged cost in USD.

        Example:
            >>> cost = tracker.log_cost('openai', 'gpt-4o', 100, 50)
            >>> cost = tracker.log_cost(
            ...     'openai', 'gpt-4o', 100, 50,
            ...     call_type='batch',
            ...     metadata={'auto_logged': True},
            ... )
        """
        resolved_call_type = self._resolve_call_type(call_type, batch_api)
        is_batch = resolved_call_type == "batch"

        if cost is None:
            cost = self.calculate_cost(
                provider,
                model,
                input_tokens,
                output_tokens,
                call_type=resolved_call_type,
            )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "call_type": resolved_call_type,
            "batch_api": is_batch,
            "metadata": metadata or {},
        }

        # Add to in-memory list
        self.costs.append(entry)

        # Append to JSONL file (TL requirement #1 - append-only)
        self._save_costs_append(entry)

        print(f"📝 Cost logged: ${cost:.6f} | {provider}/{model}")

        return cost

    def auto_log_from_llm_client(
        self,
        provider: str,
        model: str,
        response: Dict[str, Any],
        is_batch: bool = False,
        batch_discount: bool = True,
    ) -> float:
        """
        Auto-log cost from UnifiedLLMClient response.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google').
            model: Model name.
            response: Response dict from LLM client.
            is_batch: Whether this is a batch API call.
            batch_discount: Whether to apply batch discount.

        Returns:
            Cost in USD.
        """
        # Extract token usage from response
        usage = response.get("usage", {})

        # Handle different token naming conventions
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

        # Handle Anthropic thinking tokens if present
        if provider == "anthropic" and "thinking_tokens" in usage:
            # Anthropic charges thinking tokens as output tokens
            output_tokens = usage.get("output_tokens", 0) + usage.get(
                "thinking_tokens", 0
            )

        call_type: CallType = "batch" if (is_batch and batch_discount) else "sync"
        cost = self.log_cost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type=call_type,
            metadata={
                "auto_logged": True,
                "response_id": response.get("id", ""),
                "is_batch": is_batch,
                "batch_discount_applied": is_batch and batch_discount,
            },
        )

        return cost

    def log_batch_cost(
        self,
        provider: str,
        model: str,
        batch_requests: List[Dict],
        batch_results: Dict[str, str],
        total_input_tokens: Optional[int] = None,
        total_output_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Log cost for batch API calls.

        Args:
            provider: LLM provider.
            model: Model name.
            batch_requests: Original batch requests list.
            batch_results: Results from retrieve_batch_results().
            total_input_tokens: Optional pre-calculated total input tokens.
            total_output_tokens: Optional pre-calculated total output tokens.
            metadata: Additional metadata.

        Returns:
            Total batch cost in USD.
        """
        # If tokens not provided, estimate from results
        if total_input_tokens is None:
            # Estimate based on average token length of prompts
            total_input_tokens = sum(
                len(r.get("prompt", "")) // 4 for r in batch_requests  # Rough estimate
            )

        if total_output_tokens is None:
            # Estimate based on output text length
            total_output_tokens = sum(
                len(result) // 4  # Rough estimate
                for result in batch_results.values()
                if isinstance(result, str) and not result.startswith("[ERROR]")
            )

        cost = self.calculate_cost(
            provider=provider,
            model=model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            call_type="batch",
        )

        # Count successful vs failed
        successful = sum(
            1
            for r in batch_results.values()
            if isinstance(r, str) and not r.startswith("[ERROR]")
        )
        failed = len(batch_results) - successful

        # Log the cost
        return self.log_cost(
            provider=provider,
            model=model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost=cost,
            call_type="batch",
            metadata={
                "batch_size": len(batch_requests),
                "successful": successful,
                "failed": failed,
                "success_rate": (
                    successful / len(batch_requests) if batch_requests else 0
                ),
                **(metadata or {}),
            },
        )

    def log_parallel_cost(
        self,
        provider: str,
        model: str,
        requests: List[Dict],
        results: Dict[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Log cost for parallel API calls (Gemini workaround).

        Args:
            provider: LLM provider.
            model: Model name.
            requests: Original requests list.
            results: Individual API responses with usage data.
            metadata: Additional metadata.

        Returns:
            Total parallel execution cost in USD.
        """
        total_input_tokens = 0
        total_output_tokens = 0
        successful = 0
        failed = 0

        for req in requests:
            req_id = req["id"]
            result = results.get(req_id, {})

            if isinstance(result, dict) and "usage" in result:
                usage = result["usage"]
                total_input_tokens += usage.get(
                    "input_tokens", usage.get("prompt_tokens", 0)
                )
                total_output_tokens += usage.get(
                    "output_tokens", usage.get("completion_tokens", 0)
                )
                successful += 1
            else:
                failed += 1

        # Log the cost (no batch discount for parallel calls)
        return self.log_cost(
            provider=provider,
            model=model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            call_type="sync",
            metadata={
                "parallel_size": len(requests),
                "successful": successful,
                "failed": failed,
                "execution_type": "parallel",
                "success_rate": successful / len(requests) if requests else 0,
                **(metadata or {}),
            },
        )

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

    def get_batch_stats(self, costs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Get statistics about batch API usage.

        Args:
            costs: Optional list of costs to analyze.

        Returns:
            Dictionary with batch statistics.
        """
        source = costs if costs is not None else self.costs
        batch_entries = [e for e in source if e.get("batch_api", False)]

        if not batch_entries:
            return {
                "total_batch_calls": 0,
                "total_batch_cost": 0.0,
                "total_batch_savings": 0.0,
                "avg_batch_size": 0,
            }

        total_batch_cost = sum(e["cost"] for e in batch_entries)
        total_savings = 0.0
        total_batch_size = 0

        for entry in batch_entries:
            metadata = entry.get("metadata", {})
            batch_size = metadata.get("batch_size", 0)
            total_batch_size += batch_size

            # Calculate what it would have cost without batch discount
            original_cost = self.calculate_cost(
                provider=entry["provider"],
                model=entry["model"],
                input_tokens=entry["input_tokens"],
                output_tokens=entry["output_tokens"],
                call_type="sync",
            )
            total_savings += original_cost - entry["cost"]

        return {
            "total_batch_calls": len(batch_entries),
            "total_batch_cost": total_batch_cost,
            "total_batch_savings": total_savings,
            "avg_batch_size": (
                total_batch_size / len(batch_entries) if batch_entries else 0
            ),
        }

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
        batch_stats = self.get_batch_stats(costs_to_display)

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

        # Batch Statistics
        if batch_stats["total_batch_calls"] > 0:
            print(f"Batch Calls:          {batch_stats['total_batch_calls']:,}")
            print(f"Batch Savings:        ${batch_stats['total_batch_savings']:.2f}")
            print(f"Avg Batch Size:       {batch_stats['avg_batch_size']:.1f}")

        # Budget Status Section
        print("\n💰 BUDGET STATUS")
        print("-" * 80)

        # Total Budget Progress
        total_percent = (
            (total_cost / self.total_budget_limit * 100)
            if self.total_budget_limit > 0
            else 0
        )
        total_bar = self._progress_bar(total_percent, 40)
        print(f"Total Budget:         ${self.total_budget_limit:.2f}")
        print(
            f"  Spent: ${total_cost:.2f} | Remaining: ${self.total_budget_limit - total_cost:.2f}"
        )
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
        print(
            f"  Spent: ${monthly_cost:.2f} | "
            f"Remaining: ${self.monthly_budget_limit - monthly_cost:.2f}"
        )
        print(f"  {monthly_percent:>5.1f}% {monthly_bar}")

        if self.check_monthly_80_percent_alert():
            threshold = self.monthly_budget_limit * 0.8
            print(f"  ⚠️  80% MONTHLY ALERT: ${threshold:.2f} reached!")

        # Projections Section
        print("\n📈 PROJECTIONS")
        print("-" * 80)
        print(f"Projected Month End:  ${projection:.2f}")
        if projection > self.monthly_budget_limit:
            print(
                f"  ⚠️  Exceeds monthly budget by ${projection - self.monthly_budget_limit:.2f}"
            )

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
                    f"{provider:<12} {data['calls']:>8,} "
                    f"${data['cost']:>11.2f} {percentage:>11.1f}%"
                )
        else:
            print("No provider data available")

        # Model Breakdown (verbose mode)
        if verbose:
            print("\n📋 DETAILED MODEL BREAKDOWN")
            print("-" * 80)
            model_breakdown = self.get_cost_breakdown_by_model(costs_to_display)

            if model_breakdown:
                print(
                    f"{'Model':<35} {'Calls':>8} {'Cost':>12} {'Tokens':>12} {'Batch %':>10}"
                )
                print("-" * 80)

                for model, data in sorted(
                    model_breakdown.items(),
                    key=lambda x: x[1]["total_cost"],
                    reverse=True,
                ):
                    total_tokens = (
                        data["total_input_tokens"] + data["total_output_tokens"]
                    )
                    batch_calls = sum(
                        1
                        for e in costs_to_display
                        if e["model"] == model and e.get("batch_api", False)
                    )
                    batch_percent = (
                        (batch_calls / data["total_calls"] * 100)
                        if data["total_calls"] > 0
                        else 0
                    )
                    print(
                        f"{model:<35} {data['total_calls']:>8,} "
                        f"${data['total_cost']:>11.2f} {total_tokens:>11,} "
                        f"{batch_percent:>9.1f}%"
                    )
            else:
                print("No model data available")

        # Recent Activity Section
        print("\n⏱️  RECENT ACTIVITY (Last 10)")
        print("-" * 80)

        if costs_to_display:
            recent = (
                costs_to_display[-10:]
                if len(costs_to_display) > 10
                else costs_to_display
            )
            for entry in recent:
                time_str = datetime.fromisoformat(entry["timestamp"]).strftime(
                    "%m-%d %H:%M:%S"
                )
                batch_flag = " [BATCH]" if entry.get("batch_api", False) else ""
                print(
                    f"{time_str} | {entry['provider']:10} | {entry['model']:25} | "
                    f"${entry['cost']:>8.6f}{batch_flag}"
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

        parser.add_argument(
            "--dashboard", action="store_true", help="Display dashboard"
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Show detailed breakdown"
        )
        parser.add_argument(
            "--aggregate", action="store_true", help="Team aggregate view"
        )
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

        parser.add_argument(
            "--data-dir", type=str, default="data/metadata", help="Data directory"
        )
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
