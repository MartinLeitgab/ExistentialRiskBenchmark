"""
Cost Tracker - Production LLM Cost Monitoring
Provides comprehensive cost tracking across multiple LLM providers with
automatic alerting, budget management, and detailed analytics. Designed for both
research environments and production deployments.
Author:Pooja Puranik
version: 1.0.0
date:23/01/2026
Last updated: 23/01/2026
"""


import json
import csv
import warnings
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse


class CostTracker:

    """Production cost tracker for LLM API cost monitoring and budget
    management.Provides comprehensive cost tracking across multiple
    LLM providers with automatic alerting, budget management, and
    detailed analytics. Designed for both research environments
    and production deployments.

    Attributes:
        monthly_budget_limit: Monthly spending limit (default: $200.00)
        total_budget_limit: Total project budget limit (default: $1,000.00)
        month_1_2_threshold: Critical threshold for months 1-2
        (default: $800.00)
        alert_emails: List of email recipients for alerts (max 4)
        costs: List of all logged cost entries
        alerts_sent: Dictionary tracking which alerts have been sent
    """

    # Pricing per million tokens (input, output)
    PRICING = {
        "openai": {
            "gpt-4o": {"input": 2.50, "output": 10.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4.1": {"input": 2.0, "output": 8.0},
        },
        "anthropic": {
            "claude-opus-4.5": {"input": 5.0, "output": 25.0},
            "claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
            "claude-haiku-4.5": {"input": 1.0, "output": 5.0},
        },
        "google": {
            "gemini-3-pro": {"input": 2.0, "output": 12.0},
            "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        },
    }

    def __init__(
        self,
        data_path: str = "data/metadata/costs.json",
        load_existing: bool = True,
        monthly_budget_limit: float = 200.0,  # ✅ $200 Month 1
        total_budget_limit: float = 1000.0,  # ✅ $1000 total
        alert_emails: Optional[List[str]] = None,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
    ):
        """
        Initialize production cost tracker.

        Args:
            data_path: Path to costs.json
            load_existing: Load existing data
            monthly_budget_limit: $200 for Month 1
            total_budget_limit: $1000 total
            alert_emails: Email recipients (MAX 4)
            smtp_server: SMTP server
            smtp_port: SMTP port
            smtp_username: SMTP username
            smtp_password: SMTP password
        """
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        # ✅ BUDGET CONFIGURATION
        self.monthly_budget_limit = monthly_budget_limit
        self.total_budget_limit = total_budget_limit
        self.month_1_2_threshold = 800.0  # ✅ $800 for Month 1-2

        # ✅ EMAIL ALERTS FOR 4 PEOPLE
        self.alert_emails = (alert_emails or [])[:4]  # Limit to 4

        # SMTP Configuration
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username or os.getenv("SMTP_USERNAME", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")

        # Load existing costs
        self.costs: List[Dict[str, Any]] = []
        if load_existing:
            self.costs = self._load_costs()

        # Alert tracking
        self.alerts_sent = {
            "80_percent_total": False,
            "month_1_2": False,
            "80_percent_monthly": False,
        }

        print(
            f"""
✅ COST TRACKER INITIALIZED - PRODUCTION READY
   Data: {self.data_path}
   Monthly Budget: ${monthly_budget_limit:.2f} (Month 1)
   Total Budget: ${total_budget_limit:.2f}
   80% Alert: ${total_budget_limit * 0.8:.2f}
   Month 1-2: ${self.month_1_2_threshold:.2f}
   Email Alerts: {len(self.alert_emails)}/4 recipients
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
        """Calculate cost for an API call across all supported providers.

        Args:
            provider: One of 'openai', 'anthropic', or 'google'
            model: Specific model name (e.g., 'gpt-4', 'claude-3-5-sonnet')
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            batch_api: Whether to apply 50% batch discount

        Returns:
            Cost in USD rounded to 6 decimal places

        Raises:
            ValueError: When unsupported provider is specified

        Example:
            >>> tracker.calculate_cost('openai', 'gpt-4', 1000, 500)
            0.045
            >>> tracker.calculate_cost('openai', 'gpt-4', 1000, 500, batch_api=True)
            0.0225
        """
        # ✅ Handle negative tokens - return 0.0
        if input_tokens < 0 or output_tokens < 0:
            return 0.0

        if provider not in self.PRICING:
            raise ValueError(
                f"Unknown provider: {provider}. Supported: {list(self.PRICING.keys())}"
            )

        provider_pricing = self.PRICING[provider]

        # Find matching model
        if model not in provider_pricing:
            # Try to find similar model
            for available_model in provider_pricing.keys():
                if model in available_model:
                    model = available_model
                    break
            else:
                # Use first model as fallback
                model = list(provider_pricing.keys())[0]
                warnings.warn(f"Using '{model}' pricing for unknown model")

        pricing = provider_pricing[model]

        # Calculate cost
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        # Apply batch discount
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
        ✅ Log API call cost with automatic alert checking.

        Args:
            provider: LLM provider
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            cost: Pre-calculated cost
            batch_api: Whether batch API was used
            metadata: Additional metadata

        Returns:
            The logged cost
        """
        # Calculate cost if not provided
        if cost is None:
            cost = self.calculate_cost(provider, model, input_tokens, output_tokens, batch_api)

        # Create entry
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

        # Save
        self.costs.append(entry)
        self._save_costs()

        # Print confirmation
        print(f"📝 Cost logged: ${cost:.6f} | {provider}/{model}")

        # Check alerts
        self._check_alerts()

        return cost

    def auto_log_from_llm_client(
        self, provider: str, model: str, response: Dict[str, Any]
    ) -> float:
        """
        ✅ Auto-log cost from UnifiedLLMClient response.

        Usage in UnifiedLLMClient.generate():
        ```
        response = self._api_call(prompt, **kwargs)
        cost = self.cost_tracker.auto_log_from_llm_client(
            provider=self.provider,
            model=self.model,
            response=response
        )
        return response
        ```
        """
        # Extract token usage from response
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

        # Log the cost
        cost = self.log_cost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={"auto_logged": True, "response_id": response.get("id", "")},
        )

        return cost

    # ==================== DATA PERSISTENCE ====================

    def _load_costs(self) -> List[Dict[str, Any]]:
        """✅ Load costs from JSON file."""
        if not self.data_path.exists():
            return []

        try:
            with open(self.data_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            warnings.warn(f"Corrupted cost file at {self.data_path}, starting fresh")
            return []
        except Exception as e:
            warnings.warn(f"Error loading costs: {e}, starting fresh")
            return []

    def _save_costs(self) -> None:
        """✅ Save costs to JSON file."""
        try:
            with open(self.data_path, "w") as f:
                json.dump(self.costs, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save costs: {e}")

    def reset(self) -> None:
        """Reset all cost data."""
        self.costs = []
        self._save_costs()
        self.alerts_sent = {
            "80_percent_total": False,
            "month_1_2": False,
            "80_percent_monthly": False,
        }
        print("✅ Cost data reset")

    # ==================== BUDGET & ALERTS ====================

    def get_total_cost(self) -> float:
        """Get total cost across all API calls."""
        return sum(entry["cost"] for entry in self.costs)

    def get_monthly_cost(self) -> float:
        """Get cost for current month."""
        now = datetime.now()
        monthly_cost = 0.0
        for entry in self.costs:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time.month == now.month and entry_time.year == now.year:
                monthly_cost += entry["cost"]
        return monthly_cost

    def check_80_percent_alert(self) -> bool:
        """✅ Check if 80% of total budget is reached."""
        total_cost = self.get_total_cost()
        return total_cost >= (self.total_budget_limit * 0.8)

    def check_month_1_2_threshold(self) -> bool:
        """✅ Check Month 1-2 threshold of $800."""
        total_cost = self.get_total_cost()
        return total_cost >= self.month_1_2_threshold

    def check_monthly_80_percent_alert(self) -> bool:
        """Check if 80% of monthly budget is reached."""
        monthly_cost = self.get_monthly_cost()
        return monthly_cost >= (self.monthly_budget_limit * 0.8)

    def _check_alerts(self) -> None:
        """Check all budget thresholds and send alerts."""
        total_cost = self.get_total_cost()
        monthly_cost = self.get_monthly_cost()

        # ✅ 80% total budget alert
        if self.check_80_percent_alert() and not self.alerts_sent["80_percent_total"]:
            self._send_80_percent_alert(total_cost)
            self.alerts_sent["80_percent_total"] = True

        # ✅ Month 1-2 threshold alert ($800)
        if self.check_month_1_2_threshold() and not self.alerts_sent["month_1_2"]:
            self._send_month_1_2_alert(total_cost)
            self.alerts_sent["month_1_2"] = True

        # 80% monthly budget alert
        if self.check_monthly_80_percent_alert() and not self.alerts_sent["80_percent_monthly"]:
            self._send_monthly_80_percent_alert(monthly_cost)
            self.alerts_sent["80_percent_monthly"] = True

    def _send_80_percent_alert(self, total_cost: float):
        """✅ Send 80% budget alert."""
        subject = "🚨 COST TRACKER: 80% BUDGET ALERT"
        message = f"""
URGENT: 80% of total budget has been reached!

💰 BUDGET STATUS:
• Total Spent: ${total_cost:.2f}
• Total Budget: ${self.total_budget_limit:.2f}
• 80% Threshold: ${self.total_budget_limit * 0.8:.2f}
• Remaining: ${self.total_budget_limit - total_cost:.2f}

⚠️ RECOMMENDED ACTIONS:
1. Review recent API usage
2. Consider optimizing prompts
3. Monitor daily spending
4. Prepare budget increase if needed

This alert was sent to {len(self.alert_emails)} team members.
"""
        self._send_alert(subject, message, "80% total budget")

    def _send_month_1_2_alert(self, total_cost: float):
        """✅ Send Month 1-2 threshold alert."""
        subject = " COST TRACKER: MONTH 1-2 THRESHOLD REACHED"
        message = f"""
CRITICAL: Month 1-2 threshold of ${self.month_1_2_threshold:.2f} reached!

📈 CURRENT STATUS:
• Total Spent: ${total_cost:.2f}
• Month 1-2 Threshold: ${self.month_1_2_threshold:.2f}
• Total Budget: ${self.total_budget_limit:.2f}
• Over Threshold: ${total_cost - self.month_1_2_threshold:.2f}
• Remaining Budget: ${self.total_budget_limit - total_cost:.2f}

🚨 IMMEDIATE ACTIONS REQUIRED:
1. Halt non-essential API calls
2. Review all automated processes
3. Contact project manager for budget review

This is a critical alert sent to all {len(self.alert_emails)} team members.
"""
        self._send_alert(subject, message, "Month 1-2 threshold")

    def _send_monthly_80_percent_alert(self, monthly_cost: float):
        """Send monthly 80% alert."""
        subject = " COST TRACKER: MONTHLY BUDGET ALERT"
        message = f"""
Monthly budget alert: ${monthly_cost:.2f} spent this month.

📅 MONTHLY STATUS:
• Monthly Spent: ${monthly_cost:.2f}
• Monthly Budget: ${self.monthly_budget_limit:.2f}
• Remaining Monthly: ${self.monthly_budget_limit - monthly_cost:.2f}
• Projected Month End: ${self.get_monthly_projection():.2f}
"""
        self._send_alert(subject, message, "80% monthly budget")

    def _send_alert(self, subject: str, message: str, alert_type: str):
        """Send alert via email or console."""
        # Print to console
        print(f"\n{'='*70}")
        print(f"ALERT: {subject}")
        print(f"{'='*70}")
        print(f"Type: {alert_type}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        print(message)
        print(f"{'='*70}\n")

        # Send email if configured
        if self.alert_emails:
            if self.smtp_username and self.smtp_password:
                self._send_real_email(subject, message)
            else:
                print(" Email not sent: SMTP credentials missing")
                print(f"   Would send to: {', '.join(self.alert_emails)}")

    def _send_real_email(self, subject: str, message: str):
        """Send actual email via SMTP."""
        try:
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = self.smtp_username
            msg["To"] = ", ".join(self.alert_emails)

            # Add timestamp
            full_message = f"{message}\n\n---\nSent: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            msg.attach(MIMEText(full_message, "plain"))

            # Send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            print(f" Email sent to {len(self.alert_emails)} recipients")

        except Exception as e:
            print(f"❌ Failed to send email: {e}")

    # ==================== ANALYTICS ====================

    def get_monthly_projection(self) -> float:
        """✅ Project monthly spend based on current usage."""
        now = datetime.now()
        days_in_month = 30
        days_passed = now.day

        monthly_cost = self.get_monthly_cost()
        if days_passed > 0:
            return round(monthly_cost * (days_in_month / days_passed), 2)
        return 0.0

    def get_cost_breakdown_by_model(self) -> Dict[str, Dict[str, Any]]:
        """✅ Get detailed cost breakdown by model."""
        breakdown = {}

        for entry in self.costs:
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

    def get_provider_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get breakdown by provider."""
        breakdown = {}

        for entry in self.costs:
            provider = entry["provider"]
            if provider not in breakdown:
                breakdown[provider] = {"cost": 0.0, "calls": 0}
            breakdown[provider]["cost"] += entry["cost"]
            breakdown[provider]["calls"] += 1

        return breakdown

    # ==================== DASHBOARD ====================

    def dashboard(self, verbose: bool = False):
        """✅ Display comprehensive dashboard."""
        total_cost = self.get_total_cost()
        monthly_cost = self.get_monthly_cost()
        projection = self.get_monthly_projection()

        print("\n" + "=" * 80)
        print(" " * 30 + "COST TRACKER DASHBOARD")
        print("=" * 80)

        # Summary
        print("\n SUMMARY")
        print("-" * 80)
        print(f"Total API Calls:      {len(self.costs):,}")
        print(f"Total Cost:           ${total_cost:.2f}")
        print(f"Monthly Cost:         ${monthly_cost:.2f}")
        print(
            f"Avg Cost per Call:    ${total_cost/len(self.costs) if len(self.costs) > 0 else 0:.4f}"
        )

        # Budget Status
        print("\n BUDGET STATUS")
        print("-" * 80)

        # Total Budget
        total_percent = (
            (total_cost / self.total_budget_limit * 100) if self.total_budget_limit > 0 else 0
        )
        total_bar = self._progress_bar(total_percent, 50)
        print(f"Total Budget:         ${self.total_budget_limit:.2f}")
        print(
            f" Spent: ${monthly_cost:.2f} " +    
            f" | Remaining: ${self.monthly_budget_limit - monthly_cost:.2f}"
        )
        print(f"  {total_percent:.1f}% {total_bar}")

        if self.check_80_percent_alert():
            print(f"   80% ALERT: ${self.total_budget_limit * 0.8:.2f} reached!")

        if self.check_month_1_2_threshold():
            print(f" MONTH 1-2: ${self.month_1_2_threshold:.2f} threshold reached!")

        # Monthly Budget
        monthly_percent = (
            (monthly_cost / self.monthly_budget_limit * 100) if self.monthly_budget_limit > 0 else 0
        )
        monthly_bar = self._progress_bar(monthly_percent, 50)
        print(f"\nMonthly Budget:       ${self.monthly_budget_limit:.2f}")
        print(
            f"  Spent: ${monthly_cost:.2f} " +
            f"| Remaining: ${self.monthly_budget_limit - monthly_cost:.2f}"
        )
        print(f"  {monthly_percent:.1f}% {monthly_bar}")

        if self.check_monthly_80_percent_alert():
            print(f"  80% MONTHLY ALERT: ${self.monthly_budget_limit * 0.8:.2f} reached!")

        # Email Status
        print("\n EMAIL ALERTS")
        print("-" * 80)
        print(f"Recipients: {len(self.alert_emails)}/4 configured")
        if self.alert_emails:
            for email in self.alert_emails:
                print(f"  • {email}")

        # Projections
        print("\n PROJECTIONS")
        print("-" * 80)
        print(f"Projected Month End:  ${projection:.2f}")
        if projection > self.monthly_budget_limit:
            print("   Projection exceeds monthly budget!")

        # Provider Breakdown
        print("\n PROVIDER BREAKDOWN")
        print("-" * 80)
        provider_breakdown = self.get_provider_breakdown()

        if provider_breakdown:
            print(f"{'Provider':<12} {'Calls':>8} {'Cost':>12} {'% of Total':>12}")
            print("-" * 80)

            for provider, data in provider_breakdown.items():
                # percentage = (data["cost"] / total_cost * 100) if total_cost > 0 else 0
                percentage = (
                        data["cost"] / total_cost * 100
                        ) if total_cost > 0 else 0
                print(
                    f"{provider:<12} {data['calls']:>8,} ${data['cost']:>11.2f} {percentage:>11.1f}%"
                )
        else:
            print("No provider data available")

        # Model Breakdown (verbose)
        if verbose:
            print("\n DETAILED MODEL BREAKDOWN")
            print("-" * 80)
            model_breakdown = self.get_cost_breakdown_by_model()

            if model_breakdown:
                print(f"{'Model':<30} {'Calls':>8} {'Cost':>12} {'Tokens':>12}")
                print("-" * 80)

                for model, data in sorted(
                    model_breakdown.items(),
                    key=lambda x: x[1]["total_cost"],
                    reverse=True,
                ):
                    total_tokens = data["total_input_tokens"] + data["total_output_tokens"]
                    # print(f"{model:<30} {data['total_calls']:>8,} ${data['total_cost']:>11.2f} {total_tokens:>11,}")
                    print(
                        f"{model:<30} {data['total_calls']:>8,} "
                        "${data['total_cost']:>11.2f} {total_tokens:>11,}"
                    )
            else:
                print("No model data available")

        # Recent Activity
        print("\n RECENT ACTIVITY (Last 5)")
        print("-" * 80)

        if self.costs:
            recent = self.costs[-5:] if len(self.costs) > 5 else self.costs
            for entry in recent:
                time_str = datetime.fromisoformat(entry["timestamp"]).strftime("%m-%d %H:%M")
                print(
                    f"{time_str} | {entry['provider']:8} | {entry['model']:20} |"
                    f"${entry['cost']:.4f}"
                )
        else:
            print("No recent activity")

        print("\n" + "=" * 80 + "\n")

    def _progress_bar(self, percentage: float, width: int = 50) -> str:
        """Create progress bar."""
        filled = min(int(width * percentage / 100), width)
        return "█" * filled + "░" * (width - filled)

    def export_csv(self, path: str):
        """Export to CSV."""
        if not self.costs:
            print("No cost data to export")
            return

        try:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.costs[0].keys())
                writer.writeheader()
                writer.writerows(self.costs)
            print(f" Exported {len(self.costs)} rows to {path}")
        except Exception as e:
            print(f" Failed to export CSV: {e}")

    # ==================== CLI ====================

    @staticmethod
    def cli():
        """Command-line interface."""
        parser = argparse.ArgumentParser(
            description="Cost Tracker - Production LLM Cost Monitoring",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python utils/cost_tracker.py --dashboard                    # Show dashboard
  python utils/cost_tracker.py --dashboard --verbose         # Detailed dashboard
  python utils/cost_tracker.py --email team@company.com --dashboard
  python utils/cost_tracker.py --reset                       # Reset all data
  python utils/cost_tracker.py --export costs.csv            # Export to CSV

Email Configuration:
  Set SMTP credentials via environment variables:
    SMTP_USERNAME=your-email@gmail.com
    SMTP_PASSWORD=your-app-password
            """,
        )

        parser.add_argument("--dashboard", action="store_true", help="Display cost dashboard")
        parser.add_argument("--verbose", action="store_true", help="Show detailed breakdown")
        parser.add_argument("--reset", action="store_true", help="Reset all cost data")
        parser.add_argument("--export", type=str, metavar="FILE", help="Export cost data to CSV")

        parser.add_argument(
            "--monthly-budget",
            type=float,
            default=200.0,
            help="Monthly budget limit (default: $200)",
        )
        parser.add_argument(
            "--total-budget",
            type=float,
            default=1000.0,
            help="Total budget limit (default: $1000)",
        )

        parser.add_argument(
            "--email", action="append", metavar="EMAIL", help="Email recipient (max 4)"
        )

        parser.add_argument("--smtp-server", default="smtp.gmail.com", help="SMTP server")
        parser.add_argument("--smtp-port", type=int, default=587, help="SMTP port")
        parser.add_argument("--smtp-username", help="SMTP username")
        parser.add_argument("--smtp-password", help="SMTP password")

        parser.add_argument(
            "--data-path",
            type=str,
            default="data/metadata/costs.json",
            help="Path to cost data file",
        )
        parser.add_argument("--fresh", action="store_true", help="Start with fresh cost data")

        args = parser.parse_args()

        # Initialize tracker
        tracker = CostTracker(
            data_path=args.data_path,
            load_existing=not args.fresh,
            monthly_budget_limit=args.monthly_budget,
            total_budget_limit=args.total_budget,
            alert_emails=args.email,
            smtp_server=args.smtp_server,
            smtp_port=args.smtp_port,
            smtp_username=args.smtp_username,
            smtp_password=args.smtp_password,
        )

        # Execute commands
        if args.reset:
            confirm = input("Reset all cost data? (y/N): ")
            if confirm.lower() == "y":
                tracker.reset()

        if args.export:
            tracker.export_csv(args.export)

        if args.dashboard or (not args.reset and not args.export):
            tracker.dashboard(verbose=args.verbose)

        return tracker


# ==================== SINGLETON ====================
_default_tracker = None


def get_tracker() -> CostTracker:
    """Get singleton instance for easy integration."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = CostTracker()
    return _default_tracker


if __name__ == "__main__":
    CostTracker.cli()
