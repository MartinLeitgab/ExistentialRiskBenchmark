
### Module: utils.cost_tracker
**Author:** Pooja Puranik
**Created:** 2026-01-23
**Version:** 1.0.0
**Last Updated:** 2026-01-23

### 🎯 Purpose
Production-ready LLM API cost monitoring and budget management utility for tracking expenses across OpenAI, Anthropic, and Google providers with automated alerts and analytics.

## ✅ Core Features

### Multi-Provider Cost Calculation
* **OpenAI:** `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`
* **Anthropic:** `claude-opus-4.5`, `claude-sonnet-4.5`, `claude-haiku-4.5`
* **Google:** `gemini-3-pro`, `gemini-2.5-pro`, `gemini-2.5-flash-lite`

### Budget Management
* **Monthly budget:** $200 (Month 1 requirement)
* **Total budget:** $1,000 (Project requirement)
* **Month 1-2 threshold:** $800 (Critical alert)

### Automated Alert System
* 80% total budget alert ($800 threshold)
* Month 1-2 threshold alert ($800)
* 80% monthly budget alert ($160)


### Analytics & Reporting
* Monthly cost projections
* Provider breakdown analysis
* Model-level cost aggregation
* CSV export functionality
* Interactive CLI dashboard

### Data Persistence
* JSON-based cost storage (`data/metadata/costs.json`)
* Automatic loading of existing data
* Corruption recovery mechanisms

---

## 📖 Class Documentation: `CostTracker`

Main class for tracking LLM API costs, managing budgets, and generating alerts.


### 🚨 Alert System

#### Thresholds
1.  **80% Total Budget:** $800
2.  **Month 1-2 Threshold:** $800 (Critical)
3.  **80% Monthly Budget:** $160

### Setup
```python
from pipeline_a_scenarios.utils.cost_tracker import CostTracker

tracker = CostTracker(
    monthly_budget_limit=200.0,
    total_budget_limit=1000.0,
    alert_emails=["user@company.com"]
)

```
### CLI Interface
```bash
# Display the basic cost summary dashboard
python utils/cost_tracker.py --dashboard

output :

# Display detailed dashboard with granular model-level breakdown
python utils/cost_tracker.py --dashboard --verbose

# Export all logged transaction data to a CSV file
python utils/cost_tracker.py --export costs.csv

# Reset all cost logs and clear alert history (requires confirmation)
python utils/cost_tracker.py --reset

# Run dashboard with custom budget overrides for the current session
python utils/cost_tracker.py --monthly-budget 500 --total-budget 2000 --dashboard
```

### Expected Output

✅ COST TRACKER INITIALIZED - PRODUCTION READY
   Data: data/metadata/costs.json
   Monthly Budget: $200.00 (Month 1)
   Total Budget: $1000.00
   80% Alert: $800.00
   Month 1-2: $800.00
   Email Alerts: 1/4 recipients

================================================================================
                          📊 COST TRACKER DASHBOARD
================================================================================
📈 SUMMARY
--------------------------------------------------------------------------------
Total API Calls:      2
Total Cost:           $0.0115
Monthly Cost:         $0.0115

💰 BUDGET STATUS
--------------------------------------------------------------------------------
Total Budget:         $1000.00
  Spent: $0.01 | Remaining: $999.99
  [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.001%

📧 EMAIL ALERTS
--------------------------------------------------------------------------------
Recipients: team-lead@company.com

🏢 PROVIDER BREAKDOWN
--------------------------------------------------------------------------------
Provider        Calls     Cost       % of Total
--------------------------------------------------------------------------------
openai               1    $0.0025    21.7%
anthropic            1    $0.0090    78.3%


 
## Procedure after using cost tracker

Everyone should follow the gitworkflow after using Cost Tracker
```
git add data/metadata/costs_$(whoami).jsonl
git commit -m "Cost log"
git push

# For Team lead
git pull  # Get teammates' cost logs

```
