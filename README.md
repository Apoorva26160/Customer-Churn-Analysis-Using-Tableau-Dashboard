# Customer Churn Analysis — Tableau Dashboard

End-to-end customer churn analysis: feature engineering, Gradient Boosting classifier, and a Tableau-ready CSV export for interactive dashboards.

## Workflow

```
Raw Customer Data
    → Feature Engineering (tenure segments, engagement score, revenue at risk)
    → Gradient Boosting Classifier
    → Churn Probability Scores
    → CSV export → Tableau Dashboard
```

## Key Features Built

| Feature | Description |
|---|---|
| `tenure_segment` | Bucketed tenure (0-6m, 6-12m, 1-2yr, 2-4yr, 4yr+) |
| `charge_per_product` | Monthly charge normalized by number of products |
| `revenue_at_risk` | Monthly charges × tenure |
| `engagement_score` | Mean of add-on service flags |
| `contract_encoded` | Ordinal encoding of contract type |
| `churn_probability` | Model output — used in Tableau |
| `churn_risk_segment` | Low / Medium / High risk bucket |

## Usage

```bash
pip install -r requirements.txt
python churn_analysis.py
```

## Tableau Dashboard Metrics

- **Churn Rate by Contract Type**
- **Revenue at Risk by Segment**
- **Top Churn Drivers** (feature importance)
- **Churn Probability Distribution**
- **Cohort Retention Heatmap**

## Project Structure

```
churn_analysis.py    # Feature engineering, model training, Tableau export
requirements.txt
```
