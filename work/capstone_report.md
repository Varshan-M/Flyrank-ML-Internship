# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Antigravity (AI Assistant)
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** Flyrank-ML-Internship
- **Date:** 2026-07-24

## 1. Problem framing
The objective of this analysis is to score and rank existing pages based on their momentum, specifically identifying pages that are in significant decline and require a "refresh" or editorial review. 
The unit of analysis is the page (pseudonymized content item) over a 90-day window. The output is a ranked action engine that gives editors a prioritized list of pages to review along with actionable reason codes (e.g. "Sharp rank drop - Check intent", "General decay - Needs fresh information"). 
This data-driven approach allows FlyRank editors to focus their limited bandwidth on the pages that are bleeding the most traffic and offer the highest recovery ROI, rather than guessing which pages need attention.

## 2. Data safety
This analysis uses the full FlyRank internship warehouse release via Hugging Face. 
The following precautions were taken to prevent leakage and ensure data safety:
- Removed derived label-centric columns from any predictive feature logic (e.g. `trend_direction`, `trend_pct`).
- Used pseudonymous IDs (`content_hash_id`, `client_hash_id`) for grouping and splitting (GroupShuffleSplit on client) but excluded them from the feature set.
- Ensured strict time-windowing: the label (last 30 days impressions vs previous 30 days) is strictly separated from the features (which are aggregated from the preceding 30 days, day -60 to day -30).
- Confirmed that no client-identifying details or raw queries are included in this report or any repository files.

## 3. Baseline
Our baseline is a simple, transparent heuristic: if a page's average position drops by more than 1.0 rank position month-over-month, we predict it as declining.
This is a fair comparison because it simulates what a junior SEO might do—simply look at rank drops to find declining pages. 
The baseline achieves an ROC AUC of **0.54** on the holdout set, demonstrating that simple rank drops do not capture the full complexity of impression loss (e.g. query mix shifts or tail query bleeding).

## 4. Model / analysis
We framed this as a binary classification problem predicting `is_declining` (defined as impressions in the last 30 days dropping below 80% of impressions in the previous 30 days).
We used a **RandomForestClassifier** because it handles non-linear relationships well and provides feature importances for our reason codes. 
**Feature list:** 
- `imp_prev30` (historical momentum)
- `pos_prev30` (historical position)
- `pos_change` (momentum of ranking)
- `pos_volatility` (stability of ranking)
- `visible_queries`, `rare_share`, `anon_share`, `top_query_share` (query-level dependency factors).

We intentionally left out current 30-day clicks and impressions as they directly derive the label.

## 5. Evaluation
We evaluated the model using a `GroupShuffleSplit` on `client_hash_id` (25% holdout). This ensures that the model is tested on unseen clients, preventing the model from just memorizing specific client behaviors and confirming it generalizes across different domains.
**Metrics (Holdout Set):**
- Baseline ROC AUC: **0.54**
- Model ROC AUC: **0.68**

The model successfully outperformed the baseline by **0.14**. 
Error analysis reveals that the model sometimes flags pages with stable rankings but high `top_query_share` as declining. This indicates that pages overly reliant on a single query are highly susceptible to sudden impression drops (e.g. algorithm updates or seasonal shifts) even if the average position remains mathematically stable.

## 6. Interpretation
The Random Forest model identified the following key drivers of content decline:
1. **pos_volatility**: The most dominant signal.
2. **pos_change**: Strong secondary indicator.
3. **top_query_share**: Query concentration plays a significant role in stability.

It is notable that raw `pos_prev30` (average position) was less important than the *volatility* and *change* in position. This supports the hypothesis that a page ranking #8 consistently is safer than a page jumping between #2 and #15. A negative result here is that `anon_share` had minimal predictive power for this specific cohort.

## 7. Recommendation
Based on the predicted probabilities, we created an Action Engine that ranks pages by `decline_prob` and assigns specific recommendations:
- **Sharp rank drop:** Requires checking search intent and recent SERP feature changes.
- **Over-reliant on single query:** Requires content expansion and diversification (adding subtopics).
- **Low keyword footprint:** Needs basic structural SEO (H2s, FAQs) to capture tail variations.
- **General decay:** The information is likely stale; needs a date bump and fresh statistics.

FlyRank editors can use this queue daily. *Confidence limits:* These predictions are directional and intended as decision-support. A high decline probability does not guarantee the page is "dead", but rather that it warrants human review. 

## 8. Reproducibility
To reproduce these findings from a fresh clone:
1. Create a Python environment: `pip install duckdb huggingface_hub scikit-learn pandas python-dotenv`
2. Populate the `.env` file with a valid Hugging Face READ token (`HF_TOKENS=hf_...`)
3. Execute the notebook: `jupyter nbconvert --to notebook --execute --inplace work/capstone_opportunity_scoring.ipynb`
All random seeds are set to `42` to ensure consistent splits.
