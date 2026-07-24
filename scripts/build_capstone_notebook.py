import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Markdown introduction
cells.append(nbf.v4.new_markdown_cell("""# Capstone: Content Opportunity Scoring

This notebook identifies pages that are declining in search visibility and recommends targeted actions (e.g., refresh content, rewrite, monitor).

**Lane:** Refresh / Content Opportunity Scoring
**Objective:** Score pages that are declining and output a ranked action engine with reason codes.
"""))

# Cell 2: Setup and Token
cells.append(nbf.v4.new_code_cell("""%pip -q install duckdb huggingface_hub scikit-learn pandas
import os, duckdb, pandas as pd, numpy as np
from dotenv import load_dotenv

load_dotenv('.env')
HF_TOKEN = os.environ.get('HF_TOKENS') or os.environ.get('HF_TOKEN')
"""))

# Cell 3: DuckDB connection and Data loading
cells.append(nbf.v4.new_code_cell("""con = duckdb.connect()
con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{HF_TOKEN}')")

REL = 'hf://datasets/FlyRank/internship-warehouse'
TABLES = {
    'dim_clients':                f"read_parquet('{REL}/dim_clients.parquet')",
    'dim_content':                f"read_parquet('{REL}/dim_content.parquet')",
    'fact_daily':                 f"read_parquet('{REL}/fact_content_daily_performance/**/*.parquet')",
    'fact_daily_sample':          f"read_parquet('{REL}/fact_content_daily_performance_sample.parquet')",
    'fact_query_90d':             f"read_parquet('{REL}/fact_content_query_90d.parquet')",
}
"""))

# Cell 4: Feature generation query
cells.append(nbf.v4.new_markdown_cell("""## Feature Engineering

We use DuckDB to aggregate daily performance over a 90-day window.
Features: 
- `imp_prev30`: Impressions from day -60 to -30
- `imp_last30`: Impressions from day -30 to 0
- `clk_last30`: Clicks from day -30 to 0
- `pos_prev30`: Average position from day -60 to -30
- `pos_last30`: Average position from day -30 to 0
- Position volatility and query concentration

Label: `is_declining` = `imp_last30 < 0.8 * imp_prev30`
"""))

cells.append(nbf.v4.new_code_cell("""features = con.sql(f\"\"\"
    WITH bounds AS (
        SELECT MAX(report_date) AS end_d FROM {TABLES['fact_daily']}
    ),
    windowed AS (
        SELECT f.client_hash_id, f.content_hash_id,
               SUM(CASE WHEN f.report_date >  b.end_d - INTERVAL 30 DAY THEN f.gsc_impressions ELSE 0 END) AS imp_last30,
               SUM(CASE WHEN f.report_date >  b.end_d - INTERVAL 60 DAY AND f.report_date <= b.end_d - INTERVAL 30 DAY THEN f.gsc_impressions ELSE 0 END) AS imp_prev30,
               SUM(CASE WHEN f.report_date >  b.end_d - INTERVAL 30 DAY THEN f.gsc_clicks ELSE 0 END)      AS clk_last30,
               AVG(CASE WHEN f.report_date >  b.end_d - INTERVAL 30 DAY THEN f.gsc_avg_position END)       AS pos_last30,
               AVG(CASE WHEN f.report_date >  b.end_d - INTERVAL 60 DAY AND f.report_date <= b.end_d - INTERVAL 30 DAY THEN f.gsc_avg_position END)       AS pos_prev30,
               STDDEV_POP(CASE WHEN f.report_date > b.end_d - INTERVAL 60 DAY THEN f.gsc_avg_position END) AS pos_volatility
        FROM {TABLES['fact_daily']} f, bounds b
        WHERE f.report_date > b.end_d - INTERVAL 90 DAY
        GROUP BY 1, 2
        HAVING imp_prev30 >= 10
    )
    SELECT * FROM windowed
\"\"\").df()

features.head()
"""))

# Cell 5: Query-level Signals
cells.append(nbf.v4.new_code_cell("""qsignals = con.sql(f\"\"\"
    SELECT content_hash_id,
           ANY_VALUE(content_visible_query_count)     AS visible_queries,
           ANY_VALUE(rare_impressions_share)          AS rare_share,
           ANY_VALUE(anonymized_impressions_share)    AS anon_share,
           MAX(impressions_90d)                       AS top_query_impressions,
           SUM(impressions_90d)                       AS kept_impressions
    FROM {TABLES['fact_query_90d']}
    GROUP BY content_hash_id
\"\"\").df()

qsignals['top_query_share'] = qsignals['top_query_impressions'] / qsignals['kept_impressions']
data = features.merge(qsignals, on='content_hash_id', how='left')

# Drop rows where pos_volatility is NaN due to single point
data = data.dropna(subset=['pos_volatility', 'pos_prev30', 'pos_last30'])
data['pos_change'] = data['pos_last30'] - data['pos_prev30']
data['is_declining'] = (data['imp_last30'] < 0.8 * data['imp_prev30']).astype(int)
print(f'Joined: {len(data):,} rows')
"""))

# Cell 6: Baseline
cells.append(nbf.v4.new_markdown_cell("""## Baseline Rule

Baseline: If `pos_change > 1.0` (rank worsened by 1 position), predict declining.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import classification_report, roc_auc_score

data['baseline_pred'] = (data['pos_change'] > 1.0).astype(int)
print("Baseline Classification Report:")
print(classification_report(data['is_declining'], data['baseline_pred'], digits=3))
print("Baseline ROC AUC:", roc_auc_score(data['is_declining'], data['baseline_pred']))
"""))


# Cell 7: ML Model
cells.append(nbf.v4.new_markdown_cell("""## Honest Validation & Machine Learning Model

We use `GroupShuffleSplit` on `client_hash_id` to ensure clients in the train set don't leak into the test set.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
import warnings
warnings.filterwarnings('ignore')

feature_cols = ['imp_prev30', 'visible_queries', 'rare_share', 'anon_share', 'top_query_share', 'pos_volatility', 'pos_prev30', 'pos_change']
model_data = data.dropna(subset=feature_cols).copy()

X = model_data[feature_cols]
y = model_data['is_declining']
groups = model_data['client_hash_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5).fit(X_tr, y_tr)
preds = model.predict(X_te)
probs = model.predict_proba(X_te)[:, 1]

print(f'Base rate (always predict majority): {max(y_te.mean(), 1 - y_te.mean()):.3f}')
print("Model Classification Report:")
print(classification_report(y_te, preds, digits=3))
print("Model ROC AUC:", roc_auc_score(y_te, probs))
"""))

# Cell 8: Feature Importances
cells.append(nbf.v4.new_code_cell("""importances = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
importances.sort_values('importance', ascending=False, inplace=True)
importances
"""))

# Cell 9: Ranked Recommendations
cells.append(nbf.v4.new_markdown_cell("""## Ranked Recommendations (Action Engine)

We output a prioritized queue of declining pages and a recommended action for an editor.
"""))

cells.append(nbf.v4.new_code_cell("""model_data['decline_prob'] = model.predict_proba(X)[:, 1]

# Filter for pages actually in decline (or predicted to be at high risk)
action_queue = model_data[model_data['decline_prob'] > 0.6].copy()

def get_reason_code(row):
    if row['pos_change'] > 2.0:
        return 'Sharp rank drop - Check intent/SERP changes'
    elif row['top_query_share'] > 0.8:
        return 'Over-reliant on single query - Needs diversification'
    elif row['visible_queries'] < 5:
        return 'Low keyword footprint - Expand headings/subtopics'
    else:
        return 'General decay - Needs fresh information/update'

action_queue['reason_code'] = action_queue.apply(get_reason_code, axis=1)
action_queue.sort_values(['decline_prob', 'imp_prev30'], ascending=[False, False], inplace=True)

print("Top 10 High-Priority Content to Refresh:")
display(action_queue[['content_hash_id', 'decline_prob', 'imp_prev30', 'pos_change', 'reason_code']].head(10))
"""))

nb.cells = cells
with open('d:/Flyrank-ML-Internship/work/capstone_opportunity_scoring.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at work/capstone_opportunity_scoring.ipynb")
