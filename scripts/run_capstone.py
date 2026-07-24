import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'duckdb', 'huggingface_hub', 'scikit-learn', 'pandas'])
import os, duckdb, pandas as pd, numpy as np
from dotenv import load_dotenv

load_dotenv('.env')
HF_TOKEN = os.environ.get('HF_TOKENS') or os.environ.get('HF_TOKEN')

print("Connecting to DuckDB...")
con = duckdb.connect()
con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{HF_TOKEN}')")

REL = 'hf://datasets/FlyRank/internship-warehouse'
TABLES = {
    'dim_clients':                f"read_parquet('{REL}/dim_clients.parquet')",
    'dim_content':                f"read_parquet('{REL}/dim_content.parquet')",
    'fact_daily':                 f"read_parquet('{REL}/fact_content_daily_performance/**/*.parquet')",
    'fact_daily_sample':          f"read_parquet('{REL}/fact_content_daily_performance_sample.parquet')",
    'fact_query_90d':             f"read_parquet('{REL}/fact_content_query_90d.parquet')",
}

print("Running feature query...")
features = con.sql(f"""
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
""").df()

print("Running query signals...")
qsignals = con.sql(f"""
    SELECT content_hash_id,
           ANY_VALUE(content_visible_query_count)     AS visible_queries,
           ANY_VALUE(rare_impressions_share)          AS rare_share,
           ANY_VALUE(anonymized_impressions_share)    AS anon_share,
           MAX(impressions_90d)                       AS top_query_impressions,
           SUM(impressions_90d)                       AS kept_impressions
    FROM {TABLES['fact_query_90d']}
    GROUP BY content_hash_id
""").df()

qsignals['top_query_share'] = qsignals['top_query_impressions'] / qsignals['kept_impressions']
data = features.merge(qsignals, on='content_hash_id', how='left')

# Drop rows where pos_volatility is NaN due to single point
data = data.dropna(subset=['pos_volatility', 'pos_prev30', 'pos_last30'])
data['pos_change'] = data['pos_last30'] - data['pos_prev30']
data['is_declining'] = (data['imp_last30'] < 0.8 * data['imp_prev30']).astype(int)
print(f'Joined: {len(data):,} rows')

from sklearn.metrics import classification_report, roc_auc_score

data['baseline_pred'] = (data['pos_change'] > 1.0).astype(int)
print("Baseline Classification Report:")
print(classification_report(data['is_declining'], data['baseline_pred'], digits=3))
print("Baseline ROC AUC:", roc_auc_score(data['is_declining'], data['baseline_pred']))

from sklearn.ensemble import RandomForestClassifier
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

importances = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
importances.sort_values('importance', ascending=False, inplace=True)
print("\\nFeature Importances:")
print(importances)
