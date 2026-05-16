import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def calc_iv_auc(df, target='Churn', n_bins=10):
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]

    for col in numeric_cols:
        try:
            # Chia bin
            df_temp = df[[col, target]].dropna()
            df_temp['bin'] = pd.qcut(df_temp[col], q=n_bins, duplicates='drop')

            # Tính IV
            stats = df_temp.groupby('bin')[target].agg(['sum', 'count'])
            stats.columns = ['event', 'total']
            stats['non_event'] = stats['total'] - stats['event']

            total_event = stats['event'].sum()
            total_non_event = stats['non_event'].sum()

            stats['dist_event'] = stats['event'] / total_event
            stats['dist_non_event'] = stats['non_event'] / total_non_event

            stats['woe'] = np.log(
                (stats['dist_event'] + 1e-9) / (stats['dist_non_event'] + 1e-9)
            )
            stats['iv'] = (stats['dist_event'] - stats['dist_non_event']) * stats['woe']
            iv = stats['iv'].sum()

            # Tính AUC
            auc = roc_auc_score(df_temp[target], df_temp[col])
            auc = max(auc, 1 - auc)  # flip nếu AUC < 0.5

            results.append({'feature': col, 'iv': round(iv, 4), 'auc': round(auc, 4)})

        except Exception as e:
            results.append({'feature': col, 'iv': None, 'auc': None})

    result_df = pd.DataFrame(results).sort_values('iv', ascending=False).reset_index(drop=True)
    return result_df

