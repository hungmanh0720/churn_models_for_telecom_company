import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import os

# 1. Load model
model = CatBoostClassifier()
model.load_model('catboost_info/catboost_v2.cbm')

# 2. Load data
df = pd.read_csv('data/predict_data.csv')

# 3. Gen derived features
df['total_minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
df['total_calls'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
df['total_charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']
df['day_charge_per_min'] = df['Total day charge'] / df['Total day minutes'].replace(0, pd.NA)
df['day_min_per_call'] = df['Total day minutes'] / df['Total day calls'].replace(0, pd.NA)
df['day_charge_per_call'] = df['Total day charge'] / df['Total day calls'].replace(0, pd.NA)
df['eve_charge_per_min'] = df['Total eve charge'] / df['Total eve minutes'].replace(0, pd.NA)
df['eve_min_per_call'] = df['Total eve minutes'] / df['Total eve calls'].replace(0, pd.NA)
df['eve_charge_per_call'] = df['Total eve charge'] / df['Total eve calls'].replace(0, pd.NA)
df['night_charge_per_min'] = df['Total night charge'] / df['Total night minutes'].replace(0, pd.NA)
df['night_min_per_call'] = df['Total night minutes'] / df['Total night calls'].replace(0, pd.NA)
df['night_charge_per_call'] = df['Total night charge'] / df['Total night calls'].replace(0, pd.NA)
df['total_charge_per_min'] = df['total_charge'] / df['total_minutes'].replace(0, pd.NA)
df['total_min_per_call'] = df['total_minutes'] / df['total_calls'].replace(0, pd.NA)
df['total_charge_per_call'] = df['total_charge'] / df['total_calls'].replace(0, pd.NA)
df['day_minutes_ratio'] = df['Total day minutes'] / df['total_minutes']
df['day_charge_ratio'] = df['Total day charge'] / df['total_charge']

# 4. Load selected features
fi = pd.read_csv('catboost_info/feature_importance.csv')
selected_features = fi[fi['importance'] >= 1.0]['feature'].tolist()

y_true = df['Churn']
X = df[selected_features]

# 5. Predict
cat_features = ['International plan', 'Voice mail plan']
cat_idx = [X.columns.tolist().index(c) for c in cat_features if c in X.columns]
pool = Pool(X, cat_features=cat_idx)

y_pred_proba = model.predict_proba(pool)[:, 1]
y_pred_label = model.predict(pool)

# 6. Metrics
auc = roc_auc_score(y_true, y_pred_proba)
cm = confusion_matrix(y_true, y_pred_label)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*40}")
print("PREDICT RESULTS")
print(f"{'='*40}")
print(f"AUC       : {auc:.4f}")
print(f"Precision : {tp/(tp+fp):.4f}")
print(f"Recall    : {tp/(tp+fn):.4f}")
print(f"F1        : {2*tp/(2*tp+fp+fn):.4f}")
print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(classification_report(y_true, y_pred_label, target_names=['Non-Churn', 'Churn']))

# 7. Lưu kết quả
os.makedirs('output', exist_ok=True)
df['predicted_proba'] = y_pred_proba
df['predicted_label'] = y_pred_label
df.to_csv('output/predict_results.csv', index=False)
print("Saved results to output/predict_results.csv")