import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import os

# 1. Load data
df = pd.read_csv('data/processed_data.csv')

# 2. Drop cột bin EDA + cột không cần
drop_cols = [
    'Churn', 'State', 'Area code',
    'day_min_bin', 'cs_bin', 'account_bin',
    'day_minutes_group', 'cs_calls_group',
    'intl_bin', 'charge_bin', 'service_calls_bin',
    'total_charge'
]
drop_cols = [c for c in drop_cols if c in df.columns]
cat_features = ['International plan', 'Voice mail plan']

X = df.drop(columns=drop_cols)
y = df['Churn']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cat_idx = [X.columns.tolist().index(c) for c in cat_features]

# 4. Train
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool = Pool(X_test, y_test, cat_features=cat_idx)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

# 5. Đánh giá
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_label = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred_proba)

cm = confusion_matrix(y_test, y_pred_label)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*40}")
print(f"AUC       : {auc:.4f}")
print(f"Precision : {tp/(tp+fp):.4f}")
print(f"Recall    : {tp/(tp+fn):.4f}")
print(f"F1        : {2*tp/(2*tp+fp+fn):.4f}")
print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(classification_report(y_test, y_pred_label, target_names=['Non-Churn', 'Churn']))

# 6. Lưu feature importance
os.makedirs('catboost_info', exist_ok=True)
fi = pd.DataFrame({
    'feature': X.columns,
    'importance': model.get_feature_importance()
}).sort_values('importance', ascending=False).reset_index(drop=True)

fi.to_csv('catboost_info/feature_importance.csv', index=False)
print("\nSaved feature importance to catboost_info/feature_importance.csv")
print(fi.to_string())