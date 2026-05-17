from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

app = FastAPI()

# Load model
model = CatBoostClassifier()
model.load_model('catboost_info/catboost_v2.cbm')

SELECTED_FEATURES = [
    'Customer service calls', 'International plan', 'total_minutes',
    'Total day charge', 'Total intl calls', 'Total day minutes',
    'Total intl minutes', 'Voice mail plan', 'Total intl charge',
    'total_charge_per_call', 'Total eve charge', 'Number vmail messages',
    'Total eve minutes', 'night_charge_per_min', 'total_min_per_call',
    'day_minutes_ratio', 'day_charge_per_call', 'eve_charge_per_call',
    'Total night charge'
]

class CustomerInput(BaseModel):
    account_length: int
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    customer_service_calls: int

def gen_features(data: CustomerInput) -> pd.DataFrame:
    d = {
        'Total day minutes': data.total_day_minutes,
        'Total day calls': data.total_day_calls,
        'Total day charge': data.total_day_charge,
        'Total eve minutes': data.total_eve_minutes,
        'Total eve calls': data.total_eve_calls,
        'Total eve charge': data.total_eve_charge,
        'Total night minutes': data.total_night_minutes,
        'Total night calls': data.total_night_calls,
        'Total night charge': data.total_night_charge,
        'Total intl minutes': data.total_intl_minutes,
        'Total intl calls': data.total_intl_calls,
        'Total intl charge': data.total_intl_charge,
        'Customer service calls': data.customer_service_calls,
        'Number vmail messages': data.number_vmail_messages,
        'International plan': data.international_plan,
        'Voice mail plan': data.voice_mail_plan,
    }

    df = pd.DataFrame([d])

    # Gen derived features
    df['total_minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
    df['total_calls'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
    df['total_charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']
    df['total_charge_per_call'] = df['total_charge'] / df['total_calls'].replace(0, np.nan)
    df['total_min_per_call'] = df['total_minutes'] / df['total_calls'].replace(0, np.nan)
    df['day_minutes_ratio'] = df['Total day minutes'] / df['total_minutes'].replace(0, np.nan)
    df['day_charge_per_call'] = df['Total day charge'] / df['Total day calls'].replace(0, np.nan)
    df['eve_charge_per_call'] = df['Total eve charge'] / df['Total eve calls'].replace(0, np.nan)
    df['night_charge_per_min'] = df['Total night charge'] / df['Total night minutes'].replace(0, np.nan)

    return df[SELECTED_FEATURES]

@app.post('/predict')
def predict(data: CustomerInput):
    df = gen_features(data)
    cat_features = ['International plan', 'Voice mail plan']
    cat_idx = [df.columns.tolist().index(c) for c in cat_features]
    pool = Pool(df, cat_features=cat_idx)

    proba = model.predict_proba(pool)[0][1]
    label = int(model.predict(pool)[0])

    return {
        'churn_probability': round(float(proba), 4),
        'churn_prediction': label,
        'churn_label': 'Churn' if label == 1 else 'Non-Churn'
    }