# Churn Models for Telecom Company

A machine learning project that predicts customer churn for a telecom company using CatBoost, served via a FastAPI endpoint and containerized with Docker.

---

## 📌 Problem Statement

Telecom companies lose revenue when customers churn without warning. This project builds a churn prediction model trained on customer call behavior and service usage data, enabling proactive retention strategies.

---

## 🗂️ Project Structure

```
churn_models_for_telecom_company/
├── catboost_info/
│   ├── catboost_v2.cbm           # Trained model
│   ├── feature_importance.csv    # Feature importance v1
│   └── feature_importance_v2.csv # Feature importance v2
├── data/
│   ├── train_data.csv            # Training data
│   ├── predict_data.csv          # Prediction data
│   └── processed_data.csv        # Processed data with derived features
├── output/
│   └── predict_results.csv       # Prediction results
├── utils/
│   └── helper.py                 # IV and AUC calculation functions
├── build_api.py                  # FastAPI serving layer
├── Dockerfile                    # Docker configuration
├── eda_data.ipynb                # EDA notebook
├── train_model.py                # Train with full features
├── train_model_v2.py             # Retrain with selected features
├── predict.py                    # Evaluate on test set
└── requirements.txt              # Dependencies
```

---

## ⚙️ Pipeline Overview

```
Raw Data → EDA → Feature Engineering → Model Training → Evaluation → FastAPI → Docker
```

1. **EDA** — Analyze call behavior across day/evening/night/international metrics
2. **Feature Engineering** — Derive features such as `total_minutes`, `day_min_per_call`, `night_charge_per_min`
3. **Model Training** — Train CatBoost, select features via importance ranking, retrain
4. **Evaluation** — Evaluate on holdout set using AUC, Precision, Recall, F1
5. **Serving** — Expose prediction via FastAPI REST endpoint
6. **Containerization** — Dockerize the API for portable deployment

---

## 🤖 Model Results

| Metric | V1 | V2 |
|--------|----|----|
| AUC | 0.8793 | **0.9158** |
| Precision | 0.9474 | **0.9605** |
| Recall | 0.6923 | **0.7684** |
| F1 | 0.8000 | **0.8538** |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| ML | CatBoost |
| Data | Pandas, NumPy |
| API | FastAPI, Uvicorn |
| Container | Docker |
| Notebook | Jupyter Notebook |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/manhhung-2002/churn_models_for_telecom_company.git
cd churn_models_for_telecom_company

# Install dependencies
pip install -r requirements.txt

# Run API local
uvicorn build_api:app --reload --port 8000
```

---

## 🐳 Docker

```bash
docker build -t churn-api:v1 .
docker run -p 8000:8000 churn-api:v1
```

API docs available at `http://localhost:8000/docs`

---

## 📮 API

**POST /predict**

Input: 17 raw customer features

Output:
```json
{
    "churn_probability": 0.9601,
    "churn_prediction": 1,
    "churn_label": "Churn"
}
```

---

## 👤 Author

**Manh Hung Do**  
GitHub: [@manhhung-2002](https://github.com/manhhung-2002)
