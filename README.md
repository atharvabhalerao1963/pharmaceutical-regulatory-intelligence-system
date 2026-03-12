<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-1.7+-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/SHAP-Explainable_AI-00C4B4?style=for-the-badge"/>


<br/><br/>

# 💊 PRIS — Pharmaceutical Regulatory Intelligence System

### An end-to-end Machine Learning system that predicts whether a pharmaceutical drug should be classified as **Regulated** or **Non-Regulated**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge)](https://pharma-regulatory-ai.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/⭐_Star_on-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/atharvabhalerao1963/pharmaceutical-regulatory-intelligence-system)

<br/>

![PRIS Banner](https://img.shields.io/badge/60%2C000_Records-30_Features-0066cc?style=flat-square) &nbsp;
![Accuracy](https://img.shields.io/badge/Accuracy-77.9%25-success?style=flat-square) &nbsp;
![F1 Score](https://img.shields.io/badge/F1_Score-0.568-blue?style=flat-square) &nbsp;
![Models](https://img.shields.io/badge/Models_Compared-6-orange?style=flat-square)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Model Explainability](#-model-explainability-shap)
- [App Features](#-app-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation--local-setup)
- [Author](#-author)

---

## 🔬 Overview

Pharmaceutical regulatory classification is a critical process — it determines whether a drug requires government oversight, controlled distribution, and strict safety monitoring. Traditionally done manually, this project automates that process using Machine Learning.

**PRIS** (Pharmaceutical Regulatory Intelligence System) is a complete ML pipeline that:

- Trains on **60,000 pharmaceutical records** across **30+ features**
- Engineers **domain-driven features** capturing safety, market, and risk dynamics
- Deploys an **interactive Streamlit app** with real-time predictions
- Explains every prediction using **SHAP (Shapley Additive Explanations)**

> **Problem:** Given a drug's safety, market, and regulatory attributes — should it be classified as *Regulated* or *Non-Regulated*?

---

## 🚀 Live Demo

🔗 **[https://pharma-regulatory-ai.streamlit.app/](https://pharma-regulatory-ai.streamlit.app/)**

The app supports:
- **Single drug prediction** — input all attributes and get instant classification
- **Batch CSV prediction** — upload hundreds of drugs, download predictions
- **SHAP waterfall charts** — understand *why* each prediction was made
- **Feature importance visualization** — see which factors matter most

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Records | 60,000 |
| Features | 30 raw + 11 engineered = **41 total** |
| Records after cleaning | 57,000 |
| Training samples | 45,600 |
| Test samples | 11,400 |
| Train / Test Split | 80 / 20 |
| Target Classes | `Regulated Drug` / `Non-Regulated Drug` |

**Key raw features include:**

| Category | Features |
|---|---|
| 🧪 Safety | `Side_Effect_Severity_Score`, `Abuse_Potential_Score`, `Adverse_Event_Reports` |
| 📋 Regulatory | `Regulatory_Risk_Score`, `Clinical_Trial_Phase`, `Recall_History_Count` |
| 💰 Financial | `Price_Per_Unit`, `Production_Cost`, `R&D_Investment_Million`, `Marketing_Spend` |
| 🏥 Distribution | `Prescription_Rate`, `Hospital_Distribution_Percentage`, `Doctor_Recommendation_Rate` |
| 🏭 Product | `Dosage_mg`, `Drug_Form`, `Therapeutic_Class`, `Manufacturing_Region` |

---

## 🔄 Project Workflow

```
Raw Dataset (60K records)
        │
        ▼
  Data Cleaning & EDA
  ├─ Missing value handling
  ├─ Categorical encoding
  └─ Class imbalance analysis
        │
        ▼
  Feature Engineering (+11 features)
  ├─ Risk_Safety_Interaction
  ├─ Drug_Safety_Risk
  ├─ Doctor_Influence_Index
  ├─ Marketing_Efficiency
  └─ Log_Risk_Safety
        │
        ▼
  Model Training & Comparison
  ├─ Logistic Regression
  ├─ Random Forest
  ├─ XGBoost (+ SMOTE, + Threshold Tuning)
  └─ Final Tuned XGBoost ✅
        │
        ▼
  Hyperparameter Tuning
  ├─ RandomizedSearchCV (broad exploration)
  └─ GridSearchCV (fine-tuning)
        │
        ▼
  Explainability (SHAP)
        │
        ▼
  Streamlit Deployment
```

---

## ⚙️ Feature Engineering

Domain knowledge was used to create 11 new features that capture complex interactions the raw features can't express alone.

### Risk-Safety Interaction
```
Risk_Safety_Interaction = Regulatory_Risk_Score × Drug_Safety_Risk
```
> Captures the **combined amplified effect** of regulatory and safety risk — a drug high on both is exponentially more likely to be regulated.

### Drug Safety Risk
```
Drug_Safety_Risk = Side_Effect_Severity_Score + Abuse_Potential_Score
```

### Doctor Influence Index
```
Doctor_Influence_Index = Doctor_Recommendation_Rate × Prescription_Rate
```

### Marketing Efficiency
```
Marketing_Efficiency = Annual_Sales_Volume / Marketing_Spend
```

### Log Risk Safety
```
Log_Risk_Safety = log(1 + Risk_Safety_Interaction)
```
> Log transformation reduces skew in the interaction term.

---

## 🤖 Models & Results

Six models were trained and evaluated. The primary metric for selection was **F1 Score** — chosen because the dataset is imbalanced and both false positives (unnecessary regulation) and false negatives (missing dangerous drugs) carry real-world cost.

| Model | Accuracy | Recall | Precision | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.779 | 0.374 | 0.639 | 0.472 |
| Random Forest | 0.776 | 0.370 | 0.628 | 0.466 |
| Logistic Regression + SMOTE | 0.623 | 0.771 | 0.391 | 0.519 |
| Threshold Tuned Model | 0.673 | 0.692 | 0.426 | 0.528 |
| XGBoost | 0.717 | 0.694 | 0.476 | 0.564 |
| **Final Tuned XGBoost ✅** | **0.709** | **0.725** | **0.467** | **0.568** |

### Why XGBoost?

Logistic Regression + SMOTE had the **highest recall** (0.771) but very low precision (0.391), generating too many false positives. The **Tuned XGBoost** delivered the **best F1 Score** — the optimal balance between catching truly regulated drugs while minimizing false alarms.

### Final Model Hyperparameters

```python
XGBClassifier(
    max_depth        = 4,
    learning_rate    = 0.03,
    n_estimators     = 250,
    subsample        = 0.7,
    colsample_bytree = 0.7,
    gamma            = 0.2,
    scale_pos_weight = <class_weight>   # handles imbalance
)
```

---

## 🧠 Model Explainability (SHAP)

SHAP (SHapley Additive Explanations) was used to make the model's decisions transparent and interpretable. Every prediction in the app shows a **waterfall chart** explaining which features pushed the prediction toward *Regulated* or *Non-Regulated*.

**Top SHAP features:**

```
1. Risk_Safety_Interaction      ████████████████████  Most important
2. Regulatory_Risk_Score        ████████████████
3. Drug_Safety_Risk             ██████████████
4. Doctor_Influence_Index       ████████████
5. Abuse_Potential_Score        ██████████
```

> Safety and regulatory risk are the **primary drivers** of drug classification — consistent with real-world pharmaceutical regulation logic.

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| 🔍 Single Prediction | Input 41 drug attributes, get instant regulatory classification with confidence score |
| 📂 Batch Prediction | Upload a CSV of multiple drugs, download predictions with confidence scores |
| 📊 Feature Importance | Interactive bar chart of top 15 most important model features |
| 🧠 SHAP Explanation | Per-prediction waterfall chart showing exactly *why* the model predicted what it did |
| 📋 Input Review | Expandable view of all 41 engineered features passed to the model |
| ⬇️ Download Results | Export batch prediction results as a CSV file |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.10+` | Core language |
| `XGBoost` | Final classification model |
| `Scikit-Learn` | Preprocessing, model comparison, tuning |
| `SHAP` | Model explainability |
| `Pandas / NumPy` | Data manipulation & feature engineering |
| `Matplotlib` | Visualizations |
| `Streamlit` | Web app deployment |
| `Joblib` | Model serialization |

---

## 💻 Installation & Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/atharvabhalerao1963/pharmaceutical-regulatory-intelligence-system.git
cd pharmaceutical-regulatory-intelligence-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### `requirements.txt`

```
streamlit
xgboost
scikit-learn
shap
pandas
numpy
matplotlib
joblib
```

---

## 📁 Project Structure

```
📦 pharmaceutical-regulatory-intelligence-system
 ┣ 📄 app.py                    # Streamlit application (all features)
 ┣ 📄 drug_regulation_model.pkl # Trained XGBoost model
 ┣ 📄 requirements.txt          # Python dependencies
 ┣ 📓 PRIS_notebook.ipynb       # Full ML pipeline notebook
 ┗ 📄 README.md
```

---

## 🔮 Future Improvements

- [ ] Integrate a real pharmaceutical dataset (FDA / WHO)
- [ ] Build a REST API using **FastAPI**
- [ ] Add automated model monitoring & drift detection
- [ ] Multi-class classification (e.g., Schedule I / II / III)
- [ ] Docker containerization for portable deployment
- [ ] CI/CD pipeline with GitHub Actions

---

## 👤 Author

<div align="center">

**Atharva Bhalerao**
B.Tech Information Technology — MGM University

[![GitHub](https://img.shields.io/badge/GitHub-atharvabhalerao1963-181717?style=for-the-badge&logo=github)](https://github.com/atharvabhalerao1963)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Atharva_Bhalerao-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/atharva-bhalerao-b62787298/)

*If you found this project useful, consider giving it a ⭐ on GitHub!*

</div>

---

<div align="center">
<sub>Built using Python · XGBoost · SHAP · Streamlit</sub>
</div>
