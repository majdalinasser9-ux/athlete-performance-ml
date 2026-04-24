# 🏃 Athlete Performance Classification — DS301

## 📌 Project Overview
Predicting athlete performance levels (Low / Medium / High) using 
machine learning classification models applied to real-time 
physiological and performance data from university athletes.

## 👥 Team
| Name | Role |
|---|---|
| **Majd** | EDA · Preprocessing · Model Training · Evaluation |
| **Geny** | Cross Validation · Visualizations · Presentation Slides |

## 📦 Dataset
- Source: Kaggle — Athlete Performance Evaluation Dataset
- Link: https://www.kaggle.com/datasets/zara2099/athlete-performance-evaluation-dataset
- 1,000 rows · 20 features · Target: performance_category (Low/Medium/High)

## 🤖 Models Used
| Model | Split Accuracy | CV Mean |
|---|---|---|
| Logistic Regression 🥇 | 96.0% | 96.2% |
| Decision Tree | 88.5% | 84.0% |
| Random Forest | 88.0% | 88.5% |
| KNN (K=4) | 87.5% | 89.2% |

## 📊 Key Findings
- Logistic Regression performed best due to class imbalance (86% Medium)
- Top features: endurance_score, agility_score, fatigue_index
- Heavy class imbalance detected — SMOTE suggested as improvement
- Best K for KNN = 4 found via elbow curve

## 🗂️ Project Structure
athlete-performance-ml/
├── data/                        ← dataset (not pushed to GitHub)
├── notebooks/
│   └── 01_eda.ipynb             ← full ML pipeline
├── results/                     ← all charts and confusion matrices
└── README.md

## ⚙️ How to Run
1. Clone this repo
2. Create virtual environment: python3 -m venv venv
3. Activate: source venv/bin/activate
4. Install libraries: pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
5. Open notebooks/01_eda.ipynb in VS Code
6. Run all cells top to bottom

## 🔧 Suggested Improvements
- SMOTE to fix class imbalance (86% Medium)
- GridSearchCV for hyperparameter tuning
- SVM with RBF kernel
- Deploy best model as Flask API
