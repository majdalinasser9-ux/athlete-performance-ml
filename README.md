# 🏃 Athlete Performance ML — DS301 Capstone

> **Dataset:** [Kaggle — Athlete Performance Evaluation](https://www.kaggle.com/datasets/zara2099/athlete-performance-evaluation)  
> **Team:** Majd (Part 1 & 2) | Geny (Part 3 & 4)

---

## 📋 Project Overview

This capstone applies machine learning to predict and categorize athlete performance using physiological and performance data from **1,000 university athletes**. The project covers four independent parts across the full ML workflow.

| Part | Focus | Best Model | Key Metric |
|------|-------|------------|------------|
| 1 | Regression | Ridge Regression | RMSE: 0.0002, R²: 1.0000 |
| 2 | Classification | Tuned SVC (C=10, linear) | Accuracy: 98%, F1: 0.9803 |
| 3 | Time Series | Random Forest + Lag Features | MAE: 0.0294 |
| 4 | Clustering | K-Means (k=3) | Silhouette: see report |

---

## 📁 Repository Structure

```
athlete-performance-ml/
├── data/
│   └── athlete_performance_dataset.csv
├── part1_regression/
│   ├── part1_notebook.ipynb
│   ├── part1_report.pdf
│   └── part1_slides.pptx
├── part2_classification/
│   ├── part2_notebook.ipynb
│   ├── part2_report.pdf
│   └── part2_slides.pptx
├── part3_timeseries/
│   ├── part3_notebook.ipynb
│   ├── part3_report.pdf
│   └── part3_slides.pptx
├── part4_clustering/
│   ├── part4_notebook.ipynb
│   ├── part4_report.pdf
│   └── part4_slides.pptx
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | Kaggle — Athlete Performance Evaluation (zara2099) |
| Size | 1,000 records, 20 original features |
| Missing Values | None |
| Train/Test Split | 80% / 20% (random_state=42) |

**Feature Groups:**
- **Physiological:** heart rate, oxygen saturation, body temperature, respiration rate, hydration level
- **Performance:** speed, acceleration, endurance score, agility score, reaction time
- **Workload:** workload, fatigue index
- **Targets:** `performance_score` (continuous) | `performance_category` (Low/Medium/High)

---

## Part 1 — Regression

**Goal:** Predict continuous `performance_score`

**Models trained:** Linear Regression · Lasso · Ridge · SVR · Decision Tree · Random Forest · Neural Network (Keras)

| Model | RMSE | R² |
|-------|------|----|
| Linear Regression | 0.0000 | 1.0000 |
| **Ridge Regression ⭐** | **0.0002** | **1.0000** |
| Lasso | 0.0248 | 0.9507 |
| Random Forest | 0.0334 | 0.9104 |
| SVR | 0.0494 | 0.8047 |
| Decision Tree | 0.0624 | 0.6877 |
| Neural Network | 0.0675 | 0.6349 |

**Tuning:** GridSearchCV on Random Forest → best: `n_estimators=200, max_depth=10`

**Key Finding:** Endurance score (40.7%), agility score (23.0%), and fatigue index (15.0%) account for ~79% of predictive power.

---

## Part 2 — Classification

**Goal:** Categorize athletes into Low / Medium / High performance tiers

**Models trained:** Logistic Regression · SVC · Decision Tree · Random Forest · Neural Network (Keras)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 0.9200 | 0.9271 |
| SVC (Default) | 0.9250 | 0.9298 |
| **Tuned SVC ⭐** | **0.9800** | **0.9803** |
| Decision Tree | 0.8800 | 0.8691 |
| Random Forest | 0.8850 | 0.8484 |
| Neural Network | 0.9350 | 0.9316 |

**Tuning:** GridSearchCV on SVC → best: `C=10, kernel=linear` (+5.3% F1 improvement)

**Key Finding:** 86.1% class imbalance (Medium) handled via stratified split and F1 as primary metric.

---

## Part 3 — Time Series

**Goal:** Forecast `performance_score` over time

**Methods:** Holt-Winters Exponential Smoothing · Random Forest with lag features

| Method | MAE | RMSE |
|--------|-----|------|
| Holt-Winters | 0.0904 | 0.1164 |
| **Random Forest (Lag Features) ⭐** | **0.0294** | **0.0428** |

**Tuning:** GridSearchCV with TimeSeriesSplit → best: `n_estimators=300, max_depth=None`

**Key Finding:** Series is stationary (ADF p≈0), no seasonality. Lag features capture short-term patterns 3× better than Holt-Winters.

---

## Part 4 — Clustering

**Goal:** Discover natural groupings in athlete data (unsupervised)

**Algorithms:** K-Means · DBSCAN · Agglomerative Clustering (ward + complete linkage)

| Algorithm | Clusters | Silhouette |
|-----------|----------|------------|
| K-Means (k=3) ⭐ | 3 | See report |
| DBSCAN | Varies | See report |
| Agglomerative (ward) | 3 | See report |

**Key Finding:** K-Means (k=3) most interpretable — naturally aligns with Low/Medium/High tiers without labels.

---

## ⚙️ Setup & Requirements

```bash
# Create environment (Python 3.11 required for TensorFlow)
conda create -n ml_env python=3.11
conda activate ml_env

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter ipykernel

# Register Jupyter kernel
python -m ipykernel install --user --name=ml_env --display-name "Python 3.11 (ml_env)"
```

---

## 👥 Team Contributions

| Member | Responsibilities |
|--------|-----------------|
| Majd | EDA, Preprocessing, Part 1 (Regression), Part 2 (Classification) |
| Geny | Part 3 (Time Series), Part 4 (Clustering), Visualizations, Presentation |