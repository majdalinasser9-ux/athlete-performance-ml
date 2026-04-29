Athlete Performance Machine Learning — DS301 Capstone
Dataset: Kaggle — Athlete Performance Evaluation
Team: Majd (Part 1, Part 2) | Geny (Part 3, Part 4)

Project Overview
This capstone applies machine learning to predict and categorize athlete performance using physiological and performance data from 1,000 university athletes. The project is divided into four independent parts covering the full ML workflow.
PartFocusBest ModelKey MetricPart 1RegressionRidge RegressionRMSE: 0.0002, R²: 1.0000Part 2ClassificationTuned SVC (C=10, linear)F1: 0.9803, Acc: 98%Part 3Time SeriesRandom Forest + Lag FeaturesMAE: 0.0294Part 4ClusteringK-Means (k=3)Silhouette Score: see report

Repository Structure
athlete-performance-ml/
├── data/
│   └── athlete_performance_dataset.csv
├── part1_regression/
│   ├── part1_notebook.ipynb
│   ├── part1_report.pdf
│   ├── part1_slides.pptx
│   └── results/
├── part2_classification/
│   ├── part2_notebook.ipynb
│   ├── part2_report.pdf
│   ├── part2_slides.pptx
│   └── results/
├── part3_timeseries/
│   ├── part3_notebook.ipynb
│   ├── part3_report.pdf
│   ├── part3_slides.pptx
│   └── results/
├── part4_clustering/
│   ├── part4_notebook.ipynb
│   ├── part4_report.pdf
│   ├── part4_slides.pptx
│   └── results/
└── README.md

Dataset

Source: Kaggle — Athlete Performance Evaluation (zara2099)
Size: 1,000 records, 20 original features
Features: Physiological (heart rate, oxygen saturation, body temperature, respiration rate, hydration level), Performance (speed, acceleration, endurance score, agility score, reaction time), Workload (workload, fatigue index)
Targets: performance_score (continuous) for regression; performance_category (Low/Medium/High) for classification


Part 1 — Regression
Goal: Predict continuous performance_score
Models: Linear Regression, Lasso, Ridge, SVR, Decision Tree, Random Forest, Neural Network (Keras)
Best Model: Ridge Regression — RMSE: 0.0002, R²: 1.0000
Tuning: GridSearchCV on Random Forest — best params: n_estimators=200, max_depth=10
Key Finding: Endurance score (40.7%), agility score (23.0%), and fatigue index (15.0%) account for ~79% of predictive power.

Part 2 — Classification
Goal: Categorize athletes into Low / Medium / High performance tiers
Models: Logistic Regression, SVC, Decision Tree, Random Forest, Neural Network (Keras)
Best Model: Tuned SVC (C=10, kernel=linear) — Accuracy: 98%, F1: 0.9803
Tuning: GridSearchCV on SVC — improved F1 from 0.9298 to 0.9803
Key Finding: 86.1% class imbalance (Medium) handled via stratified split and F1 as primary metric.

Part 3 — Time Series
Goal: Forecast performance_score over time
Methods: Holt-Winters Exponential Smoothing (classical) + Random Forest with lag features (ML)
Best Model: Random Forest with lag features — MAE: 0.0294 vs Holt-Winters MAE: 0.0904
Tuning: GridSearchCV with TimeSeriesSplit — best params: n_estimators=300, max_depth=None
Key Finding: Series is stationary with no seasonality. Lag features capture short-term patterns that classical methods miss.

Part 4 — Clustering
Goal: Discover natural groupings in athlete data (unsupervised)
Algorithms: K-Means, DBSCAN, Agglomerative Clustering (ward + complete linkage)
Best Result: K-Means (k=3) — most interpretable, aligns with known performance tiers
Key Finding: Feature space naturally separates athletes into 3 groups without label information.

Setup & Requirements
bash# Create environment
conda create -n ml_env python=3.11
conda activate ml_env

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter ipykernel

# Register kernel
python -m ipykernel install --user --name=ml_env --display-name "Python 3.11 (ml_env)"
Python: 3.11 (TensorFlow requires ≤ 3.11)
Key Libraries: TensorFlow/Keras, scikit-learn, pandas, numpy, matplotlib, seaborn

Team Contributions
MemberContributionsMajdEDA, Preprocessing, Part 1 (Regression), Part 2 (Classification)GenyPart 3 (Time Series), Part 4 (Clustering), Visualizations, Presentation