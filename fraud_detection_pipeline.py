# fraud_detection_pipeline.py
# Full Python 3 script for fraud detection model evaluation
# Ready to upload to GitHub
# Author: ChatGPT (GPT-5)
# Description: Machine learning pipeline using Scikit-learn, XGBoost, and SMOTE for fraud detection

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

# ---------- 1) Load dataset ----------
INPUT_PATH = "/kaggle/input/fraud-detection-transactions-dataset/synthetic_fraud_dataset.csv"
df = pd.read_csv(INPUT_PATH)

# Clean and normalize column names
df.columns = df.columns.str.replace(" ", "_").str.lower()

# Drop unnecessary ID columns
if {'transaction_id', 'user_id'}.issubset(df.columns):
    df = df.drop(['transaction_id', 'user_id'], axis=1)

# ---------- 2) Feature engineering: Extract simple time features ----------
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['txn_hour'] = df['timestamp'].dt.hour
    df['txn_dayofweek'] = df['timestamp'].dt.dayofweek
    df = df.drop(columns=['timestamp'])

# ---------- 3) Split target and features ----------
TARGET = 'fraud_label'
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found!")

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# ---------- 4) Identify numeric and categorical columns ----------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# ---------- 5) Transformers ----------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

# ---------- 6) Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 7) Define classifiers ----------
pos = y_train.sum()
neg = y_train.shape[0] - pos
scale_pos_weight = (neg / pos) if pos != 0 else 1.0

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=2000, solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'SVM': SVC(probability=True, kernel='rbf'),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             scale_pos_weight=scale_pos_weight, random_state=42)
}

smote = SMOTE(random_state=42)

# ---------- 8) Evaluate on hold-out test set ----------
print("=== Hold-out Test AUC-ROC Results ===")
for name, clf in classifiers.items():
    pipe = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = pipe.decision_function(X_test)
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name:15s} AUC-ROC: {auc:.4f}")

# ---------- 9) Cross-validation evaluation ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n=== 5-Fold Cross-Validation AUC-ROC Results ===")
cv_results = {}
for name, clf in classifiers.items():
    pipe = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', clf)
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_mean = scores.mean()
    cv_std = scores.std()
    cv_results[name] = cv_mean
    print(f"{name:15s} CV AUC-ROC: {cv_mean:.4f} ± {cv_std:.4f}")

# ---------- 10) Summary ----------
print("\n=== Final Summary (Mean CV AUC-ROC) ===")
for name, score in cv_results.items():
    print(f"{name:15s}: {score:.4f}")

# Notes:
# - AUC values near 1.0 may indicate overfitting; use stricter validation or regularization.
# - Replace OneHotEncoder with target or hash encoding for high-cardinality categorical data.
# - For time-based data, use temporal validation splits to avoid data leakage.
