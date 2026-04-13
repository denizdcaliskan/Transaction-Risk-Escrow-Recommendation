# Transaction-Risk-Escrow-Recommendation
Automated Fraud Risk Scoring System

##  Project Overview
This project focuses on detecting fraudulent transactions in financial datasets to help banks and payment platforms reduce financial losses. Using machine learning models such as Random Forest, XGBoost, Logistic Regression, and SVM with SMOTE for class imbalance handling, we aim to accurately identify high-risk transactions before they impact customers.

##  Dataset & Sources
- **Source**: [Kaggle Synthetic Fraud Detection Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Size**: 1,000,000 transactions, 21 features
- **Key Features**: transaction_amount, transaction_type, account_balance, device_type, location, merchant_category, card_type, card_age, risk_score, fraud_label
- **Notes**: Timestamp was split into hour and day-of-week features. Missing values handled with median (numeric) or most frequent (categorical) imputation.

##  Key Findings & Results
- Fraud is highly imbalanced: only ~1-2% of transactions are fraudulent.
- Random Forest and XGBoost achieved near-perfect AUC-ROC on test set, but may overfit.
- Logistic Regression is more stable across cross-validation, though with lower AUC (0.76), indicating some sensitivity to class imbalance.
- Key risk indicators include transaction amount, transaction type, and account balance.
- Early detection of fraudulent transactions could save significant operational costs and reduce customer risk exposure.

##  Technologies Used
- **Programming**: Python 3
- **Libraries**: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, matplotlib, seaborn
- **Environment**: Kaggle Notebooks, can be run in Jupyter or Google Colab

##  Project Structure
fraud_detection_project/
│
├─ fraud_detection_pipeline.py # Main Python script with preprocessing and ML pipeline
├─ README.md # This file
├─ data/ # Folder to store dataset
└─ visuals/ # Optional folder for charts/plots


## 📈 Visualizations
![Fraud Distribution](visuals/fraud_distribution.png)  
*Shows the severe class imbalance between fraudulent and non-fraudulent transactions.*

![ROC Curves](visuals/roc_curves.png)  
*ROC curves for Random Forest, XGBoost, Logistic Regression, and SVM, highlighting model performance.*

##  How to Use This Project
1. **Dataset**: Download from Kaggle or place CSV in `/data` folder.
2. **Main Script**: Run `fraud_detection_pipeline.py` in Python 3 environment.
3. **Dependencies**: Install required packages:
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
Run Analysis: Execute the script to perform preprocessing, train/test split, SMOTE oversampling, model training, and evaluation.

 Future Work

Implement real-time transaction scoring for online banking systems.

Explore feature engineering from device and IP metadata.

Test ensemble methods and deep learning approaches for better generalization.

Evaluate temporal cross-validation to prevent leakage from future data.
