# Transaction-Risk-Escrow-Recommendation
Automated Fraud Risk Scoring System

## Project Overview
This project aims to detect and prevent fraudulent transactions in e-commerce platforms by developing an **Automated Fraud Risk Scoring System**.  
The system analyzes each transaction, assigns a risk score, and recommends using **Escrow** for high-risk cases — helping the company avoid fraud while keeping legitimate sales.

## Dataset
- ~50,000 transactions  
- Features: transaction amount, device type, card age, IP, location, merchant category, failed transactions in last 7 days, balance, etc.  
- Target variable: `fraud_label` (0 = genuine, 1 = fraud)

## Data Analysis & Insights
- Outliers were detected and removed based on transaction amount  
- No strong feature correlations → each adds unique information  
- **Top 2 most important features:**
  1. Failed transactions in the last 7 days  
  2. Risk score  
- Other features like card age, balance, and transaction distance also affect fraud likelihood

## Current Progress
- Compared several machine learning models to evaluate performance  
- Identified most influential risk features  
- Next step: develop a **Streamlit dashboard** for real-time fraud risk scoring and visualization

## Folder Structure
/data/ → dataset files (not included in repo)
/notebooks/ → data exploration and model training
/models/ → saved models
/app/ → Streamlit web app (under development)

## How to Use
1. Clone the repository  
2. Add your dataset inside the `data/` folder  
3. Run the notebooks in `/notebooks/`  
4. Use the upcoming Streamlit app for interactive visualization  

## Author
**Deniz Doğa Çalışkan**

---

