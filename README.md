# Fraud Detection Model - Financial Transactions

## Project Overview
This project implements a machine learning model to detect fraudulent transactions in financial data. The model analyzes transaction patterns to identify potential fraud and provides actionable insights for fraud prevention.

## Dataset Information
- **Size**: 6,362,620 rows × 10 columns
- **Source**: Financial transaction simulation data
- **Target Variable**: `isFraud` (binary classification)

## Features
- `step`: Time unit (1 step = 1 hour, 744 steps = 30 days)
- `type`: Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)
- `amount`: Transaction amount in local currency
- `nameOrig`: Origin customer ID
- `oldbalanceOrg`: Initial balance before transaction
- `newbalanceOrg`: New balance after transaction
- `nameDest`: Destination customer ID
- `oldbalanceDest`: Initial destination balance
- `newbalanceDest`: New destination balance
- `isFraud`: Fraud indicator (target variable)
- `isFlaggedFraud`: Business rule flag (>200,000 transfer)

## Project Structure
```
False_Transactions/
├── data/
│   └── fraud.csv
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset to `data/fraud.csv`
3. Run python file: `src/fraud_analysis.py`

## Methodology
1. **Data Cleaning**: Handle missing values, outliers, and multicollinearity
2. **Feature Engineering**: Create relevant features for fraud detection
3. **Model Development**: Train multiple ML models and select the best performer
4. **Performance Evaluation**: Use appropriate metrics for imbalanced data
5. **Feature Importance**: Identify key factors predicting fraud
6. **Actionable Insights**: Provide recommendations for fraud prevention 