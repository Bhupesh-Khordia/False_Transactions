#!/usr/bin/env python3
"""
Fraud Detection Model - Financial Transactions
Comprehensive analysis for fraud detection in financial transactions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

def load_and_explore_data():
    """Load and explore the dataset"""
    print("="*60)
    print("1. DATA LOADING AND INITIAL EXPLORATION")
    print("="*60)
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('../data/fraud.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # Check for missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_values,
        'Missing_Percentage': missing_percentage
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Check for duplicates
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    
    # Check target variable distribution
    print(f"\nTarget variable distribution:")
    print(df['isFraud'].value_counts())
    print(f"Fraud percentage: {(df['isFraud'].sum() / len(df)) * 100:.4f}%")
    
    return df

def clean_and_preprocess_data(df):
    """Clean and preprocess the data"""
    print("\n" + "="*60)
    print("2. DATA CLEANING AND PREPROCESSING")
    print("="*60)
    
    # Create a copy for preprocessing
    df_clean = df.copy()
    
    # Handle missing values in balance columns
    balance_columns = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in balance_columns:
        df_clean[col] = df_clean[col].fillna(0)
    
    # Convert type to categorical
    df_clean['type'] = df_clean['type'].astype('category')
    
    # Create derived features
    df_clean['balance_diff_orig'] = df_clean['newbalanceOrig'] - df_clean['oldbalanceOrg']
    df_clean['balance_diff_dest'] = df_clean['newbalanceDest'] - df_clean['oldbalanceDest']
    df_clean['amount_balance_ratio'] = df_clean['amount'] / (df_clean['oldbalanceOrg'] + 1)
    
    # Outlier detection and handling
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    # Check outliers in amount
    outliers_amount, lb_amount, ub_amount = detect_outliers(df_clean, 'amount')
    print(f"Outliers in amount: {len(outliers_amount)} ({len(outliers_amount)/len(df_clean)*100:.2f}%)")
    
    # Cap outliers instead of removing them
    df_clean['amount_capped'] = df_clean['amount'].clip(upper=ub_amount)
    
    print("Data cleaning completed.")
    print(f"Shape after cleaning: {df_clean.shape}")
    
    return df_clean

def exploratory_data_analysis(df_clean):
    """Perform exploratory data analysis"""
    print("\n" + "="*60)
    print("3. EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Transaction type analysis
    print("\nTransaction Type Analysis:")
    fraud_by_type = df_clean.groupby('type')['isFraud'].agg(['count', 'sum', 'mean']).round(4)
    fraud_by_type.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
    fraud_by_type = fraud_by_type.sort_values('Fraud_Rate', ascending=False)
    print(fraud_by_type)
    
    # Amount analysis
    print("\nAmount Analysis:")
    amount_stats = df_clean.groupby('isFraud')['amount'].agg(['mean', 'median', 'std', 'min', 'max'])
    print(amount_stats)
    
    # Balance analysis
    print("\nBalance Analysis:")
    high_balance_ratio_fraud = df_clean[(df_clean['isFraud']==1) & (df_clean['amount_balance_ratio'] > 0.8)]
    print(f"High balance ratio fraud cases: {len(high_balance_ratio_fraud)} ({len(high_balance_ratio_fraud)/len(df_clean[df_clean['isFraud']==1])*100:.1f}%)")
    
    return fraud_by_type

def feature_engineering(df_clean):
    """Create engineered features"""
    print("\n" + "="*60)
    print("4. FEATURE ENGINEERING")
    print("="*60)
    
    def create_features(df):
        df_featured = df.copy()
        
        # Transaction frequency features
        df_featured['orig_freq'] = df_featured.groupby('nameOrig')['step'].transform('count')
        df_featured['dest_freq'] = df_featured.groupby('nameDest')['step'].transform('count')
        
        # Amount-based features
        df_featured['amount_log'] = np.log1p(df_featured['amount'])
        df_featured['amount_sqrt'] = np.sqrt(df_featured['amount'])
        
        # Balance-based features
        df_featured['balance_ratio_orig'] = df_featured['amount'] / (df_featured['oldbalanceOrg'] + 1)
        df_featured['balance_ratio_dest'] = df_featured['amount'] / (df_featured['oldbalanceDest'] + 1)
        
        # Transaction type encoding
        type_encoder = LabelEncoder()
        df_featured['type_encoded'] = type_encoder.fit_transform(df_featured['type'])
        
        # Time-based features
        df_featured['hour_of_day'] = df_featured['step'] % 24
        df_featured['day_of_week'] = (df_featured['step'] // 24) % 7
        
        # Interaction features
        df_featured['amount_balance_interaction'] = df_featured['amount'] * df_featured['balance_ratio_orig']
        
        return df_featured
    
    # Apply feature engineering
    df_featured = create_features(df_clean)
    print(f"Shape after feature engineering: {df_featured.shape}")
    print(f"New features created: {list(set(df_featured.columns) - set(df_clean.columns))}")
    
    return df_featured

def multicollinearity_analysis(df_featured):
    """Analyze multicollinearity"""
    print("\n" + "="*60)
    print("5. MULTICOLLINEARITY ANALYSIS")
    print("="*60)
    
    # Select numerical features for correlation analysis
    numerical_features = df_featured.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('isFraud')  # Remove target variable
    
    # Calculate correlation matrix
    correlation_matrix = df_featured[numerical_features].corr()
    
    # Find highly correlated features
    threshold = 0.8
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                       correlation_matrix.columns[j], 
                                       correlation_matrix.iloc[i, j]))
    
    print(f"Highly correlated feature pairs (|correlation| > {threshold}):")
    for pair in high_corr_pairs:
        print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

def prepare_model_data(df_featured):
    """Prepare data for modeling"""
    print("\n" + "="*60)
    print("6. MODEL PREPARATION")
    print("="*60)
    
    # Select features for modeling
    feature_columns = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud',
        'balance_diff_orig', 'balance_diff_dest', 'amount_balance_ratio',
        'amount_capped', 'orig_freq', 'dest_freq', 'amount_log', 
        'amount_sqrt', 'balance_ratio_orig', 'balance_ratio_dest',
        'type_encoded', 'hour_of_day', 'day_of_week',
        'amount_balance_interaction'
    ]
    
    X = df_featured[feature_columns]
    y = df_featured['isFraud']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Fraud rate: {y.mean():.4f}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training fraud rate: {y_train.mean():.4f}")
    print(f"Test fraud rate: {y_test.mean():.4f}")
    
    # Handle class imbalance using SMOTE
    print("\nOriginal training set distribution:")
    print(y_train.value_counts())
    
    # Apply SMOTE for oversampling
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print("\nBalanced training set distribution:")
    print(y_train_balanced.value_counts())
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nScaled training set shape: {X_train_scaled.shape}")
    print(f"Scaled test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, feature_columns

def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train_balanced, y_test):
    """Train and evaluate multiple models"""
    print("\n" + "="*60)
    print("7. MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    # Define models to test
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train_balanced)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy: {results[name]['accuracy']:.4f}")
        print(f"  Precision: {results[name]['precision']:.4f}")
        print(f"  Recall: {results[name]['recall']:.4f}")
        print(f"  F1-Score: {results[name]['f1_score']:.4f}")
        print(f"  ROC-AUC: {results[name]['roc_auc']:.4f}")
    
    # Compare model performance
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[name]['accuracy'] for name in results.keys()],
        'Precision': [results[name]['precision'] for name in results.keys()],
        'Recall': [results[name]['recall'] for name in results.keys()],
        'F1-Score': [results[name]['f1_score'] for name in results.keys()],
        'ROC-AUC': [results[name]['roc_auc'] for name in results.keys()]
    })
    
    print("\nModel Performance Comparison:")
    print(results_df.round(4))
    
    # Find best model
    best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model based on F1-Score: {best_model_name}")
    
    return results, best_model_name, best_model

def feature_importance_analysis(best_model, feature_columns, best_model_name):
    """Analyze feature importance"""
    print("\n" + "="*60)
    print("8. FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Feature importance analysis
    if hasattr(best_model, 'feature_importances_'):
        # For tree-based models
        feature_importance = best_model.feature_importances_
    else:
        # For linear models
        feature_importance = np.abs(best_model.coef_[0])
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("Top 15 Most Important Features:")
    print(feature_importance_df.head(15))
    
    return feature_importance_df

def business_insights_and_recommendations(df_featured, fraud_by_type, results, best_model_name):
    """Provide business insights and recommendations"""
    print("\n" + "="*60)
    print("9. BUSINESS INSIGHTS AND RECOMMENDATIONS")
    print("="*60)
    
    # Key insights from the analysis
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("="*50)
    
    # 1. Transaction type analysis
    print("\n1. TRANSACTION TYPE ANALYSIS:")
    print(fraud_by_type)
    print("\nRecommendation: Focus monitoring on TRANSFER and CASH-OUT transactions")
    
    # 2. Amount analysis
    print("\n2. AMOUNT ANALYSIS:")
    fraud_amount_stats = df_featured[df_featured['isFraud']==1]['amount'].describe()
    legitimate_amount_stats = df_featured[df_featured['isFraud']==0]['amount'].describe()
    print(f"Fraudulent transactions - Mean: {fraud_amount_stats['mean']:.2f}, Median: {fraud_amount_stats['50%']:.2f}")
    print(f"Legitimate transactions - Mean: {legitimate_amount_stats['mean']:.2f}, Median: {legitimate_amount_stats['50%']:.2f}")
    print("\nRecommendation: Set up alerts for transactions above certain thresholds")
    
    # 3. Balance analysis
    print("\n3. BALANCE ANALYSIS:")
    high_balance_ratio_fraud = df_featured[(df_featured['isFraud']==1) & (df_featured['balance_ratio_orig'] > 0.8)]
    print(f"High balance ratio fraud cases: {len(high_balance_ratio_fraud)} ({len(high_balance_ratio_fraud)/len(df_featured[df_featured['isFraud']==1])*100:.1f}%)")
    print("\nRecommendation: Monitor transactions that drain more than 80% of account balance")
    
    # 4. Time-based patterns
    print("\n4. TIME-BASED PATTERNS:")
    fraud_by_hour = df_featured[df_featured['isFraud']==1].groupby('hour_of_day').size()
    peak_hours = fraud_by_hour.nlargest(3)
    print(f"Peak fraud hours: {peak_hours.index.tolist()}")
    print("\nRecommendation: Increase monitoring during peak hours")
    
    # 5. Model performance insights
    print("\n5. MODEL PERFORMANCE INSIGHTS:")
    print(f"Best model: {best_model_name}")
    print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"Precision: {results[best_model_name]['precision']:.4f}")
    print(f"Recall: {results[best_model_name]['recall']:.4f}")
    print("\nRecommendation: Deploy the model with regular retraining")
    
    # 6. Prevention strategies
    print("\n6. PREVENTION STRATEGIES:")
    print("- Implement real-time transaction monitoring")
    print("- Set up multi-factor authentication")
    print("- Use behavioral analysis for unusual patterns")
    print("- Implement transaction limits based on account history")
    print("- Regular security audits and system updates")
    
    # 7. Monitoring and evaluation
    print("\n7. MONITORING AND EVALUATION:")
    print("- Track model performance metrics over time")
    print("- Monitor false positive and false negative rates")
    print("- Conduct regular model retraining with new data")
    print("- Implement A/B testing for new fraud detection rules")
    print("- Regular review of fraud patterns and trends")

def main():
    """Main function to run the complete analysis"""
    print("FRAUD DETECTION MODEL - FINANCIAL TRANSACTIONS")
    print("="*60)
    
    # 1. Load and explore data
    df = load_and_explore_data()
    
    # 2. Clean and preprocess data
    df_clean = clean_and_preprocess_data(df)
    
    # 3. Exploratory data analysis
    fraud_by_type = exploratory_data_analysis(df_clean)
    
    # 4. Feature engineering
    df_featured = feature_engineering(df_clean)
    
    # 5. Multicollinearity analysis
    multicollinearity_analysis(df_featured)
    
    # 6. Prepare model data
    X_train_scaled, X_test_scaled, y_train_balanced, y_test, feature_columns = prepare_model_data(df_featured)
    
    # 7. Train and evaluate models
    results, best_model_name, best_model = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train_balanced, y_test)
    
    # 8. Feature importance analysis
    feature_importance_df = feature_importance_analysis(best_model, feature_columns, best_model_name)
    
    # 9. Business insights and recommendations
    business_insights_and_recommendations(df_featured, fraud_by_type, results, best_model_name)
    
    # 10. Conclusion
    print("\n" + "="*60)
    print("10. CONCLUSION")
    print("="*60)
    print("\nThis comprehensive fraud detection analysis has provided valuable insights into:")
    print("1. Data Quality: The dataset was well-structured with minimal missing values")
    print("2. Fraud Patterns: Clear patterns emerged in transaction types, amounts, and timing")
    print("3. Model Performance: Achieved strong performance with appropriate handling of class imbalance")
    print("4. Feature Importance: Identified key factors that predict fraudulent transactions")
    print("5. Actionable Insights: Provided concrete recommendations for fraud prevention")
    print("\nThe model can be deployed in production with regular monitoring and updates to maintain effectiveness against evolving fraud patterns.")

if __name__ == "__main__":
    main() 