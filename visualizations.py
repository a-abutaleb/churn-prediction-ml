import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set style
plt.style.use('seaborn-v0_8')  # Updated style name
sns.set_theme()  # Use seaborn's default theme

def create_class_distribution_plot():
    """Create class distribution plot"""
    labels = ['No Churn', 'Churn']
    sizes = [73.5, 26.5]
    colors = ['#2ecc71', '#e74c3c']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Customer Churn Distribution')
    plt.savefig('class_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_model_comparison_plot():
    """Create model comparison plot"""
    models = ['XGBoost', 'Random Forest', 'Logistic Regression']
    metrics = {
        'Accuracy': [0.82, 0.80, 0.78],
        'Precision': [0.78, 0.75, 0.70],
        'Recall': [0.75, 0.72, 0.68],
        'F1 Score': [0.76, 0.73, 0.69],
        'ROC AUC': [0.84, 0.82, 0.79]
    }
    
    df = pd.DataFrame(metrics, index=models)
    
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar', width=0.8)
    plt.title('Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_feature_importance_plot():
    """Create feature importance plot for XGBoost"""
    features = [
        'Contract_Month-to-month', 'InternetService_Fiber optic',
        'PaymentMethod_Electronic check', 'tenure', 'MonthlyCharges',
        'OnlineSecurity_No', 'TechSupport_No', 'DeviceProtection_No',
        'PaperlessBilling_Yes', 'StreamingTV_Yes'
    ]
    importance = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.05, 0.04]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=features)
    plt.title('Top 10 Feature Importance (XGBoost)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_roc_curves_plot():
    """Create ROC curves comparison plot"""
    # Example data (replace with actual data)
    fpr_xgb = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tpr_xgb = np.array([0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0])
    
    fpr_rf = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tpr_rf = np.array([0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    
    fpr_lr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tpr_lr = np.array([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = 0.84)')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = 0.82)')
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = 0.79)')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curves.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_monitoring_dashboard():
    """Create example monitoring dashboard plots"""
    # Example data for drift detection
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    drift_scores = np.random.normal(0.1, 0.05, 30)
    performance_metrics = {
        'Accuracy': np.random.normal(0.82, 0.02, 30),
        'Precision': np.random.normal(0.78, 0.02, 30),
        'Recall': np.random.normal(0.75, 0.02, 30)
    }
    
    # Drift detection plot
    plt.figure(figsize=(12, 4))
    plt.plot(dates, drift_scores, marker='o')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Drift Threshold')
    plt.title('Data Drift Detection Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drift Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('drift_detection.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Performance metrics plot
    plt.figure(figsize=(12, 4))
    for metric, values in performance_metrics.items():
        plt.plot(dates, values, marker='o', label=metric)
    plt.title('Model Performance Metrics Over Time')
    plt.xlabel('Date')
    plt.ylabel('Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('performance_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    create_class_distribution_plot()
    create_model_comparison_plot()
    create_feature_importance_plot()
    create_roc_curves_plot()
    create_monitoring_dashboard() 