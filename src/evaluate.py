import pandas as pd
import joblib
import json
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model_path, staged_data_path, metrics_path):
    # Load model and data
    model = joblib.load(f"{model_path}/model.joblib")
    df = pd.read_csv(f"{staged_data_path}/data.csv")
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    os.makedirs(metrics_path, exist_ok=True)
    
    # Save metrics to JSON
    with open(f"{metrics_path}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{metrics_path}/confusion_matrix.png")
    
    print("Evaluation complete. Metrics and plots saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--staged-data-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    args = parser.parse_args()
    evaluate(args.model_path, args.staged_data_path, args.metrics_path)