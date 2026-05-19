import pandas as pd
import yaml
import argparse
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def train(staged_data_path, model_output_path):
    # Load data
    df = pd.read_csv(f"{staged_data_path}/data.csv")
    
    # Load params
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    # Define features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.drop(['SeniorCitizen'])
    categorical_features = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Define model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Hyperparameter tuning
    param_grid = {
        f'classifier__{k}': v for k, v in params['training']['hyperparameters'].items()
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    # Save model and preprocessor
    os.makedirs(model_output_path, exist_ok=True)
    joblib.dump(best_model, f"{model_output_path}/model.joblib")
    print(f"Model trained and saved. Best AUC: {grid_search.best_score_}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged-data-path", required=True)
    parser.add_argument("--model-output-path", required=True)
    args = parser.parse_args()
    train(args.staged_data_path, args.model_output_path)