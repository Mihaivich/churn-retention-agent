import pandas as pd
import argparse
import sys

def validate_data(staged_data_path):
    df = pd.read_csv(f"{staged_data_path}/data.csv")
    
    # 1. Schema Check: Expected columns
    expected_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'Churn']
    if not all(col in df.columns for col in expected_cols):
        print("Schema validation failed: Missing expected columns.")
        sys.exit(1)
        
    # 2. Missing Values Check
    if df.isnull().sum().any():
        print("Validation failed: Data contains missing values after ingestion.")
        sys.exit(1)

    # 3. Distribution Check (Example)
    if not df['Churn'].nunique() == 2:
        print("Validation failed: Target column 'Churn' is not binary.")
        sys.exit(1)
        
    print("Data validation successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged-data-path", required=True)
    args = parser.parse_args()
    validate_data(args.staged_data_path)