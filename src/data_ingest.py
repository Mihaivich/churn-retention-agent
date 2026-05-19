import pandas as pd
import argparse
import os

def ingest_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Basic cleaning: convert 'TotalCharges' to numeric, handling errors
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}/data.csv", index=False)
    print("Data ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    ingest_data(args.input_path, args.output_path)