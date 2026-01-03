import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import yaml
from src.logger import get_logger
from src.custom_exceptions import AppException

logger = get_logger(__name__)

def main():
    try:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)["data_ingestion"]
        
        logger.info("Starting data ingestion")
        
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        
        train_df, test_df = train_test_split(
            df,
            test_size=params["test_size"],
            random_state=params["random_state"],
            stratify=df["target"]
        )
        
        os.makedirs("data/raw", exist_ok=True)
        train_df.to_csv("data/raw/train.csv", index=False)
        test_df.to_csv("data/raw/test.csv", index=False)
        
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)} samples saved")
        logger.info("Data ingestion completed")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise AppException("Failed in data_ingestion stage", e)

if __name__ == "__main__":
    main()