import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yaml
from src.logger import get_logger
from src.custom_exceptions import AppException

logger = get_logger(__name__)

def main():
    try:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)["preprocessing"]
        
        logger.info("Starting preprocessing")
        
        train_df = pd.read_csv("data/raw/train.csv")
        test_df = pd.read_csv("data/raw/test.csv")
        
        X_train = train_df.drop("target", axis=1)
        y_train = train_df["target"]
        X_test = test_df.drop("target", axis=1)
        y_test = test_df["target"]
        
        scaler_type = params["scaler_type"]
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        train_scaled["target"] = y_train.values
        test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        test_scaled["target"] = y_test.values
        
        os.makedirs("data/interim", exist_ok=True)
        train_scaled.to_csv("data/interim/train_scaled.csv", index=False)
        test_scaled.to_csv("data/interim/test_scaled.csv", index=False)
        
        logger.info(f"Applied {scaler_type} scaling")
        logger.info("Preprocessing completed")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise AppException("Failed in preprocessing stage", e)

if __name__ == "__main__":
    main()