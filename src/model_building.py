import os
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
import yaml
from src.logger import get_logger
from src.custom_exceptions import AppException

logger = get_logger(__name__)

def main():
    try:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)["model_building"]
        
        logger.info("Starting model building")
        
        train_df = pd.read_csv("data/interim/train_scaled.csv")
        X_train = train_df.drop("target", axis=1)
        y_train = train_df["target"]
        
        rs = params["random_state"]
        estimators = []
        
        bl = params["base_learners"]
        if "decision_tree" in bl:
            estimators.append(("dt", DecisionTreeClassifier(max_depth=bl["decision_tree"]["max_depth"], random_state=rs)))
        if "logistic_regression" in bl:
            estimators.append(("lr", LogisticRegression(C=bl["logistic_regression"]["C"], random_state=rs)))
        if "svc" in bl:
            estimators.append(("svc", SVC(C=bl["svc"]["C"], kernel=bl["svc"]["kernel"], probability=True, random_state=rs)))
        
        meta = RandomForestClassifier(n_estimators=params["meta_learner"]["n_estimators"], random_state=rs)
        
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=StratifiedKFold(n_splits=params["k_folds"]),
            stack_method='predict_proba',
            n_jobs=-1,
            passthrough=True
        )
        
        model.fit(X_train, y_train)
        
        os.makedirs("models", exist_ok=True)
        with open("models/stacking_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"Trained stacking model with {len(estimators)} base learners")
        logger.info("Model building completed")
        
    except Exception as e:
        logger.error(f"Model building failed: {e}")
        raise AppException("Failed in model_building stage", e)

if __name__ == "__main__":
    main()