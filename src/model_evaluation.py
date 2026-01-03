import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from dvclive import Live
import yaml
from src.logger import get_logger
from src.custom_exceptions import AppException

logger = get_logger(__name__)

def main():
    try:
        logger.info("Starting model evaluation")
        
        test_df = pd.read_csv("data/interim/test_scaled.csv")
        X_test = test_df.drop("target", axis=1)
        y_test = test_df["target"]
        
        with open("models/stacking_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results = [{"learner": "stacking_ensemble", "accuracy": acc, "f1": f1}]
        
        for name, estimator in model.named_estimators_.items():
            estimator.fit(X_test, y_test)
            pred = estimator.predict(X_test)
            results.append({
                "learner": name,
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred)
            })
        
        comparison_df = pd.DataFrame(results)
        os.makedirs("reports", exist_ok=True)
        comparison_df.to_csv("reports/comparison_table.csv", index=False)
        
        with open("reports/report.txt", "w") as f:
            f.write("Stacking Ensemble Results\n")
            f.write("Base Learners: Decision Tree, Logistic Regression, SVC\n")
            f.write("Meta Learner: Random Forest\n\n")
            f.write(comparison_df.to_string(index=False))
        
        # DVCLive tracking
        os.makedirs("dvclive/plots", exist_ok=True)
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", acc)
            live.log_metric("f1_score", f1)
            live.log_sklearn_plot("confusion_matrix", y_test, y_pred)
            live.log_sklearn_plot("roc", y_test, model.predict_proba(X_test)[:, 1])
            live.log_artifact("reports/comparison_table.csv", type="table")
            live.log_artifact("reports/report.txt", type="text")
        
        logger.info(f"Ensemble → Accuracy: {acc:.4f}, F1: {f1:.4f}")
        logger.info("Model evaluation completed")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise AppException("Failed in model_evaluation stage", e)

if __name__ == "__main__":
    main()