# Stacking Ensemble Learning Pipeline

A professional, end-to-end MLOps project implementing a **stacking ensemble classifier** using scikit-learn, with full reproducibility, experiment tracking, and data versioning powered by **DVC** and **DVCLive**.

## Overview

This project demonstrates a complete machine learning pipeline for binary classification using the **Breast Cancer Wisconsin dataset** (built-in scikit-learn). It combines three diverse base learners with a meta-learner using **stacking**:

- **Base Learners**: Decision Tree, Logistic Regression, Support Vector Classifier (SVC)
- **Meta-Learner**: Random Forest
- **Blending**: K-fold cross-validated predictions for meta-features

The pipeline includes data ingestion, preprocessing, model training, evaluation, and comparison of individual vs. ensemble performance.

## Key Features

- Parameter-driven via `params.yaml` (test size, scaler, base learner hyperparameters, etc.)
- Full pipeline automation with DVC (`dvc repro`)
- Experiment tracking with DVCLive (metrics, plots, artifacts)
- Data & model versioning with DVC
- Structured logging per stage
- Exception handling in all components
- Clean separation: Code in Git, data/models in DVC

## Project Structure
<img width="653" height="656" alt="image" src="https://github.com/user-attachments/assets/6f7fffe9-c9bc-4bd8-b531-e6720be3db89" />


# Installation


## Clone the repository
```
git clone https://github.com/yourusername/stacking-ensemble-project.git
cd stacking-ensemble-project
```

## Create virtual environment
```
python -m venv .venv
```
## Activate it
## On Windows:
```
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

## Install dependencies
```
pip install -r requirements.txt
```

## Initialize DVC (if not already done)
```
dvc init
```

Usage
Run the Full Pipeline (One Command)

```
dvc repro
```

This executes all stages:

1. Data ingestion

2. Preprocessing

3. Model building (stacking ensemble)

4. Evaluation + comparison

## Outputs:

* models/stacking_model.pkl

* reports/comparison_table.csv

* reports/report.txt

* Confusion Matrix & ROC curves in dvclive/plots/sklearn/

## Run Experiments
Change parameters in params.yaml (e.g., test_size, scaler, base learner params), then:

```
git add params.yaml
git commit -m "Experiment: try minmax scaler"
dvc exp run
```

## View all experiments:

```
dvc exp show
```

## Visualize Pipeline

```
dvc dag
```

## Example Results
<img width="711" height="226" alt="image" src="https://github.com/user-attachments/assets/571eff38-3990-44fb-8a13-9e1f877b8be9" />

## Customization


* Change dataset: Modify data_ingestion.py

* Add base learners: Update model_building.py and params.yaml

* Try different meta-learner: Edit model_building.py

## Remote Storage (Optional)
Set up S3 or other remote for large data/models:
```
dvc remote add -d storage s3://your-bucket/dvc-store
dvc push
```
# Happy experimenting! Feel free to fork and extend this template for your own ensemble projects.
