from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.data
import os

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model and hyperparameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Set experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast-cancer-rf-hp")

with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    # Log all child runs for each hyperparameter combination
    for i, params in enumerate(grid_search.cv_results_["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", grid_search.cv_results_["mean_test_score"][i])

    # Log best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_score)

    # Log training data
    train_df = X_train.copy()
    train_df["target"] = y_train
    mlflow.log_input(mlflow.data.from_pandas(train_df), context="training")

    # Log test data
    test_df = X_test.copy()
    test_df["target"] = y_test
    mlflow.log_input(mlflow.data.from_pandas(test_df), context="testing")

    # Log source code (only if __file__ is defined)
    try:
        mlflow.log_artifact(__file__)
    except NameError:
        print("Note: __file__ is undefined (likely running interactively).")

    # Log best model
    #mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path="random_forest")

    # Set tags
    mlflow.set_tag("author", "Raj Singh")
    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("tuning", "GridSearchCV")

    print("Best Parameters:", best_params)
    print("Best Accuracy:", best_score)
