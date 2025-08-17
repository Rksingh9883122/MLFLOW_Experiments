import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define model parameters
max_depth = 55
n_estimators = 88

# Set experiment name
mlflow.set_experiment('MLOPS-Exp2')

with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Create and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plot_path = "Confusion-matrix.png"
    plt.savefig(plot_path)
    plt.close()

    # Log artifacts
    mlflow.log_artifact(plot_path)

    # Log current script if available
    try:
        mlflow.log_artifact(__file__)
    except NameError:
        print("Note: __file__ is not defined (likely running in interactive mode).")

    # Set tags
    mlflow.set_tags({"Author": "Raj", "Project": "Wine Classification"})

    # Log model
    mlflow.sklearn.log_model(rf, artifact_path="Random-Forest-Model")

    print(f"Model accuracy: {accuracy:.4f}")
