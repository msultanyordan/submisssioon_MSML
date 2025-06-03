import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    data_train = pd.read_csv("data_train_obesity_preprocess.csv")
    data_test = pd.read_csv("data_test_obesity_preprocess.csv")

    X_train = data_train.drop("Index", axis=1)
    y_train = data_train["Index"]
    X_test = data_test.drop("Index", axis=1)
    y_test = data_test["Index"]

    input_example = X_train[0:5]


with mlflow.start_run():
    # Log parameters
    n_estimators = 300
    max_depth = 28
    mlflow.autolog()
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
