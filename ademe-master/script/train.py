import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8088")
mlflow.set_experiment("dpe_forever")
mlflow.autolog()

"""
[Q]
- difference entre mlflow.autolog() et mlflow.sklearn.autolog()
https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.autolog
- How does autologging work for meta estimators? (GridSeachCV)
"""

if __name__ == "__main__":
    # Load the CSV into a DataFrame called 'data'
    data = pd.read_csv("./data/dpe_tertiaire_20240307.csv")
    data = data.sample(frac=1, random_state=808).reset_index(drop=True)

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Target variable
    assert y.name == "etiquette_dpe"

    id = list(X.n_dpe)
    # X.drop(columns = ['n_dpe'], inplace= True )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    id_train = list(X_train.n_dpe)
    X_train.drop(columns=["n_dpe"], inplace=True)
    id_test = list(X_test.n_dpe)
    X_test.drop(columns=["n_dpe"], inplace=True)

    # Initialize the model
    rf = RandomForestClassifier()

    # Define the parameter grid
    param_grid = {
        "n_estimators": [300, 500],  # Number of trees
        "max_depth": [10, 15],  # Maximum depth of the trees
        "min_samples_leaf": [2, 5],
    }

    # Setup GridSearchCV with k-fold cross-validation
    cv = KFold(n_splits=5, random_state=84, shuffle=True)
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', verbose = 1)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=cv, scoring="roc_auc_ovr", verbose=1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Best parameters and best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")

    # Evaluate on the test set
    yhat = grid_search.predict(X_test)
    print(classification_report(y_test, yhat))

    probabilities = grid_search.predict_proba(X_test)

    # Calculate the maximum probability for each sample and its corresponding index
    max_probs = np.max(probabilities, axis=1)
    max_prob_indices = np.argmax(probabilities, axis=1) + 1

    predictions = pd.DataFrame()
    predictions["id"] = id_test
    predictions["prob"] = max_probs
    predictions["yhat"] = yhat
    predictions["y"] = y_test.values

    # TODO save histogram of probabilities per true category

    best_model = grid_search.best_estimator_

    feature_importances = best_model.feature_importances_

    # Get the feature names from the training DataFrame
    feature_names = X_train.columns

    # Create a dictionary mapping feature names to their importance
    importance_dict = dict(zip(feature_names, feature_importances))

    with mlflow.start_run():
        mlflow.log_dict(importance_dict, "importance_dict.json")
