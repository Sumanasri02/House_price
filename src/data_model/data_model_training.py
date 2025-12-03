import os
#  DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "Sumanasri02"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a674ec54a8dd4cbf1609cb20ee26e983bc5cc7dc"


os.environ["MLFLOW_ENABLE_LOGGED_MODEL"] = "false"

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import mlflow
import mlflow.sklearn

MODEL_PATH = "models/best_modle.pkl"
TARGET = "price"

#  dagshub tracking uri---
mlflow.set_tracking_uri("https://dagshub.com/Sumanasri02/House_Price_Cloud.mlflow")
mlflow.set_experiment("House_Price_Cloud")

# Load data
df = pd.read_csv("data/processed/train.csv")
print("Shape of training data:", df.shape)

X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True)
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to train
models = {
    'LinearRegression': {"model": LinearRegression(), "params": {}},
    'DecisionTree': {
        "model": DecisionTreeRegressor(),
        "params": {
            "max_depth": [5, 10, 20, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 6]
        }
    },
    'RandomForest': {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }
    }
}

best_r2 = -float('inf')
best_model_name = None
best_model_object = None

#  MLflow
for name, config in models.items():
    model = config["model"]
    params = config["params"]

    with mlflow.start_run(run_name=f"{name}_run"):
        print(f"\nTraining {name}")
        # Hyperparameter tuning
        if params:  
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=5,
                cv=3,
                scoring="r2",
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            mlflow.log_params(search.best_params_)
        else:
            model.fit(X_train, y_train)

        # Predictions & metrics
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # Log params & metrics
        mlflow.log_param("model_name", name)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("feature_count", X_train.shape[1])

        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"R2 Score: {r2}")

        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_object = model

# Cross-validation on best model
cv_scores = cross_val_score(best_model_object, X_train, y_train, cv=5, scoring="r2")
print("\nCross-validation on best model")
print(f"CV Scores: {cv_scores}")
print(f"Mean CV R2: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

with mlflow.start_run(run_name="Cross_Validation_Run"):
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("cv_mean_r2", cv_scores.mean())
    mlflow.log_metric("cv_std_r2", cv_scores.std())

# Save best model locally
joblib.dump(best_model_object, MODEL_PATH)
print(f"\nBest Model: {best_model_name} saved to {MODEL_PATH}")
