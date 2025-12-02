import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import mlflow
import mlflow.sklearn

MODEL_PATH = "models/best_modle.pkl"
target = "price"

#load
df = pd.read_csv("data/processed/train.csv")
print('\n shape of training data:', df.shape)

X = df.drop(columns=["price"])
y = df[target]
X = pd.get_dummies(X, drop_first=True)

#train test split#split data --- Train-test split, 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Model Training")
models = {
    'LinearRegression':LinearRegression(),
    'Decision_Tree':DecisionTreeRegressor(),
    'Random_Forest':RandomForestRegressor()
}


results = {}
best_r2 = -float('inf')
best_model_name =None
best_model_object = None
mlflow.set_experiment("Mlflow Quickstart ")
#Train + Mlflow 
for name, model in models.items():
    with mlflow.start_run(run_name=f"{name}_run"):
        print(f"\nTraining {name}")
        
        #Train model
        model.fit(X_train, y_train)
        
        #predictions
        y_pred = model.predict(X_test)
        
        #metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        #log parameters
        mlflow.log_param("model_name", name)
        mlflow.log_param("target", target)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("feature_count", X_train.shape[1])
        
        #log metrics
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        
        #log artifacts
        mlflow.sklearn.log_model(model, "model")
        results[name] = r2
        print(f"R2 score: {r2}")
        
   
best_model_name = max(results, key=results.get)
print(f"Best performing model: {best_model_name} with R2 score: {results[best_model_name]}")

best_model = models[best_model_name]
joblib.dump(best_model, MODEL_PATH)
print(f"Best Model Name: {best_model_name}")
print(f"Best Model saved to {MODEL_PATH}")