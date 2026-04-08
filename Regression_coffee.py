import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

data_file = "Coffe_sales.csv"


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    features = [
        "hour_of_day",
        "coffee_name",
        "Time_of_Day",
        "Weekdaysort",
        "Monthsort",
    ]

    X = df[features]
    y = df["money"]

    X_encoded = pd.get_dummies(
        X, columns=["coffee_name", "Time_of_Day"], drop_first=True, dtype=int
    )

    X_encoded = X_encoded.select_dtypes(include=[np.number])

    return train_test_split(X_encoded, y, test_size=0.2, random_state=42)

def evaluate_model(name, model, X_train, y_train, X_test, y_test, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MSE": mse, "MAE": mae}

    print(f"{name:22} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f}")


def plot_metrics(results):
    metrics = ["RMSE", "MAE"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for idx, metric in enumerate(metrics):
        scores = [results[model][metric] for model in results]
        axes[idx].bar(results.keys(), scores, color="teal")
        axes[idx].set_title(f"Model Comparison: {metric}")
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].set_ylabel(metric)

    plt.tight_layout()
    plt.show()


class MeanPredictor:
    def fit(self, X, y):
        self.mean_ = y.mean()

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_)


def run():
    X_train, X_test, y_train, y_test = load_and_preprocess(data_file)
    results = {}

    models = {
        "Baseline (Mean)": MeanPredictor(),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=1),
        "Polynomial Reg": make_pipeline(PolynomialFeatures(2), LinearRegression()),
        "SVR (Support Vector)": SVR(kernel="rbf"),
    }

    print(f"{'Model':22} | {'RMSE':8} | {'MAE':8}")
    print("-" * 45)

    for name, model in models.items():
        evaluate_model(name, model, X_train, y_train, X_test, y_test, results)

    plot_metrics(results)


run()
