import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def load_data(data_file):
    df = pd.read_csv(data_file)

    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    df = df.dropna()
    return df


def preprocess_data(df):
    # Features used to predict the coffee name
    features = [
        "money",
        "hour_of_day",
        "Weekdaysort",
        "Monthsort",
    ]

    X = df[features]
    y = df["coffee_name"].astype(str)

    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)
    X_encoded = X_encoded.select_dtypes(include=[np.number])

    return train_test_split(X_encoded, y, test_size=0.2, random_state=0, stratify=y)


def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return name, acc, confusion_matrix(y_test, y_pred)


def plot_results(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, (name, _, cm) in zip(axes, results):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Confusion Matrix: {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


def run(data_file):
    df = load_data(data_file)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = [
        ("Baseline", DummyClassifier(strategy="most_frequent")),
        ("Decision Tree", DecisionTreeClassifier(random_state=0)),
        ("Random Forest", RandomForestClassifier(random_state=0)),
        ("XGBoost", XGBClassifier(eval_metric="mlogloss")),
        ("Naive Bayes", GaussianNB()),
    ]

    results = []
    for name, model in models:
        results.append(
            train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test)
        )

    plot_results(results)


coffee_data_file = "Coffe_sales.csv"
run(coffee_data_file)
