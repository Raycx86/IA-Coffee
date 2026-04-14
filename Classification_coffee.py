import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def load_data(data_file):
    df = pd.read_csv(data_file)

    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    df = df.dropna()
    return df


def preprocess_data(df):
    features = [
        "money",
        "hour_of_day",
        "Weekdaysort",
        "Monthsort",
    ]

    X = df[features]
    y = df["coffee_name"].astype(str)

    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return (
        train_test_split(
            X_encoded,
            y_encoded,
            test_size=0.2,
            random_state=0,
            stratify=y_encoded,
        ),
        label_encoder,
    )


def train_and_evaluate_model(
    name, model, X_train, X_test, y_train, y_test, label_encoder
):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc:.3f}")

    y_test_names = label_encoder.inverse_transform(y_test)
    y_pred_names = label_encoder.inverse_transform(y_pred)

    print(classification_report(y_test_names, y_pred_names, zero_division=0))

    return name, acc, confusion_matrix(y_test, y_pred)


def plot_accuracy_bar(results):
    names = [r[0] for r in results]
    scores = [r[1] for r in results]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=scores)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()


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
    (X_train, X_test, y_train, y_test), label_encoder = preprocess_data(df)
    results = []

    gb_param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_iter": [100, 500],
        "max_depth": [5, 10, 15],
        "l2_regularization": [0, 0.1, 1.0],
    }

    grid_search = GridSearchCV(
        HistGradientBoostingClassifier(random_state=1, early_stopping=True),
        gb_param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_gb_model = grid_search.best_estimator_

    models = [
        ("Baseline", DummyClassifier(
            strategy="most_frequent"
        )
         ),
        ("Decision Tree", DecisionTreeClassifier(random_state=0)),
        ("Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=1,
                n_jobs=-1,
            ),
        ),
        ("Gradient Boosting", best_gb_model),
        ("XGBoost",
            XGBClassifier(
                eval_metric="mlogloss",
                objective="multi:softprob",
                num_class=len(label_encoder.classes_),
                random_state=0,
            ),
        ),
        ("Naive Bayes", GaussianNB()),
    ]

    for name, model in models:
        results.append(
            train_and_evaluate_model(
                name, model, X_train, X_test, y_train, y_test, label_encoder
            )
        )

    plot_results(results)
    plot_accuracy_bar(results)


coffee_data_file = "Coffe_sales.csv"
run(coffee_data_file)

## TODO: Commentaire
## TODO:
