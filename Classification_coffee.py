import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(data_file):
    df = pd.read_csv(data_file)

    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    return df


def preprocess_data(df):
    X = df[["money", "Time_of_Day", "Weekdaysort", "Monthsort"]]
    y = df["coffee_name"]

    X = pd.get_dummies(X, columns=["Time_of_Day"], drop_first=True, dtype=int)
    return X, y


def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.3f}")
    print(f"Confusion Matrix for {name}:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return name, acc, confusion_matrix(y_test, y_pred)


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for ax, (name, _, cm) in zip(axes, results):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()


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

def run(data_file):
    df = load_data(data_file)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    models = [
        ("Random Forest", RandomForestClassifier(random_state=0)),
        ("Baseline (Most Frequent)", DummyClassifier(strategy="most_frequent")),
        ("Decision Tree", DecisionTreeClassifier(random_state=0)),
        ("SVM", SVC(kernel="rbf", probability=True, random_state=0)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ]

    results = []
    for name, model in models:
        results.append(
            train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test)
        )

    plot_confusion_matrices(results)
    plot_accuracy_bar(results)


coffee_data_file = "Coffe_sales.csv"
run(coffee_data_file)
