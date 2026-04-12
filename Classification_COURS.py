"""
Predicting employees leaves
---------------------------------

Use classification models to predict if an employee is going
to leave the company or not.

Includes:
- Data loading and basic exploration
- Preprocessing: encoding categorical variables
- Random Forest and Most frequent value (mode) for baseline
- Accuracy and confusion matrix evaluation
- Accuracy bar chart and confusion matrices plots

- TODO: Your job is to implement and set up new models on the run function,
    this file was prepared so that you could focus on exploring the models
    no need to change anything else (except if you want further exploring)


Data used:
* satisfaction_level
* last_evaluation
* number_project
* average_monthly_hours
* time_spend_company
* work_accident
* left (y to predict)
* promotion_last_5years
* Department
* salary
"""

# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# Function Definitions
# ----------------------------

def load_data(path):
    """Load and inspect the dataset."""
    df = pd.read_csv(path)
    print("\nFirst lines of the dataset:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nStatistical summary:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    return df

def preprocess_data(df):
    """Extract features and labels from the dataset."""
    X = df[[
        'satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',
        'time_spend_company',
        'Work_accident',
        'promotion_last_5years'
    ]]
    y = df['left']
    return X, y



def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a classification model.
    Print accuracy and confusion matrix.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.3f}")
    print(f"Confusion Matrix for {name}:")
    print(confusion_matrix(y_test, y_pred))
    # classification_report provide an overview over several
    # metrics frequently used in classification problems
    # no need to know all now, but you can check them out if you are curious
    print(classification_report(y_test, y_pred))
    return name, acc, confusion_matrix(y_test, y_pred)



def plot_confusion_matrices(results):
    """Plot confusion matrices of all models."""
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    for ax, (name, _, cm) in zip(axes, results):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_accuracy_bar(results):
    """Bar plot of model accuracies."""
    names = [r[0] for r in results]
    scores = [r[1] for r in results]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=scores)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()



# ----------------------------
# Main Execution
# ----------------------------

def run():
    df = load_data('Resources/HR_dataset.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Define models
    """ ..TODO:  
    Here you can implement your models following given examples, 
        feel free to explore their hyperparameters and their influence in the performance.
        Keep in mind, in regression the best model is the one with lowest error (metric value)
    """
    models = [
        ("Random Forest", RandomForestClassifier(random_state=0)),
        ("Baseline (Most Frequent)", DummyClassifier(strategy='most_frequent'))
    ]

    results = []
    for name, model in models:
        results.append(train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test))

    plot_confusion_matrices(results)
    plot_accuracy_bar(results)

run()