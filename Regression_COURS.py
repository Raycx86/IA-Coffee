"""
Medical Insurance Cost Prediction
---------------------------------

This script analyzes the insurance dataset using several regression models and evaluates
their performance. It uses RMSE, MSE and MAE metrics to compare model effectiveness.

Includes:
- Data preprocessing (with loading and inspection)
- Baseline model
- Random Forest & Mean value for baselines
- Metrics calculation
- Bar chart comparison


- TODO: Your job is to implement and set up new models on the run function,
    this file was prepared so that you could focus on exploring the models
    no need to change anything else (except if you want further exploring)

Data used :
*age: age of primary beneficiary
*sex: insurance contractor gender, female, male
*bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
*children: Number of children covered by health insurance / Number of dependents
*smoker: Smoking
*region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest

"""

# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error



# --- Data Loading & Preprocessing ---
def load_and_preprocess(filepath):
    """
    Loads and preprocesses the insurance dataset.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        X_train, X_test, y_train, y_test (tuple): Processed train/test sets.
        df (pd.DataFrame): Original dataframe with encoded features.
    """
    df = pd.read_csv(filepath)

    # Some info about dataset:
    df.head() # print First lines
    df.info() # general infos like number of lines, columns and types
    df.describe() # statistical description of the data
    df.isnull().sum() # number of missing values per column

    # Encode categorical variables
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])        # male=1, female=0
    df['smoker'] = le.fit_transform(df['smoker'])  # yes=1, no=0
    df['region'] = le.fit_transform(df['region'])  # northeast=0, etc.

    X = df.drop('charges', axis=1)  # Features
    y = df['charges']               # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test, df



# --- Model Evaluation ---
def evaluate_model(name, model, X_train, y_train, X_test, y_test, results):
    """
    Trains a model and stores evaluation metrics.

    Parameters:
        name (str): Model name for results.
        model (sklearn model): Regressor to train.
        results (dict): Dictionary to store results.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {'RMSE': rmse, 'MSE': mse, 'MAE': mae}

    print(f"\n{name} Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MSE : {mse:.2f}")
    print(f"  MAE : {mae:.2f}")


# --- Metric Plotting ---
def plot_metrics(results):
    """
    Displays a bar chart for all metrics (RMSE, MSE, MAE).

    Parameters:
        results (dict): Collected results per model.
    """
    metrics = ['RMSE', 'MSE', 'MAE']
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

    for idx, metric in enumerate(metrics):
        scores = [results[model][metric] for model in results]
        axes[idx].bar(results.keys(), scores, color='skyblue')
        axes[idx].set_title(metric)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].set_ylabel(metric)

    plt.tight_layout()
    plt.show()




# --- Baseline Model ---
class MeanPredictor:
    """
    Baseline model that always predicts the mean of training targets.
    In AI, we always starts with "dumb" strategies for predictions,
    This way we can check the actual need for a model and its relative complexity.
    Consider it, if you can achieve close to 0 error without a model and a single line
    of code, why bother?
    """
    def fit(self, X, y):
        self.mean_ = y.mean()

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_)




# ----------------------------
# Main Execution
# ----------------------------


def run():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, df = load_and_preprocess("Resources/insurance.csv")

    results = {}  # Dictionary to collect evaluation metrics

    # Define models
    """ ..TODO:  
    Here you can implement your models following given examples, 
        feel free to explore their hyperparameters and their influence in the performance.
        Keep in mind, in regression the best model is the one with lowest error (metric value)
    """
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1),
        "Baseline (Mean)": MeanPredictor()
    }

    # Evaluate each model
    for name, model in models.items():
        evaluate_model(name, model, X_train, y_train, X_test, y_test, results)

    # Plot performance comparison
    plot_metrics(results)



run()
