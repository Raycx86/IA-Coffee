import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


def load_data(data_file):
    df = pd.read_csv(data_file)

    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    df = df.dropna()
    return df


def preprocess_data(df):
    # Ici on a choisi de mettre une liste des colomnes qu'on va utiliser pour éstimer le prix.
    features = [
        "coffee_name",
        "hour_of_day",
        "Weekdaysort",
        "Monthsort",
    ]

    X = df[features]
    y = df["money"]

    # On encore les colomne qui on du texte ou des valeurs non int car l'ia n'accepte que des int
    X_encoded = pd.get_dummies(  # get_dummies est meilleur que LabelEncoder car Label Encoder risque de mettre une importance (Nuit > jour par exemple)
        X, columns=["coffee_name"], drop_first=True, dtype=int
    )

    # On utilise donc cette variable pour ne garder que les int.
    X_encoded = X_encoded.select_dtypes(include=[np.number])

    # test_size : Pourcentage (ici 20%) du split de la base de donnée, on utilisera 20% pour tester l'IA et le reste pour l'entrainer.
    # Random_state=0 est la seed qui permet de faire toujours de la même façon quand on l'execute
    return train_test_split(X_encoded, y, test_size=0.2, random_state=0)


def evaluate_model(name, model, X_train, y_train, X_test, y_test, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MSE": mse, "MAE": mae}

    print(f"\n{name} Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MSE : {mse:.2f}")
    print(f"  MAE : {mae:.2f}")


def plot_metrics(results):
    metrics = ["RMSE", "MSE", "MAE"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

    for idx, metric in enumerate(metrics):
        scores = [results[model][metric] for model in results]
        axes[idx].bar(results.keys(), scores, color="skyblue")
        axes[idx].set_title(metric)
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].set_ylabel(metric)

    plt.tight_layout()
    plt.show()


class MeanPredictor:
    def fit(self, X, y):
        self.mean_ = y.mean()

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_)


def run(data_file):
    df = load_data(data_file)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    results = {}
    gb_param_grid = {  # Tuning (nan c'est pas une voiture lol idk il est 23h faut plus me poser de questions)
        # Entre les crochets nous voyons toutes les valeurs possible, le tuning choisira la meilleur composition.
        "learning_rate": [0.01, 0.05, 0.1],  # Vitesse à laquelle le modèle apprend
        "max_iter": [100, 500],  # Nombre d'arbres max
        "max_depth": [5, 10, 15],  # Profondeur max des arbres
        "l2_regularization": [
            0,
            0.1,
            1.0,
        ],  # Une limite pour ne pas rendre le modèle trop complexex²
    }
    grid_search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=1, early_stopping=True),
        gb_param_grid,  # On donne au modèle donner juste avant les données du dataset
        cv=3,  # Meme chause de test size, on divise les données en 3, il s'entraine sur 2 et teste sur la dernière.
        scoring="neg_mean_squared_error",  # On l'entraine en utilisant le MSE pour noter
        n_jobs=-1,  # TOUTE LA PUISSANCE OUI OUI OUI
    )
    grid_search.fit(X_train, y_train)

    # Capture the best version of the model
    best_gb_model = grid_search.best_estimator_
    models = {  # Je les ai plus ou moin trié dans l'ordre de performance attendu
        "Baseline (Mean)": MeanPredictor(),
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("lr", LinearRegression())]
        ),
        "SVR": Pipeline(
            [("scaler", StandardScaler()), ("svr", SVR(C=1.0, epsilon=0.2))]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=1,
            n_jobs=-1,
        ),
        "Gradient Boosting": best_gb_model,
        "XGBoost": XGBRegressor(
            n_estimators=100,  # Réduit de 500 à 100 car était overfit
            learning_rate=0.05,
            max_depth=5,  # Réduit de 10 à 5 car était overfit
            n_jobs=-1,
            random_state=1,
            tree_method="hist",
        ),
    }

    print(f"{'Model':22} | {'RMSE':8} | {'MAE':8}")
    print("-" * 45)

    for name, model in models.items():
        evaluate_model(name, model, X_train, y_train, X_test, y_test, results)

    # Evaluation de l'importance des chaque feature dans Random Forest
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feature_names = X_train.columns

    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)
    print(feature_importance_df.head(10))

    plot_metrics(results)


coffee_data_file = "Coffe_sales.csv"
run(coffee_data_file)

## TODO: Put more importance on the name of coffee
## TODO: Add more models (try more models)
