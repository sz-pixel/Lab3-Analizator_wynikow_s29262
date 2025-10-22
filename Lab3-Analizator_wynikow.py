import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv"
TARGET_VARIABLE = "score"
README_FILE = "README.md"
PLOTS_DIR = "plots"

def generate_readme(content):
    """Zapisuje treść do pliku README.md."""
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    """Główna funkcja wykonująca analizę i generująca raport."""

    # Przygotowanie katalogu na wykresy
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    readme_content = "# Lab3-Analizator_wynikow: Automatyczna Analiza Danych\n\n"
    readme_content += "Ten raport został wygenerowany automatycznie.\n\n"

    # Etap 1: Eksploracja danych
    df = pd.read_csv(DATA_URL, index_col=0)
    readme_content += f"## Etap 1: Eksploracja danych\n\n"
    readme_content += f"Dane wczytane z `{DATA_URL}`:\n```\n{df.head().to_string()}\n```\n\n"

    missing_values = df.isnull().sum().sum()
    readme_content += f"Suma brakujących wartości: **{missing_values}**\n\n"

    readme_content += f"Podstawowe statystyki:\n```\n{df.describe().to_string()}\n```\n\n"

    # Wykres: dystrybucja zmiennej docelowej
    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET_VARIABLE], kde=True, bins=30)
    plt.title(f'Dystrybucja zmiennej docelowej ({TARGET_VARIABLE})')
    plt.xlabel('Score')
    plt.ylabel('Częstotliwość')
    plt.grid(True)
    plot_score_path = os.path.join(PLOTS_DIR, 'score_distribution.png')
    plt.savefig(plot_score_path)
    plt.close()
    readme_content += f"![Dystrybucja Score]({plot_score_path})\n\n"

    # Wykres: macierz korelacji
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 8})
    plt.title('Macierz korelacji zmiennych numerycznych')
    plot_corr_path = os.path.join(PLOTS_DIR, 'correlation_matrix.png')
    plt.savefig(plot_corr_path)
    plt.close()
    readme_content += f"![Macierz Korelacji]({plot_corr_path})\n\n"

    # Etap 2: Przygotowanie danych
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_processed.drop(TARGET_VARIABLE, axis=1)
    y = df_processed[TARGET_VARIABLE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Etap 3: Modele
    models = {
        "Regresja Liniowa": LinearRegression(),
        "Lasy Losowe": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    for model in models.values():
        model.fit(X_train_scaled, y_train)

    # Etap 4: Ocena modeli
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        results.append({
            "Model": name,
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    readme_content += "## Wyniki oceny modeli\n\n"
    readme_content += results_df.to_markdown(index=False) + "\n\n"

    best_model = results_df.loc[results_df['R²'].idxmax()]['Model']
    readme_content += f"Najlepszy model pod względem R²: **{best_model}**.\n"

    generate_readme(readme_content)
    print(f"README.md wygenerowany. Wykresy zapisane w {PLOTS_DIR}.")

if __name__ == "__main__":
    main()
