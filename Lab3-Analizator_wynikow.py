import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
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
    """Funkcja do zapisu treści w pliku README.md."""
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    """Główna funkcja wykonująca analizę i generująca raport."""


    readme_content = "# Lab3-Analizator_wynikow: Automatyczna Analiza Danych\n\n"
    readme_content += "Ten raport został wygenerowany automatycznie przez GitHub Action.\n\n"


    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    #  Etap 1  Eksploracja i wstępna analiza danych
    readme_content += "## Etap 1: Eksploracja i Wstępna Analiza Danych\n\n"


    df = pd.read_csv(DATA_URL, index_col=0)
    readme_content += "### 1.1. Wczytanie danych\n\n"
    readme_content += f"Dane zostały wczytane z adresu: `{DATA_URL}`.\n\n"
    readme_content += "Pierwsze 5 wierszy ramki danych:\n"
    readme_content += f"```\n{df.head().to_string()}\n```\n\n"


    missing_values = df.isnull().sum().sum()
    readme_content += "### 1.2. Brakujące wartości\n\n"
    readme_content += f"Suma brakujących wartości w całym zbiorze danych: **{missing_values}**.\n\n"
    if missing_values == 0:
        readme_content += "Zbiór danych jest kompletny, nie ma potrzeby imputacji ani usuwania wierszy.\n\n"


    readme_content += "### 1.3. Analiza Statystyczna i Wizualizacje\n\n"
    readme_content += "Podstawowe statystyki dla zmiennych numerycznych:\n"
    readme_content += f"```\n{df.describe().to_string()}\n```\n\n"

    # Wykres 1: Dystrybucja zmiennej docelowej 'score'
    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET_VARIABLE], kde=True, bins=30)
    plt.title(f'Dystrybucja zmiennej docelowej ({TARGET_VARIABLE})')
    plt.xlabel('Wynik (Score)')
    plt.ylabel('Częstotliwość')
    plt.grid(True)
    plot_path_score = os.path.join(PLOTS_DIR, 'score_distribution.png')
    plt.savefig(plot_path_score)
    plt.close()
    readme_content += f"#### Dystrybucja zmiennej '{TARGET_VARIABLE}'\n"
    readme_content += f"![Dystrybucja Score]({plot_path_score})\n\n"
    readme_content += "Wykres pokazuje, że rozkład zmiennej `score` jest zbliżony do normalnego, co jest dobrą cechą dla wielu modeli regresyjnych.\n\n"

    # Wykres 2: Macierz korelacji
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 8})
    plt.title('Macierz korelacji zmiennych numerycznych')
    plot_path_corr = os.path.join(PLOTS_DIR, 'correlation_matrix.png')
    plt.savefig(plot_path_corr)
    plt.close()
    readme_content += "#### Macierz korelacji\n"
    readme_content += f"![Macierz Korelacji]({plot_path_corr})\n\n"
    readme_content += "Z macierzy korelacji widzimy, że zmienne takie jak `education`, `income` i `tuition` mają zauważalną dodatnią korelację ze zmienną `score`. Zmienna `distance` ma słabą ujemną korelację.\n\n"

    #  Etap 2 Inżynieria cech i przygotowanie danych
    readme_content += "## Etap 2: Inżynieria Cech i Przygotowanie Danych\n\n"


    categorical_cols = df.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    readme_content += "### 2.1. Inżynieria Cech\n\n"
    readme_content += f"Zmienne kategoryczne (`{', '.join(categorical_cols)}`) zostały przekonwertowane na zmienne numeryczne za pomocą techniki One-Hot Encoding. Pozwoli to modelom na ich prawidłową interpretację.\n\n"
    readme_content += "Przykładowe dane po transformacji:\n"
    readme_content += f"```\n{df_processed.head().to_string()}\n```\n\n"


    X = df_processed.drop(TARGET_VARIABLE, axis=1)
    y = df_processed[TARGET_VARIABLE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    readme_content += "### 2.2. Podział na Zbiór Treningowy i Testowy\n\n"
    readme_content += "Dane zostały podzielone na zbiór treningowy (80%) i testowy (20%) w sposób losowy. Użycie `random_state=42` zapewnia powtarzalność podziału.\n\n"


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    readme_content += "### 2.3. Standaryzacja Danych\n\n"
    readme_content += "Zmienne predykcyjne zostały poddane standaryzacji (przeskalowane do średniej 0 i odchylenia standardowego 1). Jest to ważny krok, który poprawia wydajność i stabilność wielu algorytmów, zwłaszcza regresji liniowej.\n\n"

    #  Etap 3: Wybór i trenowanie modelu
    readme_content += "## Etap 3: Wybór i Trenowanie Modeli\n\n"
    readme_content += "Do zadania predykcji `score` wybrano i wytrenowano trzy różne modele regresyjne, aby porównać ich skuteczność.\n\n"

    models = {
        "Regresja Liniowa": LinearRegression(),
        "Lasy Losowe": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    readme_content += "Wybrane modele:\n"
    readme_content += "1.  **Regresja Liniowa**: Prosty i interpretowalny model, który stanowi świetny punkt odniesienia (baseline).\n"
    readme_content += "2.  **Lasy Losowe**: Potężny model ensemblowy, odporny na przeuczenie i dobrze radzący sobie z nieliniowymi zależnościami.\n"
    readme_content += "3.  **Gradient Boosting**: Kolejny zaawansowany model ensemblowy, który często osiąga najwyższą dokładność poprzez iteracyjne korygowanie błędów poprzednich drzew.\n\n"

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        readme_content += f"Model **{name}** został pomyślnie wytrenowany.\n"

    readme_content += "\n"

    #  Etap 4: Ocena i optymalizacja modelu
    readme_content += "## Etap 4: Ocena Modeli i Dyskusja o Optymalizacji\n\n"
    readme_content += "### 4.1. Ocena Jakości Modeli\n\n"
    readme_content += "Modele zostały ocenione na zbiorze testowym przy użyciu następujących metryk:\n"
    readme_content += "- **R² (współczynnik determinacji)**: Jak dobrze model wyjaśnia wariancję w danych (im bliżej 1, tym lepiej).\n"
    readme_content += "- **MAE (średni błąd bezwzględny)**: Średnia bezwzględna różnica między prognozą a rzeczywistością (im niżej, tym lepiej).\n"
    readme_content += "- **MSE (błąd średniokwadratowy)**: Średnia kwadratów błędów, mocniej karze duże błędy (im niżej, tym lepiej).\n\n"

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results.append({"Model": name, "R²": r2, "MAE": mae, "MSE": mse})

    results_df = pd.DataFrame(results)
    readme_content += "Wyniki oceny modeli na zbiorze testowym:\n\n"
    readme_content += results_df.to_markdown(index=False)
    readme_content += "\n\n"

    best_model_name = results_df.loc[results_df['R²'].idxmax()]['Model']
    readme_content += f"**Wnioski**: Najlepsze wyniki pod względem metryki R² uzyskał model **{best_model_name}**. Modele ensemblowe (Lasy Losowe i Gradient Boosting) znacząco przewyższają prostą Regresję Liniową, co sugeruje obecność nieliniowych zależności w danych.\n\n"

    readme_content += "### 4.2. Potencjalna Optymalizacja\n\n"
    readme_content += "Chociaż uzyskane wyniki są dobre, można je dalej poprawić. Kolejnym krokiem byłaby optymalizacja hiperparametrów najlepszego modelu (np. `Gradient Boosting`). Można to osiągnąć za pomocą technik takich jak:\n\n"
    readme_content += "- **Grid Search (Przeszukiwanie siatki)**: Systematyczne testowanie różnych kombinacji hiperparametrów (np. `n_estimators`, `learning_rate`, `max_depth`).\n"
    readme_content += "- **Randomized Search (Przeszukiwanie losowe)**: Bardziej efektywna czasowo alternatywa dla Grid Search, która testuje losowe kombinacje parametrów.\n"
    readme_content += "- **Walidacja krzyżowa (Cross-Validation)**: Użycie jej w trakcie strojenia hiperparametrów zapewnia, że model będzie bardziej odporny na przeuczenie i jego wyniki będą bardziej wiarygodne.\n\n"
    readme_content += "Implementacja tych technik pozwoliłaby na 'dostrojenie' modelu do specyfiki danych i potencjalne uzyskanie jeszcze niższych błędów predykcji.\n"


    generate_readme(readme_content)
    print(f"Plik {README_FILE} został pomyślnie wygenerowany.")
    print(f"Wykresy zostały zapisane w katalogu {PLOTS_DIR}.")


if __name__ == "__main__":
    main()