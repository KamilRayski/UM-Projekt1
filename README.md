# Uczenie Maszynowe w Diagnostyce Medycznej – Projekt 1
## PneumoniaMNIST – klasyfikacja zdjęć rentgenowskich płuc w kierunku zapalenia płuc

**Autorzy:** Kamil Rayski, Miłosz Pabis  
**Przedmiot:** Uczenie Maszynowe w Diagnostyce Medycznej (Studia stacjonarne II stopnia, Wydział Informatyki i Telekomunikacji, Politechnika Krakowska)  
**Zbiór danych:** [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) / PneumoniaMNIST ($n = 5\,856$ zdjęć RTG)

---

## 1. Opis projektu i zakres realizacji

Celem projektu jest opracowanie, optymalizacja oraz wyjaśnialna ewaluacja (XAI) systemu wspomagania decyzji klinicznych (**CDSS – Clinical Decision Support System**) do automatycznego wykrywania zapalenia płuc (*Pneumonia*) na pediatrycznych zdjęciach rentgenowskich (RTG) klatki piersiowej w projekcji przednio-tylnej (AP).

| Sekcja raportu | Zakres wymagań | Sposób realizacji w projekcie |
| :--- | :--- | :--- |
| **1. Wstęp i dane** | Charakterystyka zbioru, balans klas, problemy z jakością danych | Pobranie zbioru $5\,856$ zdjęć RTG przez `kagglehub` (`paultimothymooney/chest-xray-pneumonia`). Identyfikacja niezbalansowania klas ($73{,}0\%$ `PNEUMONIA` vs $27{,}0\%$ `NORMAL`), podtypów etiologicznych (`BACTERIA`: $2\,780$, `VIRUS`: $1\,493$) oraz naprawa krytycznie małego zbioru walidacyjnego ($n=16 \rightarrow n=524$) zgodnie ze standardem **PneumoniaMNIST** ($4\,708$ `train` / $524$ `val` / $624$ `test`). |
| **2. Przygotowanie danych (EDA & Preprocessing)** | Wizualizacje rozkładów, normalizacja, inżynieria cech, redukcja wymiarowości | Analiza rozdzielczości, jasności i kontrastu pikseli oraz wyznaczenie uśrednionych szablonów płuc i mapy różnicowej. Ekstrakcja **17 cech domenowych** (asymetrii pól płucnych L-P, stref górnej/środkowej/dolnej, kwantyli luminancji i gradientów krawędziowych). Standaryzacja `StandardScaler` oraz redukcja wymiarowości **PCA ($k=100$ komponentów – *Eigenlungs*)** wyjaśniająca **91,47%** wariancji. |
| **3. Modelowanie i ewaluacja** | Porównanie modeli ML i DL, walidacja krzyżowa, tuning hiperparametrów, metryki | Strojnie `GridSearchCV` (5-fold Stratified CV) dla 3 modeli klasycznych (**Logistic Regression**, **Random Forest**, **SVM z jądrem RBF**) oraz trenowanie 4-blokowej głębokiej sieci konwolucyjnej **PneumoniaCNN (PyTorch)** z `BatchNorm2d`, `GlobalAveragePooling`, augmentacją geometryczną i ważoną funkcją straty `CrossEntropyLoss` na akceleratorze Apple Silicon GPU (`MPS`). |
| **4. Wyjaśnialność (XAI) i analiza błędów** | Feature Importance, mapy aktywacji, analiza przypadków FP i FN | Dwuwymiarowe mapy istotności pikseli (Random Forest) oraz przestrzenna projekcja wag PCA (Regresja Logistyczna) na sylwetkę klatki piersiowej. Generowanie map cieplnych **Grad-CAM** dla sieci `PneumoniaCNN` lokalizujących nacieki zapalne. Analiza błędów *False Positive* i *False Negative* z rozbiciem na etiologię bakteryjną i wirusową. |
| **5. Implementacja (GUI)** | Interfejs użytkownika do korzystania z modelu | Interaktywna aplikacja webowa w **Streamlit** (`app.py`) z wyborem zdjęć testowych lub wgrywaniem własnych plików RTG, regulacją progu decyzyjnego, wizualizacją **Grad-CAM** na żywo oraz panelem porównawczym wszystkich 4 modeli. |

---

## 2. Struktura projektu

```text
PROJEKT1/
├── README.md                                       # Dokumentacja projektu i instrukcja uruchomienia
├── pyproject.toml                                  # Projekt i zależności zarządzane przez uv
├── uv.lock                                         # Powtarzalne, zablokowane wersje zależności
├── config.py                                       # Wspólna konfiguracja przenośnych ścieżek
├── Raport Końcowy Przykład.docx                    # Oryginalny szablon sprawozdania
├── Raport Końcowy - Kamil Rayski, Miłosz Pabis.docx # Wypełniony raport końcowy (.docx) gotowy do oddania
├── Raport_Koncowy_PneumoniaMNIST.docx              # Kopia raportu końcowego (.docx)
├── PneumoniaMNIST_Analiza_i_Modelowanie.ipynb      # Notatnik Jupyter z kodem, tabelami i wykresami
├── download_data.py                                # Skrypt pobierający zbiór danych przez kagglehub
├── run_pipeline.py                                 # Główny potok: EDA, PCA, GridSearchCV, trening CNN i wykresy
├── update_eval_and_ui_fig.py                       # Kalibracja progów decyzyjnych i generowanie podglądu GUI
├── generate_report_docx.py                         # Skrypt budujący raport końcowy .docx na bazie szablonu
├── build_notebook.py                               # Generator notatnika Jupyter (.ipynb) z wynikami
├── app.py                                          # Interaktywna aplikacja diagnostyczna Streamlit (GUI)
├── data/                                           # Pobrany zbiór Chest X-Ray Pneumonia (kagglehub cache)
├── models/
│   ├── classical_models.joblib                     # Wytrenowane potoki: StandardScaler, PCA, LR, RF, SVM
│   ├── cnn_pneumonia.pt                            # Wagi wytrenowanej sieci PneumoniaCNN (PyTorch)
│   ├── dataset_metadata.csv                        # Metadane i statystyki wszystkich 5 856 zdjęć RTG
│   └── metrics_summary.json                        # Zestawienie metryk, najlepszych hiperparametrów i progów
└── figures/
    ├── fig1_class_distribution.png                 # Rozkład klas (oryginalny vs PneumoniaMNIST vs podtypy)
    ├── fig2_sample_xray_grid.png                   # Przykładowe zdjęcia RTG (NORMAL, BACTERIA, VIRUS)
    ├── fig3_image_stats_eda.png                    # Statystyki wymiarów, jasności, kontrastu i średnie płuca
    ├── fig4_pca_analysis.png                       # Wariancja wyjaśniona PCA, rzutowanie 2D i Eigenlungs
    ├── fig5_model_comparison_roc_pr.png            # Krzywe ROC, Precision-Recall oraz porównanie metryk
    ├── fig6_confusion_matrices.png                 # Macierze pomyłek na zbiorze testowym dla 4 modeli
    ├── fig7_cnn_training_curves.png                # Przebieg funkcji straty i dokładności sieci PneumoniaCNN
    ├── fig8_explainability_gradcam_importance.png  # Wyjaśnialność XAI: mapy istotności 2D oraz Grad-CAM
    ├── fig9_error_analysis.png                     # Analiza błędów FP/FN oraz skuteczność wg etiologii
    └── fig10_ui_preview.png                        # Podgląd działania aplikacji diagnostycznej Streamlit
```

---

## 3. Instrukcja uruchomienia krok po kroku

### Krok 1: Instalacja `uv` i synchronizacja zależności
Z poziomu katalogu projektu uruchom:
```bash
uv sync
```
Środowisko `.venv` zostanie utworzone automatycznie na podstawie `pyproject.toml` i `uv.lock`.

### Krok 2: Pobranie zbioru danych przez `kagglehub`
```bash
uv run python download_data.py
```
*(Zbiór jest automatycznie zapisywany w katalogu lokalnym `UM/PROJEKT1/data/`, dzięki czemu kolejne uruchomienia nie wymagają ponownego pobierania).*

### Krok 3: Uruchomienie pełnego potoku eksperymentalnego (EDA + PCA + Modele ML + Sieć CNN + Wykresy)
```bash
uv run python run_pipeline.py
uv run python update_eval_and_ui_fig.py
```

### Krok 4: Wygenerowanie wypełnionego raportu końcowego `.docx` oraz notatnika `.ipynb`
```bash
uv run python generate_report_docx.py
uv run python build_notebook.py
```

### Krok 5: Uruchomienie interaktywnej aplikacji diagnostycznej (Streamlit GUI)
```bash
uv run streamlit run app.py
```
Po uruchomieniu aplikacja jest dostępna w przeglądarce pod adresem **http://localhost:8501**.

---

## 4. Podsumowanie wyników modeli

| Model | Zbiór / Tryb | Accuracy | Balanced Acc | Precision | Recall (Czułość) | Specificity (Swoistość) | F1-score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (PCA)** | Walidacyjny ($n=524$) | 95,42% | 94,74% | 97,65% | 96,14% | 93,33% | 96,89% | 0,9892 | 0,9959 |
| **Random Forest** | Walidacyjny ($n=524$) | 96,18% | 93,80% | 96,24% | 98,71% | 88,89% | 97,46% | 0,9901 | 0,9961 |
| **SVM (RBF + PCA)** | Walidacyjny ($n=524$) | 97,33% | 96,51% | 98,20% | 98,20% | 94,81% | 98,20% | 0,9955 | 0,9984 |
| **PneumoniaCNN (PyTorch)** | Walidacyjny ($n=524$) | **98,09%** | **97,99%** | **99,22%** | **98,20%** | **97,78%** | **98,71%** | **0,9980** | **0,9993** |
| **Logistic Regression (PCA)** | Testowy ($n=624$, kalibrowany) | 81,41% | 80,60% | 86,05% | 83,85% | 77,35% | 84,94% | 0,8869 | 0,9261 |
| **SVM (RBF + PCA)** | Testowy ($n=624$, kalibrowany) | 82,69% | 82,74% | 88,95% | 82,56% | 82,91% | 85,64% | 0,9039 | 0,9327 |
| **Random Forest** | Testowy ($n=624$, kalibrowany) | **88,14%** | 86,75% | 89,11% | 92,31% | 81,20% | **90,68%** | 0,9349 | 0,9589 |
| **PneumoniaCNN (PyTorch)** | Testowy ($n=624$, próg `0.50` przesiewowy) | 83,01% | 77,69% | 79,10% | **98,97%** | 56,41% | 87,93% | **0,9421** | **0,9596** |
| **PneumoniaCNN (PyTorch)** | Testowy ($n=624$, kalibrowany) | **88,14%** | **88,12%** | **92,47%** | 88,21% | **88,03%** | 90,29% | **0,9421** | **0,9596** |

