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
├── update_eval_and_ui_fig.py                       # Regeneracja wykresów 5/6/9/10 (progi z walidacji, bez strojenia na teście)
├── report_metrics.py                               # Wspólne tabele/metryki z metrics_summary.json
├── build_report.py                                 # Raport .docx + aktualizacja tabeli w README
├── build_notebook.py                               # Generator notatnika .ipynb z bieżących wyników
├── generate_report_docx.py                         # Silnik raportu DOCX (wywoływany przez build_report.py)
├── app.py                                          # Interaktywna aplikacja diagnostyczna Streamlit (GUI)
├── data/                                           # Pobrany zbiór Chest X-Ray Pneumonia (kagglehub cache)
├── models/
│   ├── classical_models.joblib                     # Wytrenowane potoki: StandardScaler, PCA, LR, RF, SVM
│   ├── cnn_pneumonia.pt                            # Wagi wytrenowanej sieci PneumoniaCNN (PyTorch)
│   ├── dataset_metadata.csv                        # Metadane i statystyki wszystkich 5 856 zdjęć RTG
│   ├── metrics_summary.json                        # Metryki, progi walidacyjne, bootstrap i porównania par
│   └── test_predictions.csv                        # Per-obraz predykcje (surowe + Platt) na zbiorze testowym
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

### Krok 4: Wygenerowanie raportu `.docx`, notatnika `.ipynb` i tabeli w README
```bash
uv run python build_report.py
```
*(Tworzy m.in. `Raport_Koncowy_PneumoniaMNIST.docx` i `Raport Końcowy - Kamil Rayski, Miłosz Pabis.docx` w **katalogu głównym** projektu oraz kopie w `report/`.)*

### Krok 5: Uruchomienie interaktywnej aplikacji diagnostycznej (Streamlit GUI)
```bash
uv run streamlit run app.py
```
Po uruchomieniu aplikacja jest dostępna w przeglądarce pod adresem **http://localhost:8501**.

---

## 4. Podsumowanie wyników modeli

<!-- METRICS_TABLE_START -->
| Model | Zbiór / Tryb | Accuracy | Balanced Acc | Precision | Recall (Czułość) | Specificity (Swoistość) | F1-score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (PCA)** | Walidacyjny (n=524) | 95,23% | 94,61% | 97,64% | 95,89% | 93,33% | 96,76% | 0,9910 | 0,9968 |
| **Logistic Regression (PCA)** | Testowy (próg 0,50) | 74,84% | 67,05% | 71,86% | 98,21% | 35,90% | 82,99% | 0,8996 | 0,9349 |
| **Logistic Regression (PCA)** | Testowy (próg z walidacji, Youden) | 73,08% | 64,53% | 70,26% | 98,72% | 30,34% | 82,09% | 0,8996 | 0,9349 |
| **Random Forest** | Walidacyjny (n=524) | 95,04% | 91,34% | 94,59% | 98,97% | 83,70% | 96,73% | 0,9889 | 0,9959 |
| **Random Forest** | Testowy (próg 0,50) | 77,24% | 69,83% | 73,48% | 99,49% | 40,17% | 84,53% | 0,9355 | 0,9576 |
| **Random Forest** | Testowy (próg z walidacji, Youden) | 84,13% | 79,70% | 81,02% | 97,44% | 61,97% | 88,47% | 0,9355 | 0,9576 |
| **SVM (RBF + PCA)** | Walidacyjny (n=524) | 98,09% | 96,78% | 97,97% | 99,49% | 94,07% | 98,72% | 0,9961 | 0,9986 |
| **SVM (RBF + PCA)** | Testowy (próg 0,50) | 76,60% | 69,15% | 73,11% | 98,97% | 39,32% | 84,10% | 0,9050 | 0,9334 |
| **SVM (RBF + PCA)** | Testowy (próg z walidacji, Youden) | 78,37% | 71,50% | 74,66% | 98,97% | 44,02% | 85,12% | 0,9050 | 0,9334 |
| **PneumoniaCNN (PyTorch)** | Walidacyjny (n=524) | 98,09% | 97,75% | 98,97% | 98,46% | 97,04% | 98,71% | 0,9977 | 0,9992 |
| **PneumoniaCNN (PyTorch)** | Testowy (próg 0,50) | 85,58% | 81,20% | 81,91% | 98,72% | 63,68% | 89,53% | 0,9494 | 0,9610 |
| **PneumoniaCNN (PyTorch)** | Testowy (próg z walidacji, Youden) | 85,58% | 81,28% | 82,05% | 98,46% | 64,10% | 89,51% | 0,9494 | 0,9610 |
<!-- METRICS_TABLE_END -->

