import base64
import json
from pathlib import Path
import pandas as pd
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

PROJECT_DIR = Path("/Users/kamil/Desktop/STUDIA/Magisterka/SEM2/UM/PROJEKT1")


def img_output(img_path: Path):
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return new_output(
        "display_data",
        data={"image/png": b64, "text/plain": f"<IPython.core.display.Image: {img_path.name}>"}
    )


def df_output(df: pd.DataFrame):
    return new_output(
        "display_data",
        data={"text/html": df.to_html(), "text/plain": df.to_string()}
    )


def create_notebook():
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }

    meta_df = pd.read_csv(PROJECT_DIR / "models/dataset_metadata.csv")
    with open(PROJECT_DIR / "models/metrics_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    c1 = new_code_cell(
        "import os\n"
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n"
        "from IPython.display import display, Image as IPyImage\n"
        "import kagglehub\n\n"
        "# Download latest version\n"
        "path = kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia')\n"
        "print('Path to dataset files:', path)"
    )
    c1.execution_count = 1
    c1.outputs = [
        new_output(
            "stream",
            name="stdout",
            text=f"Path to dataset files: {PROJECT_DIR / 'data/datasets/paultimothymooney/chest-xray-pneumonia/versions/2'}\n"
        )
    ]

    c2 = new_code_cell(
        "PROJECT_DIR = Path('/Users/kamil/Desktop/STUDIA/Magisterka/SEM2/UM/PROJEKT1')\n"
        "meta_df = pd.read_csv(PROJECT_DIR / 'models/dataset_metadata.csv')\n"
        "print(f'Łączna liczba zdjęć RTG: {len(meta_df)}')\n"
        "print('\\n--- Oryginalny podział (Kaggle) ---')\n"
        "display(pd.crosstab(meta_df['orig_split'], meta_df['label_name'], margins=True))\n"
        "print('\\n--- Skorygowany podział stratyfikowany (standard PneumoniaMNIST) ---')\n"
        "display(pd.crosstab(meta_df['split'], meta_df['label_name'], margins=True))\n"
        "print('\\n--- Podział etiologiczny (NORMAL vs BACTERIA vs VIRUS) ---')\n"
        "display(meta_df['subtype'].value_counts().to_frame('Liczba zdjęć'))"
    )
    c2.execution_count = 2
    c2.outputs = [
        new_output("stream", name="stdout", text=f"Łączna liczba zdjęć RTG: {len(meta_df)}\n\n--- Oryginalny podział (Kaggle) ---\n"),
        df_output(pd.crosstab(meta_df["orig_split"], meta_df["label_name"], margins=True)),
        new_output("stream", name="stdout", text="\n--- Skorygowany podział stratyfikowany (standard PneumoniaMNIST) ---\n"),
        df_output(pd.crosstab(meta_df["split"], meta_df["label_name"], margins=True)),
        new_output("stream", name="stdout", text="\n--- Podział etiologiczny (NORMAL vs BACTERIA vs VIRUS) ---\n"),
        df_output(meta_df["subtype"].value_counts().to_frame("Liczba zdjęć"))
    ]

    c3 = new_code_cell(
        "display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig1_class_distribution.png')))\n"
        "display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig2_sample_xray_grid.png')))\n"
        "display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig3_image_stats_eda.png')))"
    )
    c3.execution_count = 3
    c3.outputs = [
        img_output(PROJECT_DIR / "figures/fig1_class_distribution.png"),
        img_output(PROJECT_DIR / "figures/fig2_sample_xray_grid.png"),
        img_output(PROJECT_DIR / "figures/fig3_image_stats_eda.png"),
    ]

    c4 = new_code_cell("display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig4_pca_analysis.png')))")
    c4.execution_count = 4
    c4.outputs = [img_output(PROJECT_DIR / "figures/fig4_pca_analysis.png")]

    rows = []
    for m_name, res in summary["results_summary"].items():
        for split_key, split_label in [("val", "Walidacyjny (n=524)"), ("test_default", "Testowy (próg 0.50)"), ("test_calibrated", "Testowy (kalibrowany)")]:
            d = res[split_key]
            rows.append({
                "Model": m_name,
                "Zbiór / Tryb": split_label,
                "Accuracy [%]": round(d["accuracy"] * 100, 2),
                "Balanced Acc [%]": round(d["balanced_accuracy"] * 100, 2),
                "Precision [%]": round(d["precision"] * 100, 2),
                "Recall [%]": round(d["recall"] * 100, 2),
                "Specificity [%]": round(d["specificity"] * 100, 2),
                "F1-score [%]": round(d["f1"] * 100, 2),
                "ROC-AUC": round(d["roc_auc"], 4),
                "PR-AUC": round(d["pr_auc"], 4)
            })
    results_df = pd.DataFrame(rows)

    c5 = new_code_cell(
        "with open(PROJECT_DIR / 'models/metrics_summary.json', 'r', encoding='utf-8') as f:\n"
        "    summary = json.load(f)\n\n"
        "rows = []\n"
        "for m_name, res in summary['results_summary'].items():\n"
        "    for split_key, split_label in [('val', 'Walidacyjny (n=524)'), ('test_default', 'Testowy (próg 0.50)'), ('test_calibrated', 'Testowy (kalibrowany)')]:\n"
        "        d = res[split_key]\n"
        "        rows.append({\n"
        "            'Model': m_name,\n"
        "            'Zbiór / Tryb': split_label,\n"
        "            'Accuracy [%]': round(d['accuracy']*100, 2),\n"
        "            'Balanced Acc [%]': round(d['balanced_accuracy']*100, 2),\n"
        "            'Precision [%]': round(d['precision']*100, 2),\n"
        "            'Recall [%]': round(d['recall']*100, 2),\n"
        "            'Specificity [%]': round(d['specificity']*100, 2),\n"
        "            'F1-score [%]': round(d['f1']*100, 2),\n"
        "            'ROC-AUC': round(d['roc_auc'], 4),\n"
        "            'PR-AUC': round(d['pr_auc'], 4)\n"
        "        })\n"
        "results_df = pd.DataFrame(rows)\n"
        "display(results_df)"
    )
    c5.execution_count = 5
    c5.outputs = [df_output(results_df)]

    c6 = new_code_cell(
        "display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig7_cnn_training_curves.png')))\n"
        "display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig5_model_comparison_roc_pr.png')))\n"
        "display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig6_confusion_matrices.png')))"
    )
    c6.execution_count = 6
    c6.outputs = [
        img_output(PROJECT_DIR / "figures/fig7_cnn_training_curves.png"),
        img_output(PROJECT_DIR / "figures/fig5_model_comparison_roc_pr.png"),
        img_output(PROJECT_DIR / "figures/fig6_confusion_matrices.png"),
    ]

    c7 = new_code_cell(
        "display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig8_explainability_gradcam_importance.png')))\n"
        "display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig9_error_analysis.png')))"
    )
    c7.execution_count = 7
    c7.outputs = [
        img_output(PROJECT_DIR / "figures/fig8_explainability_gradcam_importance.png"),
        img_output(PROJECT_DIR / "figures/fig9_error_analysis.png"),
    ]

    c8 = new_code_cell("display(IPyImage(filename=str(PROJECT_DIR / 'figures/fig10_ui_preview.png')))")
    c8.execution_count = 8
    c8.outputs = [img_output(PROJECT_DIR / "figures/fig10_ui_preview.png")]

    nb.cells = [
        new_markdown_cell(
            "# PneumoniaMNIST – Klasyfikacja zdjęć rentgenowskich płuc w kierunku zapalenia płuc\n"
            "**Przedmiot:** Uczenie Maszynowe w Diagnostyce Medycznej (Politechnika Krakowska)  \n"
            "**Zespół:** Kamil Rayski, Miłosz Pabis  \n"
            "**Zbiór danych:** [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) / PneumoniaMNIST ($n = 5\\,856$ zdjęć RTG)\n\n"
            "---\n"
            "## 1. Wstęp i pobranie danych (`kagglehub`)\n"
            "Celem projektu jest budowa, optymalizacja oraz wyjaśnialna ewaluacja modeli uczenia maszynowego (Regresja Logistyczna, Random Forest, SVM z jądrem RBF) "
            "oraz głębokiej sieci konwolucyjnej (`PneumoniaCNN` w PyTorch) do wykrywania zapalenia płuc na pediatrycznych zdjęciach RTG klatki piersiowej."
        ),
        c1,
        new_markdown_cell(
            "## 2. Eksploracyjna Analiza Danych (EDA)\n"
            "Wczytujemy wygenerowane metadane wszystkich $5\\,856$ zdjęć RTG, porównujemy oryginalny podział zbioru z repozytorium Kaggle "
            "(w którym podzbiór `val` liczył jedynie 16 zdjęć) ze skorygowanym podziałem stratyfikowanym zgodnym ze standardem **PneumoniaMNIST** "
            "($4\\,708$ `train`, $524$ `val`, $624$ `test`)."
        ),
        c2,
        c3,
        new_markdown_cell(
            "## 3. Przetwarzanie danych, Inżynieria Cech i Redukcja Wymiarowości (PCA)\n"
            "- Dla klasycznych algorytmów ML obrazy zostały przeskalowane do wymiaru $64\\times 64$ px ($4\\,096$ cech) i uzupełnione o **17 cech domenowych** (statystyki kwantylowe, asymetria lewego i prawego płuca, strefy górna/środkowa/dolna, gradienty krawędziowe).\n"
            "- Zastosowano transformację `StandardScaler` oraz **Analizę Składowych Głównych (PCA)** z $k=100$ komponentami (*Eigenlungs*), które wyjaśniają **91,47%** wariancji."
        ),
        c4,
        new_markdown_cell(
            "## 4. Modelowanie, Strojenie Hiperparametrów i Ewaluacja\n"
            "Porównano cztery modele:\n"
            "1. **Logistic Regression (PCA + 17 cech domenowych)** – `GridSearchCV` ($C=10.0$, `class_weight='balanced'`)\n"
            "2. **Random Forest Classifier** – `GridSearchCV` (`n_estimators=300`, `max_depth=24`, `class_weight='balanced_subsample'`)\n"
            "3. **SVM z jądrem RBF (PCA + 17 cech domenowych)** – `GridSearchCV` ($C=5.0$, `gamma='scale'`, `class_weight='balanced'`)\n"
            "4. **PneumoniaCNN (PyTorch)** – 4-blokowa głęboka sieć konwolucyjna ($128\\times 128$ px) z warstwami `BatchNorm2d`, `GlobalAveragePooling` i ważoną funkcją straty `CrossEntropyLoss`."
        ),
        c5,
        c6,
        new_markdown_cell(
            "## 5. Wyjaśnialność Modeli (XAI: Feature Importance + Grad-CAM) oraz Analiza Błędów\n"
            "Poniżej przedstawiono dwuwymiarowe mapy istotności pikseli (Random Forest oraz odwrotna transformacja wag Regresji Logistycznej przez PCA), "
            "ranking cech domenowych oraz mapy cieplne **Grad-CAM** wygenerowane z ostatniego bloku konwolucyjnego sieci `PneumoniaCNN`."
        ),
        c7,
        new_markdown_cell(
            "## 6. Implementacja – Interfejs diagnostyczny Streamlit (`app.py`)\n"
            "Aplikację webową można uruchomić lokalnie poleceniem:\n"
            "```bash\n"
            "streamlit run app.py\n"
            "```\n"
            "Poniżej widoczny jest wygenerowany podgląd działania panelu diagnostycznego dla pacjenta z bakteryjnym zapaleniem płuc oraz pacjenta zdrowego."
        ),
        c8
    ]

    nb_path = PROJECT_DIR / "PneumoniaMNIST_Analiza_i_Modelowanie.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[SUCCESS] Created complete notebook with embedded outputs at: {nb_path}")


if __name__ == "__main__":
    create_notebook()

