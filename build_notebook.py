#!/usr/bin/env python3
"""Generate ``notebooks/PneumoniaMNIST_Analiza_i_Modelowanie.ipynb`` from current pipeline artifacts."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf

from config import FIGURES_DIR, METRICS_SUMMARY_PATH, NOTEBOOK_PATH, ensure_runtime_dirs
from report_metrics import hyperparam_markdown_section, load_summary


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def build_notebook_cells(summary: dict) -> list[nbf.NotebookNode]:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cells: list[nbf.NotebookNode] = []

    cells.append(
        _md(
            f"""
            # PneumoniaMNIST – Klasyfikacja zdjęć rentgenowskich płuc w kierunku zapalenia płuc
            **Przedmiot:** Uczenie Maszynowe w Diagnostyce Medycznej (Politechnika Krakowska)  
            **Zespół:** Kamil Rayski, Miłosz Pabis  
            **Zbiór danych:** [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) / PneumoniaMNIST ($n = 5\\,856$ zdjęć RTG)

            *Notatnik wygenerowany automatycznie (`build_notebook.py`, {generated}) na podstawie bieżących artefaktów w `models/` i `figures/`.*

            ---
            ## 1. Wstęp i pobranie danych (`kagglehub`)
            Pełny potok trenowania i ewaluacji: `uv run python run_pipeline.py` (grupowy podział walidacyjny, potoki sklearn, CNN, bootstrap na teście).
            """
        )
    )

    cells.append(
        _code(
            """
            import os
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import Image as IPyImage, display

            from config import (
                DATA_DIR,
                DATASET_METADATA_PATH,
                FIGURES_DIR,
                KAGGLE_DATASET_ID,
                METRICS_SUMMARY_PATH,
            )

            os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)

            # Opcjonalnie: pobranie zbioru (pomijane, jeśli cache już istnieje)
            try:
                import kagglehub
                path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
                print("Path to dataset files:", path)
            except Exception as exc:
                print("kagglehub download skipped:", exc)
            """
        )
    )

    cells.append(
        _md(
            """
            ## 2. Eksploracyjna Analiza Danych (EDA)
            Wczytujemy metadane $5\\,856$ zdjęć RTG. Walidacja $n=524$ pochodzi ze **stratyfikowanego, grupowego** podziału
            (`StratifiedGroupKFold`) — ten sam pacjent/studium nie występuje w `train` i `val`.
            """
        )
    )

    cells.append(
        _code(
            """
            meta_df = pd.read_csv(DATASET_METADATA_PATH)
            print(f"Łączna liczba zdjęć RTG: {len(meta_df)}")
            print("\\n--- Oryginalny podział (Kaggle) ---")
            display(pd.crosstab(meta_df["orig_split"], meta_df["label_name"], margins=True))
            print("\\n--- Skorygowany podział (PneumoniaMNIST + grupy pacjent/studium) ---")
            display(pd.crosstab(meta_df["split"], meta_df["label_name"], margins=True))
            if "study_group" in meta_df.columns:
                print("\\nLiczba unikalnych grup study_group per split:")
                display(meta_df.groupby("split")["study_group"].nunique().to_frame("Liczba grup"))
            print("\\n--- Podział etiologiczny ---")
            display(meta_df["subtype"].value_counts().to_frame("Liczba zdjęć"))

            with open(METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
                summary = json.load(f)
            display(pd.DataFrame([summary.get("leakage_audit", {})]).T)
            """
        )
    )

    cells.append(
        _code(
            """
            for fig in ("fig1_class_distribution.png", "fig2_sample_xray_grid.png", "fig3_image_stats_eda.png"):
                display(IPyImage(filename=str(FIGURES_DIR / fig)))
            """
        )
    )

    cells.append(
        _md(
            """
            ## 3. Przetwarzanie danych, Inżynieria Cech i Redukcja Wymiarowości (PCA)
            Klasyczne modele: $64\\times 64$ px + 17 cech domenowych + PCA ($k=100$) wewnątrz potoku sklearn.
            Sieć CNN: $128\\times 128$ px, augmentacja geometryczna, ważona `CrossEntropyLoss`.
            """
        )
    )

    cells.append(_code("display(IPyImage(filename=str(FIGURES_DIR / 'fig4_pca_analysis.png')))"))

    cells.append(_md(hyperparam_markdown_section(summary)))

    cells.append(
        _code(
            """
            from report_metrics import (
                build_hyperparams_dataframe,
                build_paired_bootstrap_dataframe,
                build_results_dataframe,
                build_thresholds_dataframe,
            )

            with open(METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
                summary = json.load(f)

            print("=== Hiperparametry ===")
            display(build_hyperparams_dataframe(summary))
            print("=== Progi walidacyjne (raw + Platt) ===")
            display(build_thresholds_dataframe(summary))
            print("=== Metryki (walidacja / test @0.5 / test @próg z walidacji) ===")
            display(build_results_dataframe(summary))
            pairs = build_paired_bootstrap_dataframe(summary)
            if len(pairs):
                print("=== Paired bootstrap ROC-AUC (test) ===")
                display(pairs)
            """
        )
    )

    cells.append(
        _code(
            """
            for fig in (
                "fig7_cnn_training_curves.png",
                "fig5_model_comparison_roc_pr.png",
                "fig6_confusion_matrices.png",
            ):
                display(IPyImage(filename=str(FIGURES_DIR / fig)))
            """
        )
    )

    cells.append(
        _md(
            """
            ## 5. Wyjaśnialność Modeli (XAI) oraz Analiza Błędów
            Mapy istotności / Grad-CAM oraz analiza FP/FN CNN (próg Platt z walidacji).
            """
        )
    )

    cells.append(
        _code(
            """
            with open(METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
                summary = json.load(f)
            display(IPyImage(filename=str(FIGURES_DIR / "fig8_explainability_gradcam_importance.png")))
            display(IPyImage(filename=str(FIGURES_DIR / "fig9_error_analysis.png")))
            if "subtype_test_accuracy_cnn" in summary:
                display(pd.DataFrame(summary["subtype_test_accuracy_cnn"]).T)
            """
        )
    )

    cells.append(
        _md(
            """
            ## 6. Implementacja – Streamlit (`app.py`)
            Skalibrowane P(Platt), osobne progi walidacyjne dla każdego modelu.

            ```bash
            uv run streamlit run app.py
            ```
            """
        )
    )

    cells.append(_code("display(IPyImage(filename=str(FIGURES_DIR / 'fig10_ui_preview.png')))"))

    return cells


def build_notebook(output_path=None) -> Path:
    ensure_runtime_dirs()
    if not METRICS_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {METRICS_SUMMARY_PATH}. Run `uv run python run_pipeline.py` first."
        )
    summary = load_summary()
    out = Path(output_path or NOTEBOOK_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = build_notebook_cells(summary)

    nbf.write(nb, out)
    print(f"[SUCCESS] Wrote notebook: {out} ({len(nb.cells)} cells)")
    return out


def main() -> None:
    build_notebook()


if __name__ == "__main__":
    main()