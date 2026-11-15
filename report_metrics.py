"""Shared helpers for notebooks, DOCX, and README from ``metrics_summary.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import METRICS_SUMMARY_PATH

MODEL_LABELS = {
    "Logistic Regression": "Logistic Regression (PCA)",
    "Random Forest": "Random Forest",
    "SVM (RBF)": "SVM (RBF + PCA)",
    "PneumoniaCNN (PyTorch)": "PneumoniaCNN (PyTorch)",
}

METRIC_SPLITS = [
    ("val", "Walidacyjny (n=524)"),
    ("test_default", "Testowy (próg 0.50)"),
    ("test_val_threshold", "Testowy (próg z walidacji, Youden)"),
]


def load_summary(path: Path | None = None) -> dict[str, Any]:
    summary_path = path or METRICS_SUMMARY_PATH
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pct(x: float) -> float:
    return round(x * 100, 2)


def build_results_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_key, res in summary["results_summary"].items():
        for split_key, split_label in METRIC_SPLITS:
            if split_key not in res:
                continue
            d = res[split_key]
            boot = res.get("bootstrap_roc_auc", {})
            rows.append(
                {
                    "Model": MODEL_LABELS.get(model_key, model_key),
                    "Zbiór / Tryb": split_label,
                    "Accuracy [%]": _pct(d["accuracy"]),
                    "Balanced Acc [%]": _pct(d["balanced_accuracy"]),
                    "Precision [%]": _pct(d["precision"]),
                    "Recall [%]": _pct(d["recall"]),
                    "Specificity [%]": _pct(d["specificity"]),
                    "F1-score [%]": _pct(d["f1"]),
                    "ROC-AUC": round(d["roc_auc"], 4),
                    "PR-AUC": round(d["pr_auc"], 4),
                    "ROC-AUC bootstrap 95% CI": (
                        f"{boot.get('ci_low', 0):.4f}–{boot.get('ci_high', 0):.4f}"
                        if boot.get("ci_low") is not None
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_hyperparams_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_key, info in summary["best_params_summary"].items():
        val_auc = summary["results_summary"][model_key]["val"]["roc_auc"]
        rows.append(
            {
                "Model": MODEL_LABELS.get(model_key, model_key),
                "Najlepsze hiperparametry": json.dumps(info.get("best_params", {}), ensure_ascii=False),
                "5-fold grouped CV ROC-AUC": round(info.get("cv_roc_auc", 0), 4),
                "Val ROC-AUC": round(val_auc, 4),
            }
        )
    return pd.DataFrame(rows)


def build_thresholds_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_key in summary["val_thresholds_raw"]:
        rows.append(
            {
                "Model": MODEL_LABELS.get(model_key, model_key),
                "Próg raw (walidacja)": round(summary["val_thresholds_raw"][model_key], 4),
                "Próg Platt (walidacja)": round(summary["val_thresholds_platt"][model_key], 4),
            }
        )
    return pd.DataFrame(rows)


def build_paired_bootstrap_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    pairs = summary.get("bootstrap_paired_auc_comparisons_test", [])
    rows = []
    for item in pairs:
        rows.append(
            {
                "Model A": item.get("model_a"),
                "Model B": item.get("model_b"),
                "AUC A": round(item.get("auc_a", 0), 4),
                "AUC B": round(item.get("auc_b", 0), 4),
                "Śr. różnica AUC (A−B)": round(item.get("mean_auc_diff", 0), 4),
                "P(A>B)": round(item.get("p_a_greater_b", 0), 3) if item.get("p_a_greater_b") is not None else None,
                "CI różnicy 95%": (
                    f"{item['ci_diff_low']:.4f}–{item['ci_diff_high']:.4f}"
                    if item.get("ci_diff_low") is not None
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _count_cell(counts: dict, split: str, label: str) -> int:
    return int(counts.get(label, {}).get(split, 0))


def _pct_str(part: int, whole: int) -> str:
    if whole == 0:
        return "0,0%"
    return f"{part / whole * 100:.1f}%".replace(".", ",")


def build_dataset_split_rows_docx(summary: dict[str, Any]) -> list[list[str]]:
    orig = summary["dataset_counts"]["orig_splits"]
    corr = summary["dataset_counts"]["corrected_splits"]
    specs = [
        ("Zbiór treningowy (train)", "train"),
        ("Zbiór walidacyjny (val)", "val"),
        ("Zbiór testowy (test)", "test"),
        ("Razem (cały zbiór)", None),
    ]

    def row_for_split(title: str, split: str | None) -> list[str]:
        if split:
            o_n = _count_cell(orig, split, "NORMAL")
            o_p = _count_cell(orig, split, "PNEUMONIA")
            c_n = _count_cell(corr, split, "NORMAL")
            c_p = _count_cell(corr, split, "PNEUMONIA")
            total = c_n + c_p
            all_total = summary["dataset_counts"]["total"]
        else:
            o_n = sum(_count_cell(orig, s, "NORMAL") for s in ("train", "val", "test"))
            o_p = sum(_count_cell(orig, s, "PNEUMONIA") for s in ("train", "val", "test"))
            c_n = sum(_count_cell(corr, s, "NORMAL") for s in ("train", "val", "test"))
            c_p = sum(_count_cell(corr, s, "PNEUMONIA") for s in ("train", "val", "test"))
            total = c_n + c_p
            all_total = total

        o_all = o_n + o_p
        return [
            title,
            f"{o_n} ({_pct_str(o_n, o_all)})",
            f"{o_p} ({_pct_str(o_p, o_all)})",
            f"{c_n} ({_pct_str(c_n, total)})",
            f"{c_p} ({_pct_str(c_p, total)})",
            f"{total} ({_pct_str(total, all_total)})",
        ]

    return [row_for_split(title, split) for title, split in specs]


def build_readme_metrics_markdown(summary: dict[str, Any]) -> str:
    df = build_results_dataframe(summary)
    lines = [
        "| Model | Zbiór / Tryb | Accuracy | Balanced Acc | Precision | Recall (Czułość) | Specificity (Swoistość) | F1-score | ROC-AUC | PR-AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| **{row['Model']}** | {row['Zbiór / Tryb']} | "
            f"{row['Accuracy [%]']:.2f}%".replace(".", ",")
            + " | "
            + f"{row['Balanced Acc [%]']:.2f}%".replace(".", ",")
            + " | "
            + f"{row['Precision [%]']:.2f}%".replace(".", ",")
            + " | "
            + f"{row['Recall [%]']:.2f}%".replace(".", ",")
            + " | "
            + f"{row['Specificity [%]']:.2f}%".replace(".", ",")
            + " | "
            + f"{row['F1-score [%]']:.2f}%".replace(".", ",")
            + " | "
            + f"{row['ROC-AUC']:.4f}".replace(".", ",")
            + " | "
            + f"{row['PR-AUC']:.4f}".replace(".", ",")
            + " |"
        )
    return "\n".join(lines)


def hyperparam_markdown_section(summary: dict[str, Any]) -> str:
    pca_var = summary.get("pca_explained_variance_100", 0) * 100
    lines = [
        "## 4. Modelowanie, Strojenie Hiperparametrów i Ewaluacja",
        "",
        "Porównano cztery modele (preprocessing w potokach `sklearn`, **5-fold StratifiedGroupKFold** "
        "po identyfikatorze studium/pacjenta — bez wycieku między partycjami). "
        "Progi decyzyjne dobierano wyłącznie na walidacji (Youden J); na test nanoszono je bez strojenia. "
        "Prawdopodobieństwa raportowano po skalowaniu **Platta** (fit na walidacji).",
        "",
        f"- PCA ($k=100$) na zbiorze treningowym wyjaśnia **{pca_var:.2f}%** wariancji.",
        "",
        "**Wybrane konfiguracje (z `metrics_summary.json`):**",
        "",
    ]
    for model_key, info in summary["best_params_summary"].items():
        label = MODEL_LABELS.get(model_key, model_key)
        params = info.get("best_params", {})
        param_txt = ", ".join(f"`{k}={v!r}`" for k, v in params.items())
        lines.append(
            f"- **{label}** — CV ROC-AUC = `{info.get('cv_roc_auc', 0):.4f}`, parametry: {param_txt}"
        )
    audit = summary.get("leakage_audit", {})
    if audit:
        lines.extend(
            [
                "",
                f"**Audyt rozłączności grup:** liczba grup train/val/test = "
                f"{audit.get('group_counts', {})}; wszystkie przecięcia = 0 obrazów.",
            ]
        )
    return "\n".join(lines)