"""Regenerate evaluation figures 5, 6, 9, 10 without retuning thresholds on the test set."""

import os
import json

from config import (
    CLASSICAL_MODELS_PATH,
    CNN_MODEL_PATH,
    DATASET_METADATA_PATH,
    FIGURES_DIR,
    MATPLOTLIB_CACHE_DIR,
    METRICS_SUMMARY_PATH,
    ensure_runtime_dirs,
    resolve_project_path,
)

ensure_runtime_dirs()
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE_DIR)

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
from sklearn.metrics import roc_curve, precision_recall_curve

from run_pipeline import (
    PneumoniaCNN,
    ChestXRayCNNDataset,
    PlattCalibrator,
    load_classical_bundle,
    predict_classical_probabilities,
)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 180
sns.set_theme(style="whitegrid", palette="muted")

MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest": "#2ca02c",
    "SVM (RBF)": "#ff7f0e",
    "PneumoniaCNN (PyTorch)": "#d62728",
}


def load_summary():
    with open(METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    summary = load_summary()
    meta = pd.read_csv(DATASET_METADATA_PATH)
    meta["filepath"] = meta["filepath"].map(resolve_project_path)

    classical_bundle = load_classical_bundle(CLASSICAL_MODELS_PATH)
    ml_size = classical_bundle["ml_size"]
    platt_bundle = classical_bundle.get("platt_calibrators", summary.get("platt_calibrators"))

    val_df = meta[meta["split"] == "val"].reset_index(drop=True)
    test_df = meta[meta["split"] == "test"].reset_index(drop=True)
    y_test = test_df["label"].values

    def load_flat(sub_df):
        flat = np.zeros((len(sub_df), ml_size * ml_size), dtype=np.float32)
        for i, fp in enumerate(sub_df["filepath"]):
            with Image.open(fp) as im:
                g = im.convert("L").resize((ml_size, ml_size), Image.Resampling.BILINEAR)
                flat[i] = np.asarray(g, dtype=np.float32).ravel() / 255.0
        return flat

    X_raw_test = load_flat(test_df)
    test_probs_raw = predict_classical_probabilities(classical_bundle, X_raw_test)

    ckpt = torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=False)
    model = PneumoniaCNN(2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_ds = ChestXRayCNNDataset(test_df["filepath"].values, y_test, img_size=128, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    cnn_probs_list = []
    with torch.no_grad():
        for bx, _ in test_loader:
            cnn_probs_list.append(torch.softmax(model(bx), dim=1)[:, 1].numpy())
    test_probs_raw["PneumoniaCNN (PyTorch)"] = np.concatenate(cnn_probs_list)

    test_probs_platt = {}
    for m_name, scores in test_probs_raw.items():
        calibrator = platt_bundle.get(m_name) if isinstance(platt_bundle, dict) else platt_bundle
        if calibrator is None or not hasattr(calibrator, "predict_proba"):
            calibrator = PlattCalibrator()
        test_probs_platt[m_name] = calibrator.predict_proba(scores)

    for m_name, tp in test_probs_raw.items():
        opt_t = summary["val_thresholds_raw"][m_name]
        from run_pipeline import compute_metrics_dict

        summary["results_summary"][m_name]["test_default"] = compute_metrics_dict(y_test, tp, threshold=0.5)
        summary["results_summary"][m_name]["test_val_threshold"] = compute_metrics_dict(
            y_test, tp, threshold=opt_t
        )
        summary["results_summary"][m_name]["test_calibrated"] = summary["results_summary"][m_name][
            "test_val_threshold"
        ]
        summary["results_summary"][m_name]["opt_threshold"] = opt_t

    # Fig 5
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for m_name, tp in test_probs_raw.items():
        fpr, tpr, _ = roc_curve(y_test, tp)
        auc_val = summary["results_summary"][m_name]["test_default"]["roc_auc"]
        axes[0].plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.3f})", lw=2.2, color=MODEL_COLORS[m_name])
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("Krzywe ROC na zbiorze testowym (n=624)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Swoistość)")
    axes[0].set_ylabel("True Positive Rate (Czułość / Recall)")
    axes[0].legend(loc="lower right", fontsize=9)

    for m_name, tp in test_probs_raw.items():
        prec, rec, _ = precision_recall_curve(y_test, tp)
        ap_val = summary["results_summary"][m_name]["test_default"]["pr_auc"]
        axes[1].plot(rec, prec, label=f"{m_name} (AP = {ap_val:.3f})", lw=2.2, color=MODEL_COLORS[m_name])
    axes[1].set_title("Krzywe Precision-Recall (Zbiór testowy)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Czułość (Recall)")
    axes[1].set_ylabel("Precyzja (Precision)")
    axes[1].legend(loc="lower left", fontsize=9)

    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC"]
    x_pos = np.arange(len(metric_keys))
    width = 0.2
    for i, m_name in enumerate(test_probs_raw.keys()):
        vals = [summary["results_summary"][m_name]["test_val_threshold"][k] for k in metric_keys]
        axes[2].bar(
            x_pos + (i - 1.5) * width,
            vals,
            width,
            label=m_name,
            color=MODEL_COLORS[m_name],
            edgecolor="black",
            linewidth=0.5,
        )
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(metric_labels, rotation=20)
    bar_floor = min(summary["results_summary"][m]["test_val_threshold"][k] for m in test_probs_raw for k in metric_keys)
    axes[2].set_ylim(max(0.0, bar_floor - 0.08), 1.03)
    axes[2].set_title("Metryki testowe (próg dobrany na walidacji)", fontsize=11.5, fontweight="bold")
    axes[2].set_ylabel("Wartość metryki")
    axes[2].legend(fontsize=8.5, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_model_comparison_roc_pr.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 6
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    for idx_m, m_name in enumerate(test_probs_raw.keys()):
        m_info = summary["results_summary"][m_name]["test_val_threshold"]
        cm = np.array([[m_info["tn"], m_info["fp"]], [m_info["fn"], m_info["tp"]]])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx_m],
            xticklabels=["NORMAL", "PNEUMONIA"],
            yticklabels=["NORMAL", "PNEUMONIA"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[idx_m].set_title(
            f"{m_name}\n(Acc={m_info['accuracy']*100:.1f}%, F1={m_info['f1']*100:.1f}%)",
            fontsize=10.5,
            fontweight="bold",
        )
        axes[idx_m].set_xlabel("Predykcja modelu")
        axes[idx_m].set_ylabel("Rzeczywista klasa" if idx_m == 0 else "")
    plt.suptitle(
        "Macierze pomyłek na zbiorze testowym (progi z walidacji)",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 9
    cnn_tp_platt = test_probs_platt["PneumoniaCNN (PyTorch)"]
    cnn_opt_t = summary["val_thresholds_platt"]["PneumoniaCNN (PyTorch)"]
    cnn_preds = (cnn_tp_platt >= cnn_opt_t).astype(int)

    fp_indices = np.where((y_test == 0) & (cnn_preds == 1))[0]
    fn_indices = np.where((y_test == 1) & (cnn_preds == 0))[0]
    fp_sorted = fp_indices[np.argsort(-cnn_tp_platt[fp_indices])]
    fn_sorted = fn_indices[np.argsort(cnn_tp_platt[fn_indices])]

    fig = plt.figure(figsize=(15.5, 4.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.35])
    for i in range(2):
        ax_fp = fig.add_subplot(gs[i])
        if i < len(fp_sorted):
            ti = fp_sorted[i]
            ax_fp.imshow(test_ds.cached_imgs[ti], cmap="gray")
            ax_fp.set_title(
                f"Błąd FP #{i+1} (Rzecz.: NORMAL)\nPred: PNEUMONIA ({cnn_tp_platt[ti]*100:.1f}%)",
                fontsize=9.5,
                color="#c0392b",
                fontweight="bold",
            )
        ax_fp.axis("off")

    for i in range(2):
        ax_fn = fig.add_subplot(gs[2 + i])
        if i < len(fn_sorted):
            ti = fn_sorted[i]
            sub = test_df.iloc[ti]["subtype"].replace("PNEUMONIA_", "")
            ax_fn.imshow(test_ds.cached_imgs[ti], cmap="gray")
            ax_fn.set_title(
                f"Błąd FN #{i+1} (Rzecz.: {sub})\nPred: NORMAL (P={cnn_tp_platt[ti]*100:.1f}%)",
                fontsize=9.5,
                color="#2b5c8f",
                fontweight="bold",
            )
        ax_fn.axis("off")

    ax_sub = fig.add_subplot(gs[4])
    test_subtypes = test_df["subtype"].values
    sub_acc_data = []
    for sub_key, sub_label in [
        ("NORMAL", "Zdrowe\n(NORMAL)"),
        ("PNEUMONIA_BACTERIA", "Bakteryjne\n(BACTERIA)"),
        ("PNEUMONIA_VIRUS", "Wirusowe\n(VIRUS)"),
    ]:
        mask_s = test_subtypes == sub_key
        correct_rate = (cnn_preds[mask_s] == y_test[mask_s]).mean() * 100.0
        sub_acc_data.append((sub_label, correct_rate, int(mask_s.sum())))

    bars = ax_sub.bar(
        [x[0] for x in sub_acc_data],
        [x[1] for x in sub_acc_data],
        color=["#2b5c8f", "#c0392b", "#e67e22"],
        edgecolor="black",
        width=0.58,
    )
    ax_sub.set_ylim(0, 108)
    ax_sub.set_ylabel("Dokładność klasyfikacji [%]")
    ax_sub.set_title("Dokładność CNN wg podtypu (próg z walidacji)", fontsize=10.5, fontweight="bold")
    ax_sub.bar_label(bars, fmt="%.1f%%", fontsize=9.5, padding=2)
    plt.suptitle("Analiza błędów CNN na zbiorze testowym", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig9_error_analysis.png", bbox_inches="tight")
    plt.close(fig)

    summary["subtype_test_accuracy_cnn"] = {
        x[0].replace("\n", " "): {"accuracy_pct": float(x[1]), "count": x[2]} for x in sub_acc_data
    }

    from app import predict_single_image

    fig = plt.figure(figsize=(16, 7.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.35], height_ratios=[1, 1], hspace=0.38, wspace=0.52)
    short_labels = {
        "Logistic Regression (PCA)": "Log. Regression",
        "Random Forest": "Random Forest",
        "SVM (RBF + PCA)": "SVM (RBF+PCA)",
        "PneumoniaCNN (PyTorch)": "PneumoniaCNN",
    }
    sample_cases = [
        (test_df[test_df["subtype"] == "PNEUMONIA_BACTERIA"].iloc[2], "Przypadek A: Pacjent z bakteryjnym zapaleniem płuc"),
        (test_df[test_df["subtype"] == "NORMAL"].iloc[5], "Przypadek B: Pacjent zdrowy (Płuca prawidłowe)"),
    ]
    gui_thresholds = classical_bundle.get(
        "gui_model_thresholds_platt", summary.get("gui_model_thresholds_platt", {})
    )
    cnn_thr = gui_thresholds.get("PneumoniaCNN (PyTorch)", 0.5)

    for row_i, (row_data, case_title) in enumerate(sample_cases):
        pil_im = Image.open(row_data["filepath"])
        panel = predict_single_image(pil_im, classical_bundle, model, img_size=128)
        probs_d = panel["probabilities_platt"]
        gray_c, cam_m = panel["gray_cnn"], panel["cam_map"]
        cnn_prob = probs_d["PneumoniaCNN (PyTorch)"]

        ax0 = fig.add_subplot(gs[row_i, 0])
        ax0.imshow(pil_im.convert("L"), cmap="gray")
        ax0.set_title(f"{case_title}\nPlik: {row_data['filename']}", fontsize=10, fontweight="bold")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[row_i, 1])
        ax1.imshow(gray_c, cmap="gray")
        ax1.imshow(cam_m, cmap="jet", alpha=0.42)
        ax1.set_title(
            f"Grad-CAM (walidacyjny próg CNN = {cnn_thr*100:.0f}%)\nP(PNEUMONIA) = {cnn_prob*100:.1f}%",
            fontsize=10,
            fontweight="bold",
        )
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[row_i, 2])
        m_names = [short_labels.get(k, k) for k in probs_d.keys()]
        vals = [probs_d[k] * 100 for k in probs_d.keys()]
        thr_map = {short_labels.get(k, k): gui_thresholds.get(k, 0.5) * 100 for k in probs_d.keys()}
        b_cols = [
            "#c0392b" if probs_d[k] >= gui_thresholds.get(k, 0.5) else "#2b5c8f" for k in probs_d.keys()
        ]
        bars = ax2.barh(m_names, vals, color=b_cols, edgecolor="black", height=0.6)
        for name, thr_pct in thr_map.items():
            if name in m_names:
                ax2.axvline(thr_pct, color="#555555", linestyle=":", lw=1.0, alpha=0.85)
        ax2.set_xlim(0, 114)
        ax2.set_xlabel("Skalibrowane P(PNEUMONIA) [%]")
        ax2.set_title("Panel GUI – osobne progi modeli (Platt)", fontsize=10, fontweight="bold")
        ax2.bar_label(bars, fmt="%.1f%%", fontsize=8.5, padding=3)

    plt.suptitle("Podgląd aplikacji Streamlit z progami z walidacji", fontsize=13.5, fontweight="bold", y=0.98)
    plt.savefig(FIGURES_DIR / "fig10_ui_preview.png", bbox_inches="tight")
    plt.close(fig)

    with open(METRICS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[SUCCESS] Regenerated figures 5, 6, 9, 10 (validation thresholds only).")


if __name__ == "__main__":
    main()