import os

from config import (
    CLASSICAL_MODELS_PATH,
    CNN_MODEL_PATH,
    DATASET_METADATA_PATH,
    MATPLOTLIB_CACHE_DIR,
    ensure_runtime_dirs,
    resolve_project_path,
)

ensure_runtime_dirs()
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE_DIR)

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn.functional as F
from torchvision import transforms

from run_pipeline import PneumoniaCNN, PlattCalibrator, load_classical_bundle, predict_classical_probabilities

st.set_page_config(
    page_title="PneumoniaMNIST – System Wspomagania Diagnostyki RTG Płuc",
    page_icon="🫁",
    layout="wide"
)


@st.cache_resource
def load_all_models():
    classical_bundle = load_classical_bundle(CLASSICAL_MODELS_PATH)
    cnn_ckpt = torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=False)
    cnn_model = PneumoniaCNN(num_classes=2)
    cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
    cnn_model.eval()
    return classical_bundle, cnn_model, cnn_ckpt.get("img_size", 128)


def compute_gradcam_cpu(cnn_model, img_tensor_1x1xHxW, img_size=128):
    cnn_model.eval()
    activations, gradients = [], []
    h1 = cnn_model.block4.register_forward_hook(lambda m, i, o: activations.append(o))
    h2 = cnn_model.block4.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))
    x = img_tensor_1x1xHxW.clone().detach().requires_grad_(True)
    logits = cnn_model(x)
    probs = F.softmax(logits, dim=1)[0].detach().numpy()
    score = logits[0, 1]
    cnn_model.zero_grad()
    score.backward()
    h1.remove()
    h2.remove()
    act = activations[0].detach()[0]
    grad = gradients[0].detach()[0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=0)).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_pil = Image.fromarray(np.uint8(cam * 255)).resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.asarray(cam_pil, dtype=np.float32) / 255.0, float(probs[1])


GUI_MODEL_LABELS = {
    "Logistic Regression": "Logistic Regression (PCA)",
    "Random Forest": "Random Forest",
    "SVM (RBF)": "SVM (RBF + PCA)",
    "PneumoniaCNN (PyTorch)": "PneumoniaCNN (PyTorch)",
}


def _calibrators_from_bundle(classical_bundle):
    return classical_bundle.get("platt_calibrators", {})


def _thresholds_from_bundle(classical_bundle):
    defaults = {label: 0.5 for label in GUI_MODEL_LABELS.values()}
    defaults.update(classical_bundle.get("gui_model_thresholds_platt", {}))
    return defaults


def predict_single_image(pil_img: Image.Image, classical_bundle, cnn_model, img_size=128):
    ml_size = classical_bundle["ml_size"]
    gray_ml = pil_img.convert("L").resize((ml_size, ml_size), Image.Resampling.BILINEAR)
    arr_ml = (np.asarray(gray_ml, dtype=np.float32) / 255.0).reshape(1, -1)

    raw_probs = predict_classical_probabilities(classical_bundle, arr_ml)
    raw_probs = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in raw_probs.items()}

    gray_cnn = pil_img.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.482], std=[0.236]),
        ]
    )
    x_ten = tf(gray_cnn).unsqueeze(0)
    cam_map, prob_cnn_raw = compute_gradcam_cpu(cnn_model, x_ten, img_size=img_size)
    raw_probs["PneumoniaCNN (PyTorch)"] = prob_cnn_raw

    calibrators = _calibrators_from_bundle(classical_bundle)
    probs_platt = {}
    for key, raw in raw_probs.items():
        cal = calibrators.get(key)
        if cal is None:
            cal = PlattCalibrator()
            platt_p = float(raw)
        else:
            platt_p = float(cal.predict_proba(np.asarray([raw]))[0])
        display_name = GUI_MODEL_LABELS.get(key, key)
        probs_platt[display_name] = platt_p

    return {
        "probabilities_raw": {GUI_MODEL_LABELS.get(k, k): float(v) for k, v in raw_probs.items()},
        "probabilities_platt": probs_platt,
        "thresholds_platt": _thresholds_from_bundle(classical_bundle),
        "gray_cnn": np.asarray(gray_cnn),
        "cam_map": cam_map,
    }


def main():
    st.title("🫁 PneumoniaMNIST – Klasyfikacja zdjęć RTG klatki piersiowej")
    st.markdown(
        "**System wspomagania decyzji klinicznych (CDSS)** do wykrywania zapalenia płuc na zdjęciach "
        "rentgenowskich klatki piersiowej. Autorzy: **Kamil Rayski, Miłosz Pabis** (*Politechnika Krakowska*)."
    )
    st.caption(
        "Prototyp badawczo-edukacyjny – nie do użycia klinicznego. Predykcje: skalowanie Platta z walidacji, "
        "osobny próg decyzyjny dla każdego modelu (Youden na walidacji)."
    )

    if not CLASSICAL_MODELS_PATH.exists():
        st.warning("Modele nie zostały jeszcze wytrenowane. Uruchom najpierw `uv run python run_pipeline.py`.")
        return

    classical_bundle, cnn_model, img_size = load_all_models()
    meta_df = pd.read_csv(DATASET_METADATA_PATH)
    meta_df["filepath"] = meta_df["filepath"].map(resolve_project_path)
    test_df = meta_df[meta_df["split"] == "test"].reset_index(drop=True)

    col_left, col_right = st.columns([1, 2.1])

    with col_left:
        st.subheader("1. Wybór zdjęcia RTG")
        mode = st.radio(
            "Źródło obrazu:",
            ["Przykładowe zdjęcie ze zbioru testowego", "Wgraj własny plik RTG (JPEG/PNG)"]
        )

        selected_img = None
        true_label_str = "Nieznana (wgrany plik)"

        if mode == "Przykładowe zdjęcie ze zbioru testowego":
            sub_filter = st.selectbox(
                "Filtruj kategorię kliniczną:",
                ["NORMAL", "PNEUMONIA_BACTERIA", "PNEUMONIA_VIRUS"]
            )
            filtered = test_df[test_df["subtype"] == sub_filter].head(25)
            chosen_fname = st.selectbox("Wybierz plik ze zbioru testowego:", filtered["filename"].tolist())
            row = filtered[filtered["filename"] == chosen_fname].iloc[0]
            selected_img = Image.open(row["filepath"])
            true_label_str = f"{row['label_name']} ({row['subtype']})"
        else:
            uploaded = st.file_uploader("Wybierz zdjęcie RTG:", type=["jpeg", "jpg", "png"])
            if uploaded is not None:
                selected_img = Image.open(uploaded)

        use_custom_cnn_threshold = st.checkbox("Nadpisz próg decyzyjny CNN (tryb eksperymentalny)", value=False)
        model_thresholds = classical_bundle.get("gui_model_thresholds_platt", {})
        default_cnn_thr = float(model_thresholds.get("PneumoniaCNN (PyTorch)", 0.5))
        cnn_threshold = default_cnn_thr
        if use_custom_cnn_threshold:
            cnn_threshold = st.slider(
                "Próg decyzyjny CNN (tylko sieć konwolucyjna):",
                min_value=0.05,
                max_value=0.95,
                value=min(max(default_cnn_thr, 0.05), 0.95),
                step=0.01,
            )
        with st.expander("Progi walidacyjne wszystkich modeli (Platt)"):
            for model_name, thr in sorted(model_thresholds.items()):
                st.write(f"- **{model_name}**: {thr*100:.1f}%")
        st.info(f"**Rzeczywista etykieta kliniczna:** {true_label_str}")

    with col_right:
        if selected_img is not None:
            panel = predict_single_image(selected_img, classical_bundle, cnn_model, img_size)
            probs_dict = panel["probabilities_platt"]
            gray_cnn, cam_map = panel["gray_cnn"], panel["cam_map"]
            thresholds = panel["thresholds_platt"]
            cnn_prob = probs_dict["PneumoniaCNN (PyTorch)"]
            cnn_thr = cnn_threshold if use_custom_cnn_threshold else thresholds["PneumoniaCNN (PyTorch)"]
            diagnosis = "ZAPALENIE PŁUC (PNEUMONIA)" if cnn_prob >= cnn_thr else "PŁUCA PRAWIDŁOWE (NORMAL)"

            if cnn_prob >= cnn_thr:
                st.error(
                    f"### Wynik główny (PneumoniaCNN): {diagnosis} — skalibrowane P(PNEUMONIA): {cnn_prob*100:.1f}% "
                    f"(próg {cnn_thr*100:.0f}%)"
                )
            else:
                st.success(
                    f"### Wynik główny (PneumoniaCNN): {diagnosis} — skalibrowane P(PNEUMONIA): {cnn_prob*100:.1f}% "
                    f"(próg {cnn_thr*100:.0f}%)"
                )

            fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
            axes[0].imshow(selected_img.convert("L"), cmap="gray")
            axes[0].set_title("Oryginalne zdjęcie RTG", fontsize=10, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(gray_cnn, cmap="gray")
            axes[1].imshow(cam_map, cmap="jet", alpha=0.45)
            axes[1].set_title("Mapa Grad-CAM (obszar nacieku)", fontsize=10, fontweight="bold")
            axes[1].axis("off")

            models_list = list(probs_dict.keys())
            vals = [probs_dict[m] * 100 for m in models_list]
            bar_colors = [
                "#c0392b" if probs_dict[m] >= thresholds.get(m, 0.5) else "#2b5c8f" for m in models_list
            ]
            axes[2].barh(models_list, vals, color=bar_colors, edgecolor="black")
            for model_name, thr in thresholds.items():
                if model_name in models_list:
                    axes[2].axvline(thr * 100, color="#555555", linestyle=":", lw=1.0, alpha=0.85)
            axes[2].axvline(
                cnn_thr * 100,
                color="black",
                linestyle="--",
                linewidth=1.4,
                label=f"Próg CNN ({cnn_thr*100:.0f}%)",
            )
            axes[2].set_xlim(0, 105)
            axes[2].set_xlabel("Skalibrowane P(PNEUMONIA) [%]")
            axes[2].set_title("Porównanie modeli (progi z walidacji)", fontsize=10, fontweight="bold")
            axes[2].legend(loc="lower right", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            res_table = pd.DataFrame([
                {
                    "Model ML / DL": m,
                    "P(PNEUMONIA) Platt [%]": f"{p*100:.2f}%",
                    f"Próg walid. [%]": f"{thresholds.get(m, 0.5)*100:.1f}%",
                    "Decyzja": "PNEUMONIA" if p >= thresholds.get(m, 0.5) else "NORMAL",
                }
                for m, p in probs_dict.items()
            ])
            st.dataframe(res_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

