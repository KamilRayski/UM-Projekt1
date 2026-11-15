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

from run_pipeline import PneumoniaCNN, extract_engineered_features

st.set_page_config(
    page_title="PneumoniaMNIST – System Wspomagania Diagnostyki RTG Płuc",
    page_icon="🫁",
    layout="wide"
)


@st.cache_resource
def load_all_models():
    classical_bundle = joblib.load(CLASSICAL_MODELS_PATH)
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


def predict_single_image(pil_img: Image.Image, classical_bundle, cnn_model, img_size=128):
    ml_size = classical_bundle["ml_size"]
    gray_ml = pil_img.convert("L").resize((ml_size, ml_size), Image.Resampling.BILINEAR)
    arr_ml = (np.asarray(gray_ml, dtype=np.float32) / 255.0).reshape(1, -1)

    eng_feat = extract_engineered_features(arr_ml, ml_size=ml_size)
    X_scaled = classical_bundle["pixel_scaler"].transform(arr_ml)
    X_pca = classical_bundle["pca"].transform(X_scaled)
    X_eng_scaled = classical_bundle["eng_scaler"].transform(eng_feat)

    X_full = np.hstack([X_pca, X_eng_scaled])
    X_rf = np.hstack([arr_ml, X_eng_scaled])

    prob_lr = float(classical_bundle["lr"].predict_proba(X_full)[0, 1])
    prob_rf = float(classical_bundle["rf"].predict_proba(X_rf)[0, 1])
    prob_svm = float(classical_bundle["svm"].predict_proba(X_full)[0, 1])

    gray_cnn = pil_img.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.482], std=[0.236]),
    ])
    x_ten = tf(gray_cnn).unsqueeze(0)
    cam_map, prob_cnn = compute_gradcam_cpu(cnn_model, x_ten, img_size=img_size)

    return {
        "Logistic Regression (PCA)": prob_lr,
        "Random Forest": prob_rf,
        "SVM (RBF + PCA)": prob_svm,
        "PneumoniaCNN (PyTorch)": prob_cnn,
    }, np.asarray(gray_cnn), cam_map


def main():
    st.title("🫁 PneumoniaMNIST – Klasyfikacja zdjęć RTG klatki piersiowej")
    st.markdown(
        "**System wspomagania decyzji klinicznych (CDSS)** do wykrywania zapalenia płuc na zdjęciach "
        "rentgenowskich klatki piersiowej. Autorzy: **Kamil Rayski, Miłosz Pabis** (*Politechnika Krakowska*)."
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

        threshold = st.slider(
            "Próg decyzyjny klasyfikatora P(PNEUMONIA):",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05
        )
        st.info(f"**Rzeczywista etykieta kliniczna:** {true_label_str}")

    with col_right:
        if selected_img is not None:
            probs_dict, gray_cnn, cam_map = predict_single_image(selected_img, classical_bundle, cnn_model, img_size)
            cnn_prob = probs_dict["PneumoniaCNN (PyTorch)"]
            diagnosis = "ZAPALENIE PŁUC (PNEUMONIA)" if cnn_prob >= threshold else "PŁUCA PRAWIDŁOWE (NORMAL)"

            if cnn_prob >= threshold:
                st.error(f"### Wynik główny (PneumoniaCNN): {diagnosis} — Prawdopodobieństwo: {cnn_prob*100:.1f}%")
            else:
                st.success(f"### Wynik główny (PneumoniaCNN): {diagnosis} — Prawdopodobieństwo zmian zapalnych: {cnn_prob*100:.1f}%")

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
            bar_colors = ["#c0392b" if v >= threshold * 100 else "#2b5c8f" for v in vals]
            axes[2].barh(models_list, vals, color=bar_colors, edgecolor="black")
            axes[2].axvline(threshold * 100, color="black", linestyle="--", label=f"Próg ({threshold*100:.0f}%)")
            axes[2].set_xlim(0, 105)
            axes[2].set_xlabel("P(PNEUMONIA) [%]")
            axes[2].set_title("Porównanie predykcji 4 modeli", fontsize=10, fontweight="bold")
            axes[2].legend(loc="lower right", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            res_table = pd.DataFrame([
                {
                    "Model ML / DL": m,
                    "P(PNEUMONIA) [%]": f"{p*100:.2f}%",
                    "Decyzja klasyfikatora": "PNEUMONIA" if p >= threshold else "NORMAL"
                }
                for m, p in probs_dict.items()
            ])
            st.dataframe(res_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

