import os
import json
import time
import random
import re
from pathlib import Path

from config import (
    CLASSICAL_MODELS_PATH,
    CNN_MODEL_PATH,
    DATA_DIR,
    DATASET_METADATA_PATH,
    FIGURES_DIR,
    KAGGLE_DATASET_CACHE_DIR,
    KAGGLE_DATASET_ID,
    MATPLOTLIB_CACHE_DIR,
    METRICS_SUMMARY_PATH,
    PROJECT_DIR,
    TEST_PREDICTIONS_PATH,
    ensure_runtime_dirs,
)

ensure_runtime_dirs()
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE_DIR)
os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, balanced_accuracy_score, brier_score_loss
)
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
sns.set_theme(style="whitegrid", palette="muted")


def resolve_dataset_path():
    cached_versions = sorted(KAGGLE_DATASET_CACHE_DIR.glob("versions/*/chest_xray"), reverse=True)
    if cached_versions:
        print(f"[INFO] Using cached dataset at: {cached_versions[0]}")
        return cached_versions[0]
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    print("Path to dataset files:", path)
    return Path(path) / "chest_xray"


def scan_dataset(base_dir: Path) -> pd.DataFrame:
    records = []
    for split in ["train", "val", "test"]:
        for label_idx, cls_name in enumerate(["NORMAL", "PNEUMONIA"]):
            folder = base_dir / split / cls_name
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    continue
                fpath = folder / fname
                subtype = "NORMAL"
                if cls_name == "PNEUMONIA":
                    if "bacteria" in fname.lower():
                        subtype = "PNEUMONIA_BACTERIA"
                    elif "virus" in fname.lower():
                        subtype = "PNEUMONIA_VIRUS"
                    else:
                        subtype = "PNEUMONIA_OTHER"
                records.append({
                    "filepath": str(fpath),
                    "filename": fname,
                    "orig_split": split,
                    "label_name": cls_name,
                    "label": label_idx,
                    "subtype": subtype,
                })
    return pd.DataFrame(records)


def infer_study_group(filename: str) -> str:
    """Return a conservative, stable patient/study group inferred from a filename.

    The Kaggle source does not provide a separate patient identifier.  Its
    pneumonia filenames contain a person number and etiology; normal studies
    use an IM study identifier, sometimes with multiple views.  Keeping the
    two normal filename series distinct avoids treating unrelated records as a
    single patient while still preventing known multi-view studies from being
    split across partitions.
    """
    name = Path(filename).stem.lower()
    pneumonia_match = re.match(r"person(?P<person>\d+)_(?P<etiology>bacteria|virus)", name)
    if pneumonia_match:
        return f"pneumonia-{pneumonia_match['etiology']}-{pneumonia_match['person']}"

    normal_match = re.match(r"(?P<series>normal2-)?im-(?P<study>\d+-\d+)(?:-\d+)?$", name)
    if normal_match:
        series = "normal2" if normal_match["series"] else "normal"
        return f"{series}-{normal_match['study']}"

    # Never silently group unrecognised files: each receives its own group.
    return f"unparsed-{name}"


def grouped_holdout_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    target_size: int,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the closest stratified, group-disjoint holdout to ``target_size``."""
    n_splits = max(2, round(len(labels) / target_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    candidates = list(splitter.split(np.zeros(len(labels)), labels, groups))
    overall_distribution = np.bincount(labels, minlength=len(np.unique(labels))) / len(labels)

    def score(candidate):
        _, holdout_idx = candidate
        distribution = np.bincount(labels[holdout_idx], minlength=len(overall_distribution)) / len(holdout_idx)
        return abs(len(holdout_idx) - target_size) + len(labels) * np.abs(distribution - overall_distribution).sum()

    train_idx, holdout_idx = min(candidates, key=score)
    return train_idx, holdout_idx


def assert_disjoint_groups(df: pd.DataFrame, split_col: str = "split") -> dict:
    """Fail fast if any study_group appears in more than one data split."""
    splits = ("train", "val", "test")
    group_by_split = {
        split: set(df.loc[df[split_col] == split, "study_group"].unique()) for split in splits
    }
    audit = {"group_counts": {s: len(group_by_split[s]) for s in splits}, "overlaps": {}}
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            shared = group_by_split[left] & group_by_split[right]
            audit["overlaps"][f"{left}_vs_{right}"] = {
                "n_shared_groups": len(shared),
                "n_shared_images": int(df["study_group"].isin(shared).sum()) if shared else 0,
            }
            if shared:
                sample = sorted(shared)[:5]
                raise ValueError(
                    f"Patient/study leakage between {left} and {right}: "
                    f"{len(shared)} shared groups (e.g. {sample})"
                )
    print(
        "[INFO] Group-disjoint split audit:",
        {k: v for k, v in audit["group_counts"].items()},
        "| overlaps:",
        audit["overlaps"],
    )
    return audit


def youden_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    return float(thresh[int(np.argmax(tpr - fpr))])


def _safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def bootstrap_roc_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = SEED,
) -> dict:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        auc = _safe_roc_auc(y_true[idx], y_score[idx])
        if not np.isnan(auc):
            samples.append(auc)
    point = _safe_roc_auc(y_true, y_score)
    if not samples:
        return {"point": point, "ci_low": None, "ci_high": None, "n_bootstrap": 0}
    arr = np.asarray(samples, dtype=float)
    return {
        "point": point,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n_bootstrap": len(samples),
    }


def bootstrap_paired_auc_comparison(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    label_a: str,
    label_b: str,
    n_bootstrap: int = 1000,
    random_state: int = SEED,
) -> dict:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    n = len(y_true)
    diffs = []
    wins_a = 0
    valid = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        auc_a = _safe_roc_auc(y_true[idx], prob_a[idx])
        auc_b = _safe_roc_auc(y_true[idx], prob_b[idx])
        if np.isnan(auc_a) or np.isnan(auc_b):
            continue
        valid += 1
        diffs.append(auc_a - auc_b)
        if auc_a > auc_b:
            wins_a += 1
    point_a = _safe_roc_auc(y_true, prob_a)
    point_b = _safe_roc_auc(y_true, prob_b)
    if not diffs:
        return {
            "model_a": label_a,
            "model_b": label_b,
            "auc_a": point_a,
            "auc_b": point_b,
            "mean_auc_diff": point_a - point_b,
            "p_a_greater_b": None,
            "ci_diff_low": None,
            "ci_diff_high": None,
            "n_bootstrap": 0,
        }
    arr = np.asarray(diffs, dtype=float)
    return {
        "model_a": label_a,
        "model_b": label_b,
        "auc_a": point_a,
        "auc_b": point_b,
        "mean_auc_diff": float(arr.mean()),
        "p_a_greater_b": float(wins_a / valid),
        "ci_diff_low": float(np.percentile(arr, 2.5)),
        "ci_diff_high": float(np.percentile(arr, 97.5)),
        "n_bootstrap": valid,
    }


PCA_COMPONENTS = 100


class EngineeredFeaturesTransformer(TransformerMixin, BaseEstimator):
    """Picklable engineering step for sklearn pipelines (64×64 flattened pixels)."""

    def __init__(self, ml_size: int = 64):
        self.ml_size = ml_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return extract_engineered_features(np.asarray(X, dtype=np.float32), ml_size=self.ml_size)


def build_lr_svm_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "pca_branch",
                            Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    ("pca", PCA(n_components=PCA_COMPONENTS, random_state=SEED)),
                                ]
                            ),
                        ),
                        (
                            "eng",
                            Pipeline(
                                [
                                    ("eng", EngineeredFeaturesTransformer()),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", estimator),
        ]
    )


def build_rf_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("raw_pixels", "passthrough"),
                        (
                            "eng",
                            Pipeline(
                                [
                                    ("eng", EngineeredFeaturesTransformer()),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", estimator),
        ]
    )


def extract_pca_from_pipeline(pipeline: Pipeline) -> tuple[StandardScaler, PCA]:
    pca_branch = pipeline.named_steps["features"].transformer_list[0][1]
    return pca_branch.named_steps["scaler"], pca_branch.named_steps["pca"]


def compute_image_stats_and_features(df: pd.DataFrame, ml_size: int = 64):
    print(f"[INFO] Extracting image metadata and {ml_size}x{ml_size} features for {len(df)} images...")
    widths, heights, modes, means, stds = [], [], [], [], []
    flat_images = np.zeros((len(df), ml_size * ml_size), dtype=np.float32)

    for idx, row in enumerate(df.itertuples()):
        with Image.open(row.filepath) as img:
            w, h = img.size
            m = img.mode
            widths.append(w)
            heights.append(h)
            modes.append(m)
            gray = img.convert("L")
            resized = gray.resize((ml_size, ml_size), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
            flat_images[idx] = arr.ravel()

    df = df.copy()
    df["width"] = widths
    df["height"] = heights
    df["aspect_ratio"] = df["width"] / df["height"]
    df["mode"] = modes
    df["mean_intensity"] = means
    df["std_intensity"] = stds
    return df, flat_images


def generate_eda_figures(df: pd.DataFrame, flat_images: np.ndarray, ml_size: int = 64):
    print("[INFO] Generating EDA figures (Fig 1 - Fig 4)...")
    # Fig 1: Class distribution across original vs corrected splits and pneumonia subtypes
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    orig_counts = df.groupby(["orig_split", "label_name"]).size().unstack(fill_value=0)
    orig_counts = orig_counts.loc[["train", "val", "test"]]
    orig_counts.plot(kind="bar", ax=axes[0], color=["#2b5c8f", "#d95f02"], width=0.72, edgecolor="black")
    axes[0].set_title("Oryginalny podział zbioru (Kaggle)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Podzbiór danych")
    axes[0].set_ylabel("Liczba zdjęć RTG")
    axes[0].tick_params(axis="x", rotation=0)
    for container in axes[0].containers:
        axes[0].bar_label(container, fontsize=9, padding=2)

    new_counts = df.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    new_counts = new_counts.loc[["train", "val", "test"]]
    new_counts.plot(kind="bar", ax=axes[1], color=["#2b5c8f", "#d95f02"], width=0.72, edgecolor="black")
    axes[1].set_title("Skorygowany podział (standard PneumoniaMNIST)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Podzbiór danych")
    axes[1].set_ylabel("Liczba zdjęć RTG")
    axes[1].tick_params(axis="x", rotation=0)
    for container in axes[1].containers:
        axes[1].bar_label(container, fontsize=9, padding=2)

    sub_counts = df["subtype"].value_counts()
    sub_labels = {
        "PNEUMONIA_BACTERIA": "Zapalenie płuc\n(bakteryjne)",
        "NORMAL": "Płuca zdrowe\n(NORMAL)",
        "PNEUMONIA_VIRUS": "Zapalenie płuc\n(wirusowe)"
    }
    bars = axes[2].bar(
        [sub_labels.get(k, k) for k in sub_counts.index],
        sub_counts.values,
        color=["#c0392b", "#2b5c8f", "#e67e22"],
        edgecolor="black",
        width=0.6
    )
    axes[2].set_title("Podział etiologiczny w całym zbiorze (n=5856)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Liczba zdjęć RTG")
    axes[2].bar_label(bars, fmt="%d", fontsize=9, padding=2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_class_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Sample Chest X-Rays Grid (Normal, Bacterial Pneumonia, Viral Pneumonia)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
    categories = [
        ("NORMAL", "NORMAL (Płuca prawidłowe)"),
        ("PNEUMONIA_BACTERIA", "PNEUMONIA – Bakteryjne zapalenie płuc"),
        ("PNEUMONIA_VIRUS", "PNEUMONIA – Wirusowe zapalenie płuc"),
    ]
    for row_idx, (sub, title_prefix) in enumerate(categories):
        sample_rows = df[df["subtype"] == sub].iloc[:4]
        for col_idx, r in enumerate(sample_rows.itertuples()):
            ax = axes[row_idx, col_idx]
            with Image.open(r.filepath) as im:
                gray = im.convert("L")
                ax.imshow(gray, cmap="gray")
                ax.set_title(f"{title_prefix.split(' ')[0]}\n({r.width}x{r.height} px, śr={r.mean_intensity:.2f})", fontsize=9.5)
                ax.axis("off")
    plt.suptitle("Przykładowe zdjęcia rentgenowskie klatki piersiowej ze zbioru danych", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_sample_xray_grid.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: Image Statistics & Mean Lung Templates
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(
        data=df.sample(min(1500, len(df)), random_state=SEED),
        x="width", y="height", hue="label_name", alpha=0.5, s=22,
        palette={"NORMAL": "#2b5c8f", "PNEUMONIA": "#d95f02"}, ax=ax1
    )
    ax1.set_title("Rozkład wymiarów obrazów (szerokość vs wysokość)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Szerokość [px]")
    ax1.set_ylabel("Wysokość [px]")

    ax2 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(data=df, x="mean_intensity", hue="label_name", fill=True, common_norm=False,
                palette={"NORMAL": "#2b5c8f", "PNEUMONIA": "#d95f02"}, ax=ax2)
    ax2.set_title("Rozkład średniej jasności pikseli", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Średnia znormalizowana jasność [0-1]")
    ax2.set_ylabel("Gęstość")

    ax3 = fig.add_subplot(gs[0, 2])
    sns.kdeplot(data=df, x="std_intensity", hue="label_name", fill=True, common_norm=False,
                palette={"NORMAL": "#2b5c8f", "PNEUMONIA": "#d95f02"}, ax=ax3)
    ax3.set_title("Rozkład kontrastu (odchylenia std pikseli)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Odchylenie standardowe jasności")
    ax3.set_ylabel("Gęstość")

    mean_norm = flat_images[df["label"] == 0].mean(axis=0).reshape(ml_size, ml_size)
    mean_pneu = flat_images[df["label"] == 1].mean(axis=0).reshape(ml_size, ml_size)
    diff_img = mean_pneu - mean_norm

    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(mean_norm, cmap="bone")
    ax4.set_title("Uśredniony obraz RTG: NORMAL", fontsize=11, fontweight="bold")
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(mean_pneu, cmap="bone")
    ax5.set_title("Uśredniony obraz RTG: PNEUMONIA", fontsize=11, fontweight="bold")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(diff_img, cmap="coolwarm", vmin=-0.12, vmax=0.12)
    ax6.set_title("Mapa różnicowa (PNEUMONIA - NORMAL)", fontsize=11, fontweight="bold")
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_image_stats_eda.png", bbox_inches="tight")
    plt.close(fig)


def extract_engineered_features(flat_images: np.ndarray, ml_size: int = 64) -> np.ndarray:
    """
    Combines raw pixel intensities (64x64=4096) with domain-specific statistical & regional lung features:
    - Global mean, std, skewness, kurtosis, quantiles (10%, 25%, 50%, 75%, 90%)
    - Left lung field vs Right lung field mean/std/asymmetry
    - Upper/Middle/Lower lung zone intensities
    - Horizontal & Vertical Sobel-like gradient magnitudes
    """
    n = flat_images.shape[0]
    imgs = flat_images.reshape(n, ml_size, ml_size)
    feats = []
    for i in range(n):
        im = imgs[i]
        g_mean = im.mean()
        g_std = im.std()
        q10, q25, q50, q75, q90 = np.quantile(im, [0.1, 0.25, 0.5, 0.75, 0.9])
        # Central lung ROI (excluding borders/shoulders)
        roi = im[8:56, 8:56]
        left_lung = roi[:, :24]
        right_lung = roi[:, 24:]
        upper_zone = roi[:16, :]
        mid_zone = roi[16:32, :]
        lower_zone = roi[32:, :]
        # Gradients
        gx = np.abs(np.diff(im, axis=1)).mean()
        gy = np.abs(np.diff(im, axis=0)).mean()
        feats.append([
            g_mean, g_std, q10, q25, q50, q75, q90,
            left_lung.mean(), left_lung.std(),
            right_lung.mean(), right_lung.std(),
            abs(left_lung.mean() - right_lung.mean()),
            upper_zone.mean(), mid_zone.mean(), lower_zone.mean(),
            gx, gy
        ])
    return np.array(feats, dtype=np.float32)


def engineered_feature_transformer(flat_images: np.ndarray) -> np.ndarray:
    """sklearn-compatible, picklable transformation for 64×64 raw pixels."""
    return extract_engineered_features(flat_images, ml_size=64)


class PlattCalibrator:
    """Train-only Platt scaling for a model's positive-class scores."""

    def __init__(self):
        self.model = LogisticRegression(C=1_000_000.0, max_iter=1_000, random_state=SEED)

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray):
        self.model.fit(self._logit(probabilities), labels)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(probabilities))[:, 1]


def register_pickle_main_aliases() -> None:
    """Expose pipeline symbols on ``__main__`` for joblib pickles saved via ``python run_pipeline.py``."""
    import sys

    import run_pipeline as pipeline_mod

    main = sys.modules["__main__"]
    for name in (
        "engineered_feature_transformer",
        "EngineeredFeaturesTransformer",
        "PlattCalibrator",
    ):
        if hasattr(pipeline_mod, name):
            setattr(main, name, getattr(pipeline_mod, name))


def load_classical_bundle(path: Path | str):
    register_pickle_main_aliases()
    return joblib.load(path)


def predict_classical_probabilities(classical_bundle: dict, raw_pixels: np.ndarray) -> dict[str, np.ndarray]:
    """Predict with both the corrected pipeline bundle and the legacy bundle."""
    register_pickle_main_aliases()
    if "classical_models" in classical_bundle:
        return {
            name: model.predict_proba(raw_pixels)[:, 1]
            for name, model in classical_bundle["classical_models"].items()
        }

    # Compatibility with artifacts generated before the grouped-CV migration.
    ml_size = classical_bundle["ml_size"]
    engineered = extract_engineered_features(raw_pixels, ml_size=ml_size)
    full_features = np.hstack([
        classical_bundle["pca"].transform(classical_bundle["pixel_scaler"].transform(raw_pixels)),
        classical_bundle["eng_scaler"].transform(engineered),
    ])
    rf_features = np.hstack([raw_pixels, classical_bundle["eng_scaler"].transform(engineered)])
    return {
        "Logistic Regression": classical_bundle["lr"].predict_proba(full_features)[:, 1],
        "Random Forest": classical_bundle["rf"].predict_proba(rf_features)[:, 1],
        "SVM (RBF)": classical_bundle["svm"].predict_proba(full_features)[:, 1],
    }


class ChestXRayCNNDataset(Dataset):
    def __init__(self, filepaths, labels, img_size=128, is_train=False):
        self.filepaths = list(filepaths)
        self.labels = list(labels)
        self.img_size = img_size
        self.is_train = is_train
        # Preload resized 128x128 uint8 arrays in RAM for super fast epoch training!
        self.cached_imgs = []
        for fp in self.filepaths:
            with Image.open(fp) as im:
                g = im.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
                self.cached_imgs.append(np.asarray(g, dtype=np.uint8))

        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.482], std=[0.236]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.482], std=[0.236]),
            ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        x = self.transform(self.cached_imgs[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 64x64
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32x32
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 16x16
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        pooled = self.pool(feat)
        out = self.classifier(pooled)
        return out


def compute_metrics_dict(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(spec),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold)
    }


def main():
    t0 = time.time()
    base_dir = resolve_dataset_path()
    df = scan_dataset(base_dir)
    print(f"[INFO] Scanned {len(df)} total images.")
    df["study_group"] = df["filename"].map(infer_study_group)

    # Group-disjoint holdout: ~524 validation images from original train+val; official test untouched.
    train_val_mask = df["orig_split"].isin(["train", "val"])
    df_tv = df[train_val_mask].copy()
    tv_labels = df_tv["label"].to_numpy()
    tv_groups = df_tv["study_group"].to_numpy()
    tv_train_pos, tv_val_pos = grouped_holdout_indices(tv_labels, tv_groups, target_size=524, random_state=SEED)

    df["split"] = np.where(df["orig_split"] == "test", "test", "train")
    val_global_idx = df_tv.index[tv_val_pos]
    df.loc[val_global_idx, "split"] = "val"

    leakage_audit = assert_disjoint_groups(df)
    print("[INFO] Split counts after group-disjoint validation correction:")
    print(df.groupby(["split", "label_name"]).size())

    ml_size = 64
    df, flat_images = compute_image_stats_and_features(df, ml_size=ml_size)
    generate_eda_figures(df, flat_images, ml_size=ml_size)

    # Persist portable paths while retaining full paths for this pipeline run.
    metadata_df = df.copy()
    metadata_df["filepath"] = metadata_df["filepath"].map(
        lambda path: os.path.relpath(path, PROJECT_DIR)
    )
    metadata_df.to_csv(DATASET_METADATA_PATH, index=False)

    # Prepare train/val/test arrays for classical ML
    eng_features = extract_engineered_features(flat_images, ml_size=ml_size)

    idx_train = np.where(df["split"] == "train")[0]
    idx_val = np.where(df["split"] == "val")[0]
    idx_test = np.where(df["split"] == "test")[0]

    X_raw_train, y_train = flat_images[idx_train], df["label"].values[idx_train]
    X_raw_val, y_val = flat_images[idx_val], df["label"].values[idx_val]
    X_raw_test, y_test = flat_images[idx_test], df["label"].values[idx_test]
    groups_train = df.iloc[idx_train]["study_group"].to_numpy()

    # PCA summary for EDA (train-only fit; mirrors the LR/SVM pipeline branch).
    print("[INFO] Fitting train-only StandardScaler + PCA for EDA figures...")
    pixel_scaler = StandardScaler()
    X_scaled_train = pixel_scaler.fit_transform(X_raw_train)
    n_components = PCA_COMPONENTS
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca_train = pca.fit_transform(X_scaled_train)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"[INFO] PCA with {n_components} components explains {cum_var[-1]*100:.2f}% variance.")

    # Generate Fig 4: PCA Variance, 2D Projection & Eigenlungs
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.3])

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(range(1, n_components + 1), cum_var * 100, color="#2b5c8f", lw=2.5)
    ax1.axhline(90, color="#d95f02", linestyle="--", label="90% wariancji")
    ax1.axvline(np.searchsorted(cum_var, 0.90) + 1, color="#d95f02", linestyle=":")
    ax1.set_title("Skumulowana wariancja wyjaśniona (PCA)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Liczba składowych głównych (PC)")
    ax1.set_ylabel("Wyjaśniona wariancja [%]")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    sample_idx = np.random.choice(len(X_pca_train), size=1200, replace=False)
    scatter = ax2.scatter(
        X_pca_train[sample_idx, 0], X_pca_train[sample_idx, 1],
        c=y_train[sample_idx], cmap="coolwarm", alpha=0.55, s=18, edgecolors="none"
    )
    ax2.set_title("Rzutowanie zbioru treningowego na PC1 vs PC2", fontsize=11, fontweight="bold")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% wariancji)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% wariancji)")
    handles, _ = scatter.legend_elements()
    ax2.legend(handles, ["NORMAL", "PNEUMONIA"], title="Klasa")

    # Eigenlungs subgrid (first 4 principal components)
    gs_sub = gs[2].subgridspec(2, 2, wspace=0.1, hspace=0.25)
    for k in range(4):
        ax_e = fig.add_subplot(gs_sub[k // 2, k % 2])
        eigen_img = pca.components_[k].reshape(ml_size, ml_size)
        ax_e.imshow(eigen_img, cmap="RdBu_r")
        ax_e.set_title(f"Eigenlung PC{k+1} ({pca.explained_variance_ratio_[k]*100:.1f}%)", fontsize=9.5, fontweight="bold")
        ax_e.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_pca_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Classical ML Models + grouped GridSearchCV (preprocessing inside pipelines)
    # ---------------------------------------------------------
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    results_summary = {}
    val_probs_raw = {}
    test_probs_raw = {}
    val_probs_platt = {}
    test_probs_platt = {}
    platt_calibrators: dict[str, PlattCalibrator] = {}
    val_thresholds: dict[str, float] = {}
    val_thresholds_platt: dict[str, float] = {}
    best_params_summary = {}

    print("[INFO] Tuning Logistic Regression via grouped GridSearchCV...")
    lr_grid = GridSearchCV(
        build_lr_svm_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
        param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    lr_grid.fit(X_raw_train, y_train, groups=groups_train)
    best_lr = lr_grid.best_estimator_
    best_params_summary["Logistic Regression"] = {
        "best_params": {k.replace("clf__", ""): v for k, v in lr_grid.best_params_.items()},
        "cv_roc_auc": float(lr_grid.best_score_),
    }
    val_probs_raw["Logistic Regression"] = best_lr.predict_proba(X_raw_val)[:, 1]
    test_probs_raw["Logistic Regression"] = best_lr.predict_proba(X_raw_test)[:, 1]
    print(f"  -> Best LR: {best_params_summary['Logistic Regression']['best_params']}, CV AUC={lr_grid.best_score_:.4f}")

    print("[INFO] Tuning Random Forest via grouped GridSearchCV...")
    rf_grid = GridSearchCV(
        build_rf_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1)),
        param_grid={
            "clf__n_estimators": [150, 300],
            "clf__max_depth": [12, 24, None],
            "clf__class_weight": ["balanced", "balanced_subsample"],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    rf_grid.fit(X_raw_train, y_train, groups=groups_train)
    best_rf = rf_grid.best_estimator_
    best_params_summary["Random Forest"] = {
        "best_params": {k.replace("clf__", ""): v for k, v in rf_grid.best_params_.items()},
        "cv_roc_auc": float(rf_grid.best_score_),
    }
    val_probs_raw["Random Forest"] = best_rf.predict_proba(X_raw_val)[:, 1]
    test_probs_raw["Random Forest"] = best_rf.predict_proba(X_raw_test)[:, 1]
    print(f"  -> Best RF: {best_params_summary['Random Forest']['best_params']}, CV AUC={rf_grid.best_score_:.4f}")

    print("[INFO] Tuning SVM (RBF) via grouped GridSearchCV...")
    svm_grid = GridSearchCV(
        build_lr_svm_pipeline(SVC(kernel="rbf", probability=True, random_state=SEED)),
        param_grid={
            "clf__C": [0.5, 1.0, 5.0, 10.0],
            "clf__gamma": ["scale", 0.001, 0.0005],
            "clf__class_weight": [None, "balanced"],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    svm_grid.fit(X_raw_train, y_train, groups=groups_train)
    best_svm = svm_grid.best_estimator_
    best_params_summary["SVM (RBF)"] = {
        "best_params": {k.replace("clf__", ""): v for k, v in svm_grid.best_params_.items()},
        "cv_roc_auc": float(svm_grid.best_score_),
    }
    val_probs_raw["SVM (RBF)"] = best_svm.predict_proba(X_raw_val)[:, 1]
    test_probs_raw["SVM (RBF)"] = best_svm.predict_proba(X_raw_test)[:, 1]
    print(f"  -> Best SVM: {best_params_summary['SVM (RBF)']['best_params']}, CV AUC={svm_grid.best_score_:.4f}")

    for m_name in ["Logistic Regression", "Random Forest", "SVM (RBF)"]:
        cal = PlattCalibrator().fit(val_probs_raw[m_name], y_val)
        platt_calibrators[m_name] = cal
        val_probs_platt[m_name] = cal.predict_proba(val_probs_raw[m_name])
        test_probs_platt[m_name] = cal.predict_proba(test_probs_raw[m_name])
        val_thresholds[m_name] = youden_optimal_threshold(y_val, val_probs_raw[m_name])
        val_thresholds_platt[m_name] = youden_optimal_threshold(y_val, val_probs_platt[m_name])

    # ---------------------------------------------------------
    # 4. PyTorch Convolutional Neural Network (PneumoniaCNN)
    # ---------------------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Training PyTorch PneumoniaCNN on device: {device}...")

    img_size_cnn = 128
    train_ds = ChestXRayCNNDataset(df["filepath"].values[idx_train], y_train, img_size=img_size_cnn, is_train=True)
    val_ds = ChestXRayCNNDataset(df["filepath"].values[idx_val], y_val, img_size=img_size_cnn, is_train=False)
    test_ds = ChestXRayCNNDataset(df["filepath"].values[idx_test], y_test, img_size=img_size_cnn, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = PneumoniaCNN(num_classes=2).to(device)
    # Inverse class frequency weighting to combat 74/26 class imbalance
    n_norm = int((y_train == 0).sum())
    n_pneu = int((y_train == 1).sum())
    class_weights = torch.tensor([len(y_train) / (2.0 * n_norm), len(y_train) / (2.0 * n_pneu)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)
        scheduler.step()
        tr_loss = running_loss / total
        tr_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        v_probs_list, v_targets_list = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                probs = F.softmax(logits, dim=1)[:, 1]
                v_loss += loss.item() * bx.size(0)
                v_correct += (logits.argmax(dim=1) == by).sum().item()
                v_total += bx.size(0)
                v_probs_list.append(probs.cpu().numpy())
                v_targets_list.append(by.cpu().numpy())
        val_loss_ep = v_loss / v_total
        val_acc_ep = v_correct / v_total
        v_probs_arr = np.concatenate(v_probs_list)
        v_targets_arr = np.concatenate(v_targets_list)
        val_auc_ep = roc_auc_score(v_targets_arr, v_probs_arr)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss_ep)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc_ep)
        history["val_auc"].append(val_auc_ep)

        if val_auc_ep > best_val_auc:
            best_val_auc = val_auc_ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | Val Loss={val_loss_ep:.4f} Acc={val_acc_ep:.4f} AUC={val_auc_ep:.4f}")

    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "img_size": img_size_cnn,
        "history": history,
        "best_val_auc": best_val_auc
    }, CNN_MODEL_PATH)

    # Get CNN validation & test probabilities
    model.eval()
    def predict_cnn_loader(loader):
        probs_all = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(device)
                logits = model(bx)
                probs = F.softmax(logits, dim=1)[:, 1]
                probs_all.append(probs.cpu().numpy())
        return np.concatenate(probs_all)

    val_probs_raw["PneumoniaCNN (PyTorch)"] = predict_cnn_loader(val_loader)
    test_probs_raw["PneumoniaCNN (PyTorch)"] = predict_cnn_loader(test_loader)
    best_params_summary["PneumoniaCNN (PyTorch)"] = {
        "best_params": {
            "img_size": "128x128",
            "optimizer": "AdamW (lr=1e-3, wd=1e-4)",
            "scheduler": "CosineAnnealingLR",
            "loss": "Weighted CrossEntropy",
            "epochs": epochs,
        },
        "cv_roc_auc": float(best_val_auc),
    }
    cnn_platt = PlattCalibrator().fit(val_probs_raw["PneumoniaCNN (PyTorch)"], y_val)
    platt_calibrators["PneumoniaCNN (PyTorch)"] = cnn_platt
    val_probs_platt["PneumoniaCNN (PyTorch)"] = cnn_platt.predict_proba(val_probs_raw["PneumoniaCNN (PyTorch)"])
    test_probs_platt["PneumoniaCNN (PyTorch)"] = cnn_platt.predict_proba(test_probs_raw["PneumoniaCNN (PyTorch)"])
    val_thresholds["PneumoniaCNN (PyTorch)"] = youden_optimal_threshold(
        y_val, val_probs_raw["PneumoniaCNN (PyTorch)"]
    )
    val_thresholds_platt["PneumoniaCNN (PyTorch)"] = youden_optimal_threshold(
        y_val, val_probs_platt["PneumoniaCNN (PyTorch)"]
    )

    gui_threshold_keys = {
        "Logistic Regression": "Logistic Regression (PCA)",
        "Random Forest": "Random Forest",
        "SVM (RBF)": "SVM (RBF + PCA)",
        "PneumoniaCNN (PyTorch)": "PneumoniaCNN (PyTorch)",
    }
    gui_val_thresholds_platt = {
        gui_threshold_keys[k]: val_thresholds_platt[k] for k in val_thresholds_platt
    }

    joblib.dump(
        {
            "classical_models": {
                "Logistic Regression": best_lr,
                "Random Forest": best_rf,
                "SVM (RBF)": best_svm,
            },
            "platt_calibrators": platt_calibrators,
            "val_thresholds_raw": val_thresholds,
            "val_thresholds_platt": val_thresholds_platt,
            "gui_model_thresholds_platt": gui_val_thresholds_platt,
            "ml_size": ml_size,
            "pca_explained_variance_100": float(cum_var[-1]),
            # Train-only transforms for spatial explainability maps (not used at inference).
            "eda_pixel_scaler": pixel_scaler,
            "eda_pca": pca,
        },
        CLASSICAL_MODELS_PATH,
    )

    bootstrap_cis = {}
    for m_name, tp in test_probs_raw.items():
        bootstrap_cis[m_name] = bootstrap_roc_auc_ci(y_test, tp)

    paired_comparisons = []
    model_names = list(test_probs_raw.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            paired_comparisons.append(
                bootstrap_paired_auc_comparison(
                    y_test,
                    test_probs_raw[model_names[i]],
                    test_probs_raw[model_names[j]],
                    model_names[i],
                    model_names[j],
                )
            )

    test_meta = df.iloc[idx_test][
        ["filename", "filepath", "label", "label_name", "subtype", "study_group"]
    ].reset_index(drop=True)
    predictions_frames = [test_meta.copy()]
    for m_name, tp in test_probs_raw.items():
        predictions_frames.append(pd.Series(tp, name=f"score_raw__{m_name}"))
        predictions_frames.append(pd.Series(test_probs_platt[m_name], name=f"score_platt__{m_name}"))
    test_predictions_df = pd.concat(predictions_frames, axis=1)
    test_predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    # Evaluate all models: validation metrics, test@0.5, test@validation Youden threshold (never tuned on test).
    for m_name in model_names:
        vp = val_probs_raw[m_name]
        tp = test_probs_raw[m_name]
        vp_platt = val_probs_platt[m_name]
        tp_platt = test_probs_platt[m_name]
        val_m = compute_metrics_dict(y_val, vp, threshold=0.5)
        test_m = compute_metrics_dict(y_test, tp, threshold=0.5)
        opt_t = val_thresholds[m_name]
        test_m_val_threshold = compute_metrics_dict(y_test, tp, threshold=opt_t)
        test_m_platt = compute_metrics_dict(y_test, tp_platt, threshold=val_thresholds_platt[m_name])
        val_m_platt = compute_metrics_dict(y_val, vp_platt, threshold=val_thresholds_platt[m_name])
        results_summary[m_name] = {
            "val": val_m,
            "val_platt": val_m_platt,
            "test_default": test_m,
            "test_val_threshold": test_m_val_threshold,
            "test_platt_at_val_threshold": test_m_platt,
            # Backward-compatible keys consumed by older report scripts.
            "test_calibrated": test_m_val_threshold,
            "opt_threshold": opt_t,
            "val_threshold_platt": val_thresholds_platt[m_name],
            "brier_score_test_raw": float(brier_score_loss(y_test, tp)),
            "brier_score_test_platt": float(brier_score_loss(y_test, tp_platt)),
            "bootstrap_roc_auc": bootstrap_cis[m_name],
        }
        ci = bootstrap_cis[m_name]
        ci_txt = (
            f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
            if ci["ci_low"] is not None and ci["ci_high"] is not None
            else "[n/a]"
        )
        print(
            f"[RESULT] {m_name:24s} | Val AUC={val_m['roc_auc']:.4f} | "
            f"Test AUC={test_m['roc_auc']:.4f} {ci_txt} | "
            f"Test@val_thr Acc={test_m_val_threshold['accuracy']:.4f}"
        )

    test_probs = test_probs_raw
    val_probs = val_probs_raw

    # ---------------------------------------------------------
    # Fig 5: Model Comparison (ROC curves, PR curves, Metrics Bar Chart)
    # ---------------------------------------------------------
    print("[INFO] Generating Model Evaluation & Explainability figures (Fig 5 - Fig 9)...")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#2ca02c",
        "SVM (RBF)": "#ff7f0e",
        "PneumoniaCNN (PyTorch)": "#d62728"
    }

    for m_name, tp in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, tp)
        auc_val = results_summary[m_name]["test_default"]["roc_auc"]
        axes[0].plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.3f})", lw=2.2, color=colors[m_name])
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("Krzywe ROC na zbiorze testowym (n=624)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Swoistość)")
    axes[0].set_ylabel("True Positive Rate (Czułość / Recall)")
    axes[0].legend(loc="lower right", fontsize=9)

    for m_name, tp in test_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, tp)
        ap_val = results_summary[m_name]["test_default"]["pr_auc"]
        axes[1].plot(rec, prec, label=f"{m_name} (AP = {ap_val:.3f})", lw=2.2, color=colors[m_name])
    axes[1].set_title("Krzywe Precision-Recall (Zbiór testowy)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Czułość (Recall)")
    axes[1].set_ylabel("Precyzja (Precision)")
    axes[1].legend(loc="lower left", fontsize=9)

    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC"]
    x_pos = np.arange(len(metric_keys))
    width = 0.2
    for i, m_name in enumerate(test_probs.keys()):
        # Show calibrated test metrics for fair clinical comparison
        vals = [results_summary[m_name]["test_val_threshold"][k] for k in metric_keys]
        axes[2].bar(x_pos + (i - 1.5) * width, vals, width, label=m_name, color=colors[m_name], edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(metric_labels, rotation=20)
    bar_floor = min(
        results_summary[m]["test_val_threshold"][k]
        for m in test_probs
        for k in metric_keys
    )
    axes[2].set_ylim(max(0.0, bar_floor - 0.08), 1.03)
    axes[2].set_title(
        "Metryki na zbiorze testowym (próg z walidacji, bez strojenia na teście)",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[2].set_ylabel("Wartość metryki")
    axes[2].legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_model_comparison_roc_pr.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Fig 6: Confusion Matrices on Test Set
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    for idx_m, m_name in enumerate(test_probs.keys()):
        m_info = results_summary[m_name]["test_val_threshold"]
        cm = np.array([[m_info["tn"], m_info["fp"]], [m_info["fn"], m_info["tp"]]])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx_m],
            xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"],
            annot_kws={"size": 13, "weight": "bold"}
        )
        axes[idx_m].set_title(f"{m_name}\n(Acc={m_info['accuracy']*100:.1f}%, F1={m_info['f1']*100:.1f}%)", fontsize=10.5, fontweight="bold")
        axes[idx_m].set_xlabel("Predykcja modelu")
        axes[idx_m].set_ylabel("Rzeczywista klasa" if idx_m == 0 else "")
    plt.suptitle("Macierze pomyłek (Confusion Matrices) na zbiorze testowym (n=624)", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Fig 7: CNN Training History Curves
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ep_range = range(1, epochs + 1)
    axes[0].plot(ep_range, history["train_loss"], "o-", label="Zbiór treningowy (Train Loss)", color="#1f77b4", lw=2)
    axes[0].plot(ep_range, history["val_loss"], "s--", label="Zbiór walidacyjny (Val Loss)", color="#d62728", lw=2)
    axes[0].set_title("Przebieg funkcji straty (Weighted Cross-Entropy)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Wartość funkcji straty (Loss)")
    axes[0].legend()

    axes[1].plot(ep_range, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy [%]", color="#1f77b4", lw=2)
    axes[1].plot(ep_range, [a * 100 for a in history["val_acc"]], "s--", label="Val Accuracy [%]", color="#2ca02c", lw=2)
    axes[1].plot(ep_range, [a * 100 for a in history["val_auc"]], "^-.", label="Val ROC-AUC [%]", color="#d62728", lw=2)
    axes[1].set_title("Dokładność (Accuracy) oraz ROC-AUC w trakcie uczenia CNN", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Wartość [%]")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_cnn_training_curves.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Fig 8: Explainability (Random Forest 2D Pixel Importance + Engineered Feature Importance + Grad-CAM for CNN)
    # ---------------------------------------------------------
    rf_importances = best_rf.named_steps["clf"].feature_importances_
    pixel_imp_2d = rf_importances[:ml_size * ml_size].reshape(ml_size, ml_size)
    eng_imp = rf_importances[ml_size * ml_size:]
    eng_names = [
        "Średnia jasność", "Odch. std (kontrast)", "Kwantyl 10%", "Kwantyl 25%",
        "Mediana (Q50)", "Kwantyl 75%", "Kwantyl 90%", "Lewe płuco (śr.)",
        "Lewe płuco (std)", "Prawe płuco (śr.)", "Prawe płuco (std)",
        "Asymetria L-P", "Górna strefa płuc", "Środkowa strefa płuc",
        "Dolna strefa płuc", "Gradient poziomy", "Gradient pionowy"
    ]

    # Compute Grad-CAM on 3 test PNEUMONIA examples
    def compute_gradcam(cnn_model, img_tensor_1x1xHxW):
        cnn_model.eval()
        activations = []
        gradients = []
        def fwd_hook(module, inp, out):
            activations.append(out)
        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])
        h1 = cnn_model.block4.register_forward_hook(fwd_hook)
        h2 = cnn_model.block4.register_full_backward_hook(bwd_hook)
        img_tensor_1x1xHxW = img_tensor_1x1xHxW.to(device).requires_grad_(True)
        logits = cnn_model(img_tensor_1x1xHxW)
        score = logits[0, 1]
        cnn_model.zero_grad()
        score.backward()
        h1.remove()
        h2.remove()
        act = activations[0].detach().cpu()[0]      # (256, 16, 16)
        grad = gradients[0].detach().cpu()[0]       # (256, 16, 16)
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0)).numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_pil = Image.fromarray(np.uint8(cam * 255)).resize((img_size_cnn, img_size_cnn), Image.Resampling.BILINEAR)
        return np.asarray(cam_pil, dtype=np.float32) / 255.0, float(torch.softmax(logits, dim=1)[0, 1].item())

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])

    # Top-left: 2D Spatial Feature Importance from Random Forest overlaid on mean lung
    ax_imp2d = fig.add_subplot(gs[0, 0])
    mean_lung = flat_images[idx_train].mean(axis=0).reshape(ml_size, ml_size)
    ax_imp2d.imshow(mean_lung, cmap="gray")
    im_heat = ax_imp2d.imshow(pixel_imp_2d, cmap="inferno", alpha=0.65)
    ax_imp2d.set_title("Mapa istotności pikseli 2D (Random Forest)\nnałożona na sylwetkę klatki piersiowej", fontsize=10.5, fontweight="bold")
    ax_imp2d.axis("off")
    plt.colorbar(im_heat, ax=ax_imp2d, fraction=0.046, pad=0.04)

    # Top-middle: Logistic Regression / SVM effective 2D weight map via PCA inverse transform
    ax_lr2d = fig.add_subplot(gs[0, 1])
    _, pipeline_pca = extract_pca_from_pipeline(best_lr)
    lr_clf = best_lr.named_steps["clf"]
    lr_pca_weights = lr_clf.coef_[0][:n_components]
    lr_spatial_map = np.dot(lr_pca_weights, pipeline_pca.components_).reshape(ml_size, ml_size)
    vmax_lr = np.percentile(np.abs(lr_spatial_map), 98)
    im_lr = ax_lr2d.imshow(lr_spatial_map, cmap="coolwarm", vmin=-vmax_lr, vmax=vmax_lr)
    ax_lr2d.set_title("Przestrzenna mapa wag (Regresja Logistyczna + PCA)\n[Czerwony = PNEUMONIA, Niebieski = NORMAL]", fontsize=10.5, fontweight="bold")
    ax_lr2d.axis("off")
    plt.colorbar(im_lr, ax=ax_lr2d, fraction=0.046, pad=0.04)

    # Top-right: Top Engineered Features bar plot
    ax_eng = fig.add_subplot(gs[0, 2])
    top_k_idx = np.argsort(eng_imp)[-10:]
    ax_eng.barh([eng_names[i] for i in top_k_idx], eng_imp[top_k_idx], color="#2b5c8f", edgecolor="black")
    ax_eng.set_title("Istotność cech domenowych (Random Forest)", fontsize=10.5, fontweight="bold")
    ax_eng.set_xlabel("Gini Feature Importance")

    # Bottom row: 3 Grad-CAM examples on Pneumonia test cases
    pneu_test_indices = np.where((y_test == 1) & (test_probs["PneumoniaCNN (PyTorch)"] > 0.90))[0][:3]
    for col_i, t_idx in enumerate(pneu_test_indices):
        ax_cam = fig.add_subplot(gs[1, col_i])
        x_ten, _ = test_ds[t_idx]
        cam_map, prob_pneu = compute_gradcam(model, x_ten.unsqueeze(0))
        raw_im = test_ds.cached_imgs[t_idx]
        ax_cam.imshow(raw_im, cmap="gray")
        ax_cam.imshow(cam_map, cmap="jet", alpha=0.42)
        sub_name = df.iloc[idx_test[t_idx]]["subtype"].replace("PNEUMONIA_", "")
        ax_cam.set_title(f"Grad-CAM CNN: {sub_name}\nP(PNEUMONIA) = {prob_pneu*100:.1f}%", fontsize=10.5, fontweight="bold")
        ax_cam.axis("off")

    plt.suptitle("Wyjaśnialność modeli (XAI): Istotność cech przestrzennych oraz mapy aktywacji Grad-CAM", fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_explainability_gradcam_importance.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Fig 9: Error Analysis (False Positives & False Negatives + Subtype Error Breakdown)
    # ---------------------------------------------------------
    cnn_opt_t = results_summary["PneumoniaCNN (PyTorch)"]["val_threshold_platt"]
    cnn_tp_platt = test_probs_platt["PneumoniaCNN (PyTorch)"]
    cnn_preds = (cnn_tp_platt >= cnn_opt_t).astype(int)

    fp_indices = np.where((y_test == 0) & (cnn_preds == 1))[0]
    fn_indices = np.where((y_test == 1) & (cnn_preds == 0))[0]

    # Sort FPs by highest predicted probability, FNs by lowest predicted probability
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
            sub = df.iloc[idx_test[ti]]["subtype"].replace("PNEUMONIA_", "")
            ax_fn.imshow(test_ds.cached_imgs[ti], cmap="gray")
            ax_fn.set_title(
                f"Błąd FN #{i+1} (Rzecz.: {sub})\nPred: NORMAL (P={cnn_tp_platt[ti]*100:.1f}%)",
                fontsize=9.5,
                color="#2b5c8f",
                fontweight="bold",
            )
        ax_fn.axis("off")

    # Subtype recall comparison on test set (Normal vs Bacteria vs Virus)
    ax_sub = fig.add_subplot(gs[4])
    test_subtypes = df.iloc[idx_test]["subtype"].values
    sub_acc_data = []
    for sub_key, sub_label in [("NORMAL", "Zdrowe\n(NORMAL)"), ("PNEUMONIA_BACTERIA", "Bakteryjne\n(BACTERIA)"), ("PNEUMONIA_VIRUS", "Wirusowe\n(VIRUS)")]:
        mask_s = (test_subtypes == sub_key)
        correct_rate = (cnn_preds[mask_s] == y_test[mask_s]).mean() * 100.0
        sub_acc_data.append((sub_label, correct_rate, int(mask_s.sum())))

    bars = ax_sub.bar(
        [x[0] for x in sub_acc_data],
        [x[1] for x in sub_acc_data],
        color=["#2b5c8f", "#c0392b", "#e67e22"],
        edgecolor="black", width=0.58
    )
    ax_sub.set_ylim(0, 108)
    ax_sub.set_ylabel("Dokładność klasyfikacji [%]")
    ax_sub.set_title(
        "Dokładność CNN wg podtypu (próg z walidacji)\nna zbiorze testowym",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_sub.bar_label(bars, fmt="%.1f%%", fontsize=9.5, padding=2)

    plt.suptitle("Analiza błędów (Error Analysis): Przykłady FP i FN oraz skuteczność względem etiologii", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig9_error_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # Save complete summary JSON
    summary_payload = {
        "dataset_counts": {
            "total": int(len(df)),
            "orig_splits": df.groupby(["orig_split", "label_name"]).size().unstack(fill_value=0).to_dict(),
            "corrected_splits": df.groupby(["split", "label_name"]).size().unstack(fill_value=0).to_dict(),
            "subtypes": df["subtype"].value_counts().to_dict(),
            "width_mean": float(df["width"].mean()),
            "width_std": float(df["width"].std()),
            "height_mean": float(df["height"].mean()),
            "height_std": float(df["height"].std()),
            "intensity_normal_mean": float(df[df["label"] == 0]["mean_intensity"].mean()),
            "intensity_pneumonia_mean": float(df[df["label"] == 1]["mean_intensity"].mean()),
            "std_normal_mean": float(df[df["label"] == 0]["std_intensity"].mean()),
            "std_pneumonia_mean": float(df[df["label"] == 1]["std_intensity"].mean()),
        },
        "leakage_audit": leakage_audit,
        "val_thresholds_raw": val_thresholds,
        "val_thresholds_platt": val_thresholds_platt,
        "gui_model_thresholds_platt": gui_val_thresholds_platt,
        "pca_explained_variance_100": float(cum_var[-1]),
        "best_params_summary": best_params_summary,
        "results_summary": results_summary,
        "bootstrap_paired_auc_comparisons_test": paired_comparisons,
        "test_predictions_path": str(TEST_PREDICTIONS_PATH.relative_to(PROJECT_DIR)),
        "subtype_test_accuracy_cnn": {
            x[0].replace("\n", " "): {"accuracy_pct": float(x[1]), "count": x[2]} for x in sub_acc_data
        },
        "evaluation_notes": {
            "threshold_selection": "Validation Youden J only; test labels never used for threshold tuning.",
            "probability_display": "GUI/report Platt scaling fit on validation split.",
        },
        "elapsed_seconds": float(time.time() - t0),
    }
    with open(METRICS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Pipeline completed in {time.time() - t0:.1f}s! Summary saved to {METRICS_SUMMARY_PATH}.")


if __name__ == "__main__":
    main()

