"""Shared, portable project configuration.

All paths are derived from this file's location.  Nothing here depends on a
developer's home directory or on the directory from which a command is run.
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).parent

DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
CACHE_DIR = PROJECT_DIR / ".cache"
MATPLOTLIB_CACHE_DIR = CACHE_DIR / "matplotlib"

KAGGLE_DATASET_ID = "paultimothymooney/chest-xray-pneumonia"
KAGGLE_DATASET_CACHE_DIR = DATA_DIR / "datasets" / "paultimothymooney" / "chest-xray-pneumonia"

CLASSICAL_MODELS_PATH = MODELS_DIR / "classical_models.joblib"
CNN_MODEL_PATH = MODELS_DIR / "cnn_pneumonia.pt"
DATASET_METADATA_PATH = MODELS_DIR / "dataset_metadata.csv"
METRICS_SUMMARY_PATH = MODELS_DIR / "metrics_summary.json"

REPORT_TEMPLATE_PATH = PROJECT_DIR / "Raport Końcowy Przykład.docx"
REPORT_PATH = PROJECT_DIR / "Raport_Koncowy_PneumoniaMNIST.docx"
REPORT_AUTHORS_PATH = PROJECT_DIR / "Raport Końcowy - Kamil Rayski, Miłosz Pabis.docx"
NOTEBOOK_PATH = PROJECT_DIR / "PneumoniaMNIST_Analiza_i_Modelowanie.ipynb"


def ensure_runtime_dirs() -> None:
    """Create directories used for downloaded and generated runtime files."""
    for directory in (DATA_DIR, FIGURES_DIR, MODELS_DIR, MATPLOTLIB_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path stored relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate

