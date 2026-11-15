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
TEST_PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"

REPORT_DIR = PROJECT_DIR / "report"
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "PneumoniaMNIST_Analiza_i_Modelowanie.ipynb"
README_PATH = PROJECT_DIR / "README.md"

# Primary deliverables (project root — where submission copies are expected)
REPORT_PATH = REPORT_DIR / "Raport Końcowy - Kamil Rayski, Miłosz Pabis.docx"

_TEMPLATE_CANDIDATES = (
    REPORT_DIR / "Raport Końcowy Przykład.docx",
)


def resolve_report_template() -> Path:
    """Locate the DOCX template (report/ or legacy project root)."""
    for candidate in _TEMPLATE_CANDIDATES:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(p) for p in _TEMPLATE_CANDIDATES)
    raise FileNotFoundError(
        f"Brak szablonu raportu DOCX. Oczekiwany m.in.: {REPORT_DIR / 'Raport Końcowy Przykład.docx'}. "
        f"Sprawdzone: {tried}"
    )


def ensure_runtime_dirs() -> None:
    """Create directories used for downloaded and generated runtime files."""
    for directory in (DATA_DIR, FIGURES_DIR, MODELS_DIR, MATPLOTLIB_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path stored relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate

