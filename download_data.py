import os

from config import DATA_DIR, KAGGLE_DATASET_ID, ensure_runtime_dirs

ensure_runtime_dirs()
os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)

import kagglehub

# Download latest version
path = kagglehub.dataset_download(KAGGLE_DATASET_ID)

print("Path to dataset files:", path)

