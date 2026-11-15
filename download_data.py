import os

os.environ["KAGGLEHUB_CACHE"] = "/Users/kamil/Desktop/STUDIA/Magisterka/SEM2/UM/PROJEKT1/data"

import kagglehub

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)

