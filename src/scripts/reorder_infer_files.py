import os
import shutil
from pathlib import Path

path = "/data/afatehi/gbm/experiments"
inlcude = "results-infer"
exclude = "16464_6"
filelist = []
dist = '/data/afatehi/gbm/mouse-results/'

for root, dirs, files in os.walk(path):
    for file in files:
        if inlcude in root and exclude not in root and ('.gif' in file or '.tif' in file):
            filelist.append(os.path.join(root, file))


for file in filelist:
    file_path = Path(file)
    sample_name = file_path.parent.stem

    sample_path = Path(os.path.join(dist, sample_name))
    sample_path.mkdir(parents=True, exist_ok=True)

    model_name = file_path.parent.parent.parent.parent.stem

    dist_path = os.path.join(sample_path, f"{model_name}{file_path.suffix}")
    shutil.copy2(file_path, dist_path)
    print(file_path)
    print(dist_path)
    print('-------')
