# Data

The official UCI HAR Smartphones dataset 240 is downloaded locally, not committed. Copyright and licensing follow its authors: distributed under CC BY 4.0 with required attribution, https://doi.org/10.24432/C54S4K.

From the repository root run `python scripts/download_dataset.py` (public internet needed), or manually extract the archive to `data/UCI HAR Dataset/`. Required files: `features.txt`, `activity_labels.txt`, `train/X_train.txt`, `train/y_train.txt`, `train/subject_train.txt`, and their `test/` analogues. `features_info.txt` explains the 561 derived measurements. Raw windows in `Inertial Signals/` are not processed in the benchmark. The code validates file existence, dimensions, subject partition disjointness, class ids, and SHA256. The original test rows are held out from training.

CSV: configure `kind: csv` and explicit target, split, subject/time fields in `configs/dataset.yaml`.
