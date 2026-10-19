"""Copy the SAM questionnaire files out of the WESAD download.

The regression notebook (03_Regression.ipynb) reads the self-reported arousal scores from a
flat folder `WESAD_quest/` containing one `S<id>_quest.csv` per subject. WESAD ships them
inside each subject folder (WESAD/S2/S2_quest.csv, ...), so this script collects them.

Usage (from the repository root):
    python scripts/prepare_quest.py
"""
import os
import shutil

WESAD_DIR = "WESAD"
OUT_DIR = "WESAD_quest"
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]   # S1 and S12 do not exist in WESAD

os.makedirs(OUT_DIR, exist_ok=True)
missing = []
for sid in SUBJECTS:
    src = os.path.join(WESAD_DIR, f"S{sid}", f"S{sid}_quest.csv")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT_DIR, f"S{sid}_quest.csv"))
    else:
        missing.append(src)

print(f"Copied {len(SUBJECTS) - len(missing)} questionnaire files to {OUT_DIR}/")
if missing:
    print("Missing files:")
    for m in missing:
        print("  ", m)
