import shutil
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
IMAGES_DIR  = Path('data/yolo_RW_dataset2/images')    # your existing flat images folder
LABELS_DIR  = Path('data/yolo_RW_dataset2/labels')    # your existing flat labels folder
OUTPUT_DIR  = Path('data/yolo_RW_dataset2')
VAL_SPLIT   = 0.15
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

all_images = list(IMAGES_DIR.glob('*.jpg')) + list(IMAGES_DIR.glob('*.png'))
random.shuffle(all_images)

n_val   = int(len(all_images) * VAL_SPLIT)
val_set = all_images[:n_val]
trn_set = all_images[n_val:]

print(f'Train: {len(trn_set)} | Val: {len(val_set)}')

for split, files in [('train', trn_set), ('val', val_set)]:
    img_out = OUTPUT_DIR / 'images' / split
    lbl_out = OUTPUT_DIR / 'labels' / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path in files:
        shutil.copy(img_path, img_out / img_path.name)

        lbl_path = LABELS_DIR / f'{img_path.stem}.txt'
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_out / lbl_path.name)
        else:
            print(f'  WARNING: no label for {img_path.name}')

print('Done.')