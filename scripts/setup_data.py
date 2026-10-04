import os
import random
import pathlib
import subprocess
import yaml

ROOT_DIR = pathlib.Path(".").resolve()
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_IMAGES = ROOT_DIR / "MH-Weed16" / "Crop with Weeds" / "intel Real Sense Depth_Clicks" / "intel Real Sense Depth_Clicks"
RAW_LABELS = ROOT_DIR / "MH-Weed16" / "Crop with Weeds" / "intel Real Sense Depth_Annotations" / "intel Real Sense Depth_Annotations" / "YOLO_darknet"

IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

# Setup cross-platform junctions / symlinks
if not IMAGES_DIR.exists():
    if os.name == 'nt':
        cmd = f'cmd /c mklink /J "{IMAGES_DIR}" "{RAW_IMAGES}"'
        subprocess.run(cmd, shell=True, check=True)
    else:
        IMAGES_DIR.symlink_to(RAW_IMAGES)
    print("Created images junction/symlink.")

if not LABELS_DIR.exists():
    if os.name == 'nt':
        cmd = f'cmd /c mklink /J "{LABELS_DIR}" "{RAW_LABELS}"'
        subprocess.run(cmd, shell=True, check=True)
    else:
        LABELS_DIR.symlink_to(RAW_LABELS)
    print("Created labels junction/symlink.")

# Collect image paths via data/images (do NOT call .resolve() so images/ in path is preserved)
image_files = sorted([p for p in IMAGES_DIR.glob('*.*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
all_paths = [os.path.normpath(str(p)) for p in image_files]
print(f"Total images found: {len(all_paths):,}")

# Deterministic split 80/10/10
random.seed(42)
shuffled = list(all_paths)
random.shuffle(shuffled)

n = len(shuffled)
n_train = int(0.80 * n)
n_val = int(0.10 * n)

train_imgs = shuffled[:n_train]
val_imgs = shuffled[n_train:n_train + n_val]
test_imgs = shuffled[n_train + n_val:]

print(f"Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

def write_txt(path, paths):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(paths) + '\n')

write_txt(ROOT_DIR / "train.txt", train_imgs)
write_txt(ROOT_DIR / "val.txt", val_imgs)
write_txt(ROOT_DIR / "test.txt", test_imgs)

# Class names mapping
CLASS_NAMES = {
    0: "kena",
    1: "lavhala",
    2: "lambs_quarters",
    3: "little_mallow",
    4: "moti_dudhi",
    5: "obscure_morning_glory",
    6: "asian_pigeonwings",
    7: "bilayat",
    8: "choti_dudhi",
    9: "digitaria",
    10: "gajar_gavat",
    11: "graceful_sandmat",
    12: "sicklepod",
    13: "harali",
    14: "dwarf_cassia"
}

yaml_data = {
    'path': str(ROOT_DIR).replace('\\', '/'),
    'train': 'train.txt',
    'val': 'val.txt',
    'test': 'test.txt',
    'names': CLASS_NAMES
}

with open(ROOT_DIR / "mhweed16.yaml", 'w', encoding='utf-8') as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

print("Updated mhweed16.yaml successfully.")
