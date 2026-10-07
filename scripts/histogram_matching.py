import numpy as np
from PIL import Image
from pathlib import Path
from skimage.exposure import match_histograms
import random

BASE = Path(r'D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection')
LROC_DIR = BASE / 'data/rmam/moon/train/images'
OHRC_DIR = BASE / 'data/tiles/usable'
OUT_DIR  = BASE / 'data/rmam/moon/train_hm_multi/images'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the same deterministic 20-tile OHRC reference set across train/val/test.
ref_artifact = BASE / "results/histogram_matching/averaged_reference_seed_42.png"
if ref_artifact.exists():
    reference = np.array(Image.open(ref_artifact).convert('L'))
else:
    ohrc_tiles = sorted(OHRC_DIR.glob('*.png'))
    rng = random.Random(42)
    sample = rng.sample(ohrc_tiles, 20)
    refs = [np.array(Image.open(f).convert('L')).astype(float) for f in sample]
    reference = np.mean(refs, axis=0).astype(np.uint8)

# Match all LROC train images to the same fixed reference.
lroc_imgs = list(LROC_DIR.glob('*.tif'))
for i, img_path in enumerate(lroc_imgs):
    img = np.array(Image.open(img_path).convert('L'))
    matched = match_histograms(img, reference).astype(np.uint8)
    Image.fromarray(matched).save(OUT_DIR / (img_path.stem + '.png'))
    if i % 50 == 0:
        print(f'{i}/{len(lroc_imgs)} done')

print(f'Done — {len(lroc_imgs)} images matched to deterministic averaged OHRC reference (n=20, seed=42)')
