"""Apply deterministic averaged-reference histogram matching to Prieur lunar test set."""
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.exposure import match_histograms
import random, shutil

PRIEUR_TEST = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\prieur\moon_only\test")
OHRC = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\tiles\usable")
OUT_IMG = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\combined_hm\test\images")
OUT_LBL = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection\data\combined_hm\test\labels")

if __name__ == '__main__':
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

    # Keep the reference identical to train/val: sorted tiles + fixed seed.
    ohrc_tiles = sorted(OHRC.glob('*.png'))
    rng = random.Random(42)
    sample = rng.sample(ohrc_tiles, 20)
    refs = [np.array(Image.open(f).convert('L')).astype(float) for f in sample]
    reference = np.mean(refs, axis=0).astype(np.uint8)

    test_imgs = list((PRIEUR_TEST / 'images').glob('*.png'))
    for i, f in enumerate(test_imgs):
        img = np.array(Image.open(f).convert('L'))
        matched = match_histograms(img, reference).astype(np.uint8)
        Image.fromarray(matched).save(OUT_IMG / f.name)
        lbl = PRIEUR_TEST / 'labels' / (f.stem + '.txt')
        if lbl.exists():
            shutil.copy(lbl, OUT_LBL / (f.stem + '.txt'))
        if i % 50 == 0:
            print(f'{i}/{len(test_imgs)}')
    print(f'Done — {len(test_imgs)} test images HM applied using fixed averaged OHRC reference (n=20, seed=42)')
