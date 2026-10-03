from pathlib import Path
from PIL import Image
import numpy as np

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")

def main():
    print("=== DATASET VERIFICATION ===")
    
    # 1. Check data/combined/train vs data/combined_hm/train
    c_img_dir = BASE / "data/combined/train/images"
    c_lbl_dir = BASE / "data/combined/train/labels"
    hm_img_dir = BASE / "data/combined_hm/train/images"
    hm_lbl_dir = BASE / "data/combined_hm/train/labels"
    
    c_imgs = list(c_img_dir.glob("*.png"))
    hm_imgs = list(hm_img_dir.glob("*.png"))
    print(f"data/combined/train/images: {len(c_imgs)}")
    print(f"data/combined_hm/train/images: {len(hm_imgs)}")
    
    # Check if image filenames match
    c_names = set(f.name for f in c_imgs)
    hm_names = set(f.name for f in hm_imgs)
    print(f"Filenames match: {c_names == hm_names}")
    
    # Check if pixel values differ between combined (raw) and combined_hm (HM)
    sample_name = list(c_names)[0]
    img_raw = np.array(Image.open(c_img_dir / sample_name))
    img_hm = np.array(Image.open(hm_img_dir / sample_name))
    diff = np.abs(img_raw.astype(float) - img_hm.astype(float))
    print(f"Sample: {sample_name}")
    print(f"  Raw mean: {img_raw.mean():.2f}, std: {img_raw.std():.2f}")
    print(f"  HM mean:  {img_hm.mean():.2f}, std: {img_hm.std():.2f}")
    print(f"  Mean abs pixel difference: {diff.mean():.2f}")
    
    # Count labels & bounding boxes in data/combined/train/labels
    c_lbls = list(c_lbl_dir.glob("*.txt"))
    hm_lbls = list(hm_lbl_dir.glob("*.txt"))
    print(f"\ndata/combined/train/labels files: {len(c_lbls)}")
    print(f"data/combined_hm/train/labels files: {len(hm_lbls)}")
    
    c_boxes = sum(len(open(f).readlines()) for f in c_lbls)
    hm_boxes = sum(len(open(f).readlines()) for f in hm_lbls)
    print(f"data/combined/train total bounding boxes: {c_boxes}")
    print(f"data/combined_hm/train total bounding boxes: {hm_boxes}")

    # Check validation and test raw sources
    val_raw_dir = BASE / "data/prieur/moon_only/validation/images"
    val_raw_lbl = BASE / "data/prieur/moon_only/validation/labels"
    test_raw_dir = BASE / "data/prieur/moon_only/test/images"
    test_raw_lbl = BASE / "data/prieur/moon_only/test/labels"
    
    val_imgs = list(val_raw_dir.glob("*.png"))
    val_boxes = sum(len(open(f).readlines()) for f in val_raw_lbl.glob("*.txt"))
    test_imgs = list(test_raw_dir.glob("*.png"))
    test_boxes = sum(len(open(f).readlines()) for f in test_raw_lbl.glob("*.txt"))
    
    print(f"\nValidation Raw: {len(val_imgs)} images, {val_boxes} bounding boxes")
    print(f"Test Raw: {len(test_imgs)} images, {test_boxes} bounding boxes")

if __name__ == '__main__':
    main()
