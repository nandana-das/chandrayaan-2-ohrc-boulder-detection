"""Construct data/combined_nohm dataset directory using exact hard-links to raw source files.
Guarantees identical train (4,379), val (697), and test (262) splits with exactly identical labels.
"""
import os
import shutil
from pathlib import Path
import yaml

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
DATA_DIR = BASE / "data"

OUT = DATA_DIR / "combined_nohm"

def count_boxes(lbl_dir):
    total = 0
    for f in lbl_dir.glob("*.txt"):
        with open(f) as fp:
            total += len(fp.readlines())
    return total

def main():
    print("=" * 60)
    print("Creating data/combined_nohm with exact raw source splits")
    print("=" * 60)
    
    # 1. Create directories
    for split in ['train', 'val', 'test']:
        (OUT / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUT / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 2. Source mapping
    sources = {
        'train': {
            'images': DATA_DIR / 'combined/train/images',
            'labels': DATA_DIR / 'combined/train/labels',
        },
        'val': {
            'images': DATA_DIR / 'prieur/moon_only/validation/images',
            'labels': DATA_DIR / 'prieur/moon_only/validation/labels',
        },
        'test': {
            'images': DATA_DIR / 'prieur/moon_only/test/images',
            'labels': DATA_DIR / 'prieur/moon_only/test/labels',
        },
    }

    # 3. Hard-link images and labels
    for split, paths in sources.items():
        dst_img_dir = OUT / split / 'images'
        dst_lbl_dir = OUT / split / 'labels'
        
        src_imgs = sorted(list(paths['images'].glob("*.png")))
        src_lbls = sorted(list(paths['labels'].glob("*.txt")))
        
        print(f"\nLinking {split}: {len(src_imgs)} images, {len(src_lbls)} labels...")
        for img in src_imgs:
            dst = dst_img_dir / img.name
            if not dst.exists():
                os.link(img, dst)
                
        for lbl in src_lbls:
            dst = dst_lbl_dir / lbl.name
            if not dst.exists():
                os.link(lbl, dst)

    # 4. Create dataset_combined_nohm.yaml
    yaml_content = {
        'path': str(OUT).replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['boulder'],
    }
    yaml_path = OUT / "dataset_combined_nohm.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"\nCreated YAML config: {yaml_path}")

    # 5. Verification
    print("\n--- VERIFICATION ---")
    tr_imgs = len(list((OUT / 'train/images').glob('*.png')))
    val_imgs = len(list((OUT / 'val/images').glob('*.png')))
    tst_imgs = len(list((OUT / 'test/images').glob('*.png')))
    
    tr_boxes = count_boxes(OUT / 'train/labels')
    val_boxes = count_boxes(OUT / 'val/labels')
    tst_boxes = count_boxes(OUT / 'test/labels')
    
    print(f"Train: {tr_imgs} images (Expected 4,379), {tr_boxes} boxes (Expected 122,537)")
    print(f"Val:   {val_imgs} images (Expected 697),   {val_boxes} boxes (Expected 24,303)")
    print(f"Test:  {tst_imgs} images (Expected 262),   {tst_boxes} boxes (Expected 7,268)")
    
    assert tr_imgs == 4379, f"Train count mismatch: {tr_imgs}"
    assert val_imgs == 697, f"Val count mismatch: {val_imgs}"
    assert tst_imgs == 262, f"Test count mismatch: {tst_imgs}"
    assert tr_boxes == 122537, f"Train box mismatch: {tr_boxes}"
    assert val_boxes == 24303, f"Val box mismatch: {val_boxes}"
    assert tst_boxes == 7268, f"Test box mismatch: {tst_boxes}"
    print("\n[SUCCESS] All split and instance counts match exactly!")

if __name__ == '__main__':
    main()
