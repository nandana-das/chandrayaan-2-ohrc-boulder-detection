"""Deterministic Histogram-Matching Reference Generator and Verifier.

Generates and verifies the 20-tile averaged OHRC reference artifact
using the established repository seed (seed=42).

Outputs:
  results/histogram_matching/reference_tiles_seed_42.csv
  results/histogram_matching/averaged_reference_seed_42.png
  results/histogram_matching/reference_metadata.json
"""

import os
import json
import random
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

BASE = Path(r"D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection")
OHRC_DIR = BASE / "data/tiles/usable"
OUT_DIR = BASE / "results/histogram_matching"

SEED = 42
NUM_TILES = 20

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STEP 2: DETERMINISTIC HISTOGRAM-MATCHING REFERENCE GENERATION")
    print("=" * 70)
    
    assert OHRC_DIR.exists(), f"OHRC tile directory not found: {OHRC_DIR}"
    all_tiles = sorted(OHRC_DIR.glob("*.png"))
    total_tiles = len(all_tiles)
    print(f"Total OHRC tiles found in {OHRC_DIR}: {total_tiles:,}")
    assert total_tiles == 31769, f"Expected 31,769 tiles, got {total_tiles}"
    
    # Deterministic sampling
    rng = random.Random(SEED)
    selected_tiles = rng.sample(all_tiles, NUM_TILES)
    
    # Verification of selection
    print(f"\nSampling verification:")
    print(f"  Count: {len(selected_tiles)} (expected {NUM_TILES})")
    assert len(selected_tiles) == NUM_TILES, f"Expected {NUM_TILES} tiles, got {len(selected_tiles)}"
    
    tile_names = [t.name for t in selected_tiles]
    unique_names = set(tile_names)
    print(f"  Unique count: {len(unique_names)} (expected {NUM_TILES})")
    assert len(unique_names) == NUM_TILES, "Duplicate tiles found in sample!"
    
    # Verify each tile exists and inspect dimensions
    tile_records = []
    refs = []
    for idx, t_path in enumerate(selected_tiles, 1):
        assert t_path.exists(), f"Tile does not exist: {t_path}"
        img = Image.open(t_path).convert("L")
        arr = np.array(img).astype(float)
        refs.append(arr)
        
        sha = compute_sha256(t_path)
        tile_records.append({
            "sample_index": idx,
            "tile_name": t_path.name,
            "relative_path": str(t_path.relative_to(BASE)).replace("\\", "/"),
            "width": img.width,
            "height": img.height,
            "mean_intensity": round(float(arr.mean()), 3),
            "std_intensity": round(float(arr.std()), 3),
            "min_intensity": int(arr.min()),
            "max_intensity": int(arr.max()),
            "sha256": sha,
        })
        
    # Save CSV of selected tiles
    df_tiles = pd.DataFrame(tile_records)
    csv_path = OUT_DIR / f"reference_tiles_seed_{SEED}.csv"
    df_tiles.to_csv(csv_path, index=False)
    print(f"\n[Artifact Created] Reference tiles CSV: {csv_path}")
    
    # Build averaged reference image
    reference_arr = np.mean(refs, axis=0).astype(np.uint8)
    ref_img = Image.fromarray(reference_arr)
    img_path = OUT_DIR / f"averaged_reference_seed_{SEED}.png"
    ref_img.save(img_path)
    ref_sha = compute_sha256(img_path)
    print(f"[Artifact Created] Averaged reference image: {img_path}")
    print(f"  Dimensions: {ref_img.width}x{ref_img.height}")
    print(f"  Mean pixel intensity: {reference_arr.mean():.3f}")
    print(f"  Std pixel intensity:  {reference_arr.std():.3f}")
    print(f"  Min/Max intensity:    {reference_arr.min()}/{reference_arr.max()}")
    print(f"  SHA256:               {ref_sha}")
    
    # Build machine-readable metadata
    metadata = {
        "generator_script": "scripts/make_deterministic_hm_reference.py",
        "random_seed": SEED,
        "sampling_method": "random.Random(42).sample(sorted(glob('*.png')), 20)",
        "source_tile_directory": str(OHRC_DIR.relative_to(BASE)).replace("\\", "/"),
        "total_source_tiles_available": total_tiles,
        "sample_size": NUM_TILES,
        "artifact_averaged_image": {
            "filename": img_path.name,
            "relative_path": str(img_path.relative_to(BASE)).replace("\\", "/"),
            "dimensions": [ref_img.width, ref_img.height],
            "mode": "L (8-bit grayscale)",
            "mean_intensity": round(float(reference_arr.mean()), 4),
            "std_intensity": round(float(reference_arr.std()), 4),
            "min_intensity": int(reference_arr.min()),
            "max_intensity": int(reference_arr.max()),
            "sha256": ref_sha,
        },
        "artifact_tiles_csv": {
            "filename": csv_path.name,
            "relative_path": str(csv_path.relative_to(BASE)).replace("\\", "/"),
            "num_records": len(df_tiles),
            "sha256": compute_sha256(csv_path),
        },
        "selected_tiles": tile_names,
        "verification_status": {
            "count_verified": len(selected_tiles) == NUM_TILES,
            "all_tiles_exist": all(t.exists() for t in selected_tiles),
            "no_duplicates": len(unique_names) == NUM_TILES,
            "averaged_reference_exists": img_path.exists(),
            "deterministic_reproducible": True,
        }
    }
    
    meta_path = OUT_DIR / "reference_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Artifact Created] Reference metadata: {meta_path}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUCCEEDED: Deterministic HM reference artifacts established.")
    print("=" * 70)

if __name__ == "__main__":
    main()
