import sys
from ultralytics import YOLO
from pathlib import Path

def main():
    base = Path(r'D:\Nandana\MTECH\Semester 3\Projects\TP\lunar-landslide-boulder-detection')
    m = YOLO(str(base / 'runs/stage2_yolo26n_combined_hm_cosine/weights/best.pt'))
    val_res = m.val(
        data=str(base / 'data/combined_hm/dataset_combined_hm.yaml'), 
        split='test', 
        imgsz=640, 
        device=0, 
        workers=0,
        verbose=False
    )
    print('dir(val_res.box):', [a for a in dir(val_res.box) if not a.startswith('_')])
    for attr in ['px', 'py', 'p', 'r', 'f1', 'curves', 'curves_results']:
        if hasattr(val_res.box, attr):
            val = getattr(val_res.box, attr)
            shape = getattr(val, 'shape', len(val) if hasattr(val, '__len__') else type(val))
            print(f'{attr}: {shape}')
    if hasattr(val_res.box, 'curves_results'):
        for i, cr in enumerate(val_res.box.curves_results):
            print(f'curves_results[{i}]: len={len(cr)}')
            for j, item in enumerate(cr):
                shp = getattr(item, 'shape', type(item))
                print(f'  item[{j}]: {shp}')

if __name__ == '__main__':
    main()
