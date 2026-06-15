from pathlib import Path
import cv2

for split in ['train', 'val']:
    img_dir = Path(f'/home/nicola/ros2_ws/data/yolo_RW_dataset/images/{split}')
    for img_path in img_dir.glob('*'):
        img = cv2.imread(str(img_path))
        if img is not None:
            flipped = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(str(img_path), flipped)
    print(f'{split} done')

for split in ['train', 'val']:
    lbl_dir = Path(f'/home/nicola/ros2_ws/data/yolo_RW_dataset/labels/{split}')
    for lbl_path in lbl_dir.glob('*.txt'):
        lines = lbl_path.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            vals = line.split()
            cls = vals[0]
            coords = list(map(float, vals[1:]))
            # flip each x,y pair: x→1-x, y→1-y
            flipped = [1.0 - v for v in coords]
            new_lines.append(cls + ' ' + ' '.join(f'{v:.6f}' for v in flipped))
        lbl_path.write_text('\n'.join(new_lines))
    print(f'{split} labels done')