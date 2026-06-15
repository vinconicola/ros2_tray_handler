import json

with open('/home/nicola/ros2_ws/src/training/RW_COCO_dataset2/result_coco.json', 'r') as f:
    data = json.load(f)

# Remap: rack(1)->1, tray(2)->0 in YOLO 0-indexed
id_remap = {}
for cat in data['categories']:
    if cat['name'] == 'tray':
        id_remap[cat['id']] = 0
    elif cat['name'] == 'rack':
        id_remap[cat['id']] = 1

# Reorder categories list so tray is first (index 0)
data['categories'] = [
    {'id': 1, 'name': 'tray', 'supercategory': ''},
    {'id': 2, 'name': 'rack', 'supercategory': ''},
]

# Remap annotation category_ids
for ann in data['annotations']:
    original = ann['category_id']
    # tray was id=2, rack was id=1 — swap them
    ann['category_id'] = 1 if original == 2 else 2

with open('result_coco_fixed.json', 'w') as f:
    json.dump(data, f)

print('Done — tray=0, rack=1 after YOLO conversion')