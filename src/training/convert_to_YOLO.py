from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir='src/training/RW_COCO_dataset2/',
    use_segments=True,
    use_keypoints=False
)