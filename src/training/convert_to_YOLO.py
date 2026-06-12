from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir='/home/nicola/ros2_ws/src/training/RW_COCO_dataset2/',
    use_segments=True,
    use_keypoints=False
)