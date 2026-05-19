from ultralytics import YOLO

model = YOLO('/home/nicola/ros2_ws/src/yolo_inference/weights/best.pt')  # your existing checkpoint

model.train(
    data='/home/nicola/ros2_ws/data/yolo_RW_dataset/dataset.yaml',
    epochs=200,
    imgsz=640,
    task='segment',
    batch=8,           # adjust based on your GPU VRAM
    patience=30,       # early stopping
    device=0,          # GPU, or 'cpu'
    
    # Augmentation — critical for synthetic→real gap
    hsv_h=0.02,        # hue shift (lighting differences)
    hsv_s=0.7,         # saturation (synthetic colors are often too clean)
    hsv_v=0.5,         # brightness variation
    degrees=10,        # small rotations
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,         # blends real images together

    # Freezing — fine-tune only the head first
    freeze=10,         # freeze first 10 backbone layers

    lr0=0.001,         # lower LR since we're fine-tuning
    lrf=0.01,
    warmup_epochs=3,
    
    
    project='runs',
    name='tray_rack_v2',
)