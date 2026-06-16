from ultralytics import YOLO

model = YOLO('src/yolo_inference/weights/best.pt')  # your existing checkpoint

model.train(
    data='data/yolo_RW_dataset2/dataset.yaml',
    epochs=100,          # early stopping will handle the rest
    imgsz=640,
    task='segment',
    batch=8,
    patience=20,         # tighter — 125 imgs epochs are fast
    device=0,

    # Augmentation
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    mosaic=0.7,          # slightly reduced for small dataset
    mixup=0.1,

    freeze=5,            # unfreeze more layers for domain shift

    lr0=0.0005,          # even lower LR for gentle fine-tuning
    lrf=0.01,
    warmup_epochs=2,

    project='runs',
    name='tray_rack_v3',
)