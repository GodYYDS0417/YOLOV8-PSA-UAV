from ultralytics import YOLO

for tag, w in {
    "yolov8": "./runs/detect/原/weights/best.pt",
    "yolov8_psa": "./runs/detect/PSA/weights/best.pt"
}.items():
    m = YOLO(w)
    print(f"\n===== {tag} =====")
    for i, layer in enumerate(m.model.model):
        print(i, layer.__class__.__name__)