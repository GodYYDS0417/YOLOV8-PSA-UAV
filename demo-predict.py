from ultralytics import YOLO
yolo=YOLO('./yolov8n.pt', task='detect')
results = yolo('./ultralytics/assets/bus.jpg')