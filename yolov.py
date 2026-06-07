from ultralytics import YOLO
model=YOLO('yolov8n.pt')
model.train(data='yolo-cvn.yaml',workers=0,epochs=200,batch=16)