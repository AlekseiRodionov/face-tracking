from ultralytics import YOLO

model = YOLO("yolo11n.pt")

for k, v in model.named_parameters():
    print(k)

train_result = model.train(
    data='wider_face.yaml',
    imgsz=640,
    time=9,
    freeze=15,
)

model.save('my_yolo11n.pt')
model.export(format='onnx')
