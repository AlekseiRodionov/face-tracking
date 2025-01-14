import cv2
from ultralytics import YOLO

model = YOLO("my_yolo11n.pt")
camera = cv2.VideoCapture(0)
while True:
    return_value, image = camera.read()
    image = cv2.flip(image, 1)
    result = model.track(image, show=False, conf=0.4, iou=0.5)
    cv2.imshow('video', result[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
del camera