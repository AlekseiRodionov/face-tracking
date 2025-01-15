import cv2
from ultralytics import YOLO

model = YOLO("my_yolo11n.pt")
camera = cv2.VideoCapture(0)
while True:
    return_value, image = camera.read()
    image = cv2.flip(image, 1)
    center = 0.5
    result = model.track(image, verbose=False, show=False, conf=0.4, iou=0.5)
    objects_distance_and_shape = []
    for i in range(len(result[0].boxes.cls)):
        coords = list(result[0].boxes.xywhn[i])
        x1 = coords[0] - coords[2] / 2
        x2 = coords[0] + coords[2] / 2
        y1 = coords[1] - coords[3] / 2
        y2 = coords[1] + coords[3] / 2
        if (center > x1 and center < x2) and (center > y1 and center < y2):
            print(f"Объект {i} в центре изображения")
        objects_distance_and_shape.append((i, ((coords[0] - center)**2 + (coords[1] - center)**2)**0.5, (x2-x1)*(y2-y1)))
        print(f"Расстояние до объекта {i} = {objects_distance_and_shape[i][1]}")
    if objects_distance_and_shape:
        print(f"Самый большой объект: {max(objects_distance_and_shape, key=lambda x: x[2])[0]}")
    cv2.imshow('video', result[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
del camera