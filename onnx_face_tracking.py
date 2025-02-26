import cv2
import onnxruntime as ort
import numpy as np
from time import time

# Подробности: https://dev.to/andreygermanov/how-to-create-yolov8-based-object-detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e#explore

def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

model = ort.InferenceSession('my_yolo11n.onnx')
inputs = model.get_inputs()
input = inputs[0]

outputs = model.get_outputs()
output = outputs[0]

camera = cv2.VideoCapture(0)

current_moment = time()
time_list = []
while True:
    return_value, image = camera.read()
    image = cv2_letterbox_image(image, (640, 640))
    image = cv2.flip(image, 1)

    inference_image = image.transpose(2, 0, 1)[None, ...]
    inference_image = inference_image/255.0
    inference_image = inference_image.astype(np.float32)

    result = model.run(['output0'], {'images': inference_image})[0][0]
    result = result.transpose()
    result = result[result[:, 4] > 0.1]
    result = sorted(result, key=lambda x: x[4], reverse=True)
    boxes = []
    while len(result) > 0:
        boxes.append(result[0])
        result = [res for res in result if iou(res, result[0]) < 0.9]
    if len(boxes) < 1:
        cv2.imshow('video', image)
        time_list.append(time() - current_moment)
        continue
    for box in boxes:
        start_point = (int(box[0] - box[2] / 2), int(box[1] + box[3] / 2))
        end_point = (int(box[0] + box[2] / 2), int(box[1] - box[3] / 2))
        color = (255, 0, 0)
        image = cv2.rectangle(image, start_point, end_point, color, 2)

    cv2.imshow('video', image)
    time_list.append(time() - current_moment)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

delta_time_list = []
for i in range(1, len(time_list)):
    delta_time_list.append(time_list[i] - time_list[i-1])
print('FPS =', 1 // (sum(delta_time_list) / len(delta_time_list)))

del camera