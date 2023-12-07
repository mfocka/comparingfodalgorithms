import os
from pathlib import Path
import time
import pandas as pd
from typing import List
import cv2

import numpy as np
from detectors.yolov4 import YOLOv4
from detectors.yolov8 import YOLOv8
from detectors.ssdmobilenet import SSDMobileNet
from detectors.efficientdet import EfficientDet
from detectors.detector import Detector

MODELNAMES = ["paddlepaddle", "rtdetr","ssdmobilenet","yolov4","yolov8","efficientdet"]

# def calc_iou(trueBox, detBox):
#     sX1, sY1, eX1, eY1 = trueBox
#     sX2, sY2, eX2, eY2 = detBox
    
#     intersection_width = max(0, min(eX1, eX2) - max(sX1, sX2))
#     intersection_height = max(0, min(eY1, eY2) - max(sY1, sY2))
#     intersection = intersection_width * intersection_height
#     area_trueBox = (eX1 - sX1) * (eY1 - sY1)
#     area_detBox = (eX2 - sX2) * (eY2 - sY2)
#     union = area_trueBox + area_detBox - intersection

#     iou = intersection / union if union != 0 else 0

#     return iou
    
# def find_nearest_neighbor(bbox, bboxes):
#     nn = np.argmax([calc_iou(bbox, box) for box in bboxes])
#     return nn, calc_iou(bbox, bboxes[nn])

# def calc_precision_recall(detections, expectedDetections, iouThreshold: float = 0.35):
#     result = [
#         [0 for _ in range(len(detections))] for _ in range(len(expectedDetections))
#         ]
#     resultNamed = [
#         ["" for _ in range(len(detections))] for _ in range(len(expectedDetections))
#     ]
#     for idE, expectedDetection in enumerate(expectedDetections):
#         nn, iou = find_nearest_neighbor(expectedDetection, detections)
#         if result[idE][nn] < iou and iou >= iouThreshold:
#             result[idE][nn] = iou
            
#     for idR, row in enumerate(result):
#         for idC, col in enumerate(row):
#             tdInRow = [colId for colId, val in enumerate(resultNamed[idR]) if val == "TD"]
#             if col == 0:
#                 resultNamed[idR][idC] = "MD"
#             elif len(tdInRow) > 0:
#                 nId = np.argmax([row[idc] for idc in tdInRow])
#                 resultNamed[idR][idC] = "TD"
#             else:
#                 resultNamed[idR][idC]
            
#     for idD, detection in enumerate(detections):
#         nn, iou = find_nearest_neighbor(detection, expectedDetections)
#         if result[nn][idD] < iou and iou >= iouThreshold:
#             result[nn][idD] = iou
    
    
#     for row in resultNamed:
#         r = ''
#         for col in row:
#             r += f"| {col} |"
#         print(r)
    
#     for row in result:
#         r = ''
#         for col in row:
#             r += f"| {col:.2f} |"
#         print(r)
    
        
#     return 0.0, 0.0

# def check_correctness_detections(detections, expectedDetections):
#     precision, recall = calc_precision_recall(detections, expectedDetections)
#     beta = 2 # recall is considered beta times as important as precision
#     assert beta >= 0
#     f_beta = (1 + beta**2) * ((precision * recall) / ((beta**2*precision) + recall))

#     return f_beta

def update_csv_file(ID, Model, ConfidenceThreshold, AvgWarmUpTime, AvgInferenceTime, AverageDetectionLen, AverageF_BetaScore):
    csv_file_path = "./outputs/logging.csv"
    row_headers = ["ID", "Model", "ConfidenceThreshold", "AvgWarmUpTimeMs", "AvgInferenceTimeMs","Avg#Detection", "AverageF_BetaScore"]
    new_row = [ID, Model, ConfidenceThreshold, AvgWarmUpTime, AvgInferenceTime, AverageDetectionLen, AverageF_BetaScore]

    if not os.path.exists(csv_file_path):
        df = pd.DataFrame(columns=row_headers)
    else:
        df = pd.read_csv(csv_file_path)

    row_index = df[df["ID"] == ID].index

    if not row_index.empty:
        df.loc[row_index] = new_row
    else:
        new_df = pd.DataFrame([new_row], columns=row_headers)
        df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(csv_file_path, index=False)

def run_model(model: Detector, images: List[np.ndarray], imageNames: str, expectedDetectionsList):
    warmUps = []
    inferns = []
    detectionLengths = []
    f_betaScores = []
    ID = ""
    for idx, (image, name, expectedDetections) in enumerate(zip(images, imageNames, expectedDetectionsList)):
        startTime  = time.time()
        model.warm_up_inference()
        delta      = time.time() - startTime
        warmUps.append(delta)
        startTime  = time.time()
        detections = model.get_detections(image)
        delta      = time.time() - startTime
        inferns.append(delta)
        detectionLengths.append(len(detections))
        nFrame     = model.draw_detections(image, detections)
        folder = Path(f"outputs/{name}")
        ID = f"{folder}_{model.name}_{model.confidenceThreshold}_{idx}"
        cv2.imwrite(f"{ID}.png", nFrame)
    #     f_betaScore = check_correctness_detections(detections, expectedDetections)
    #     f_betaScores.append(f_betaScore)
    f_betaScores = [0]
    if len(warmUps) > 0:
        avg_warm_up_time = sum(warmUps) / len(warmUps)
        avg_inference_time = sum(inferns) / len(inferns)
        avg_f_beta_score = sum(f_betaScores) / len(f_betaScores)
        avg_detection_len = sum(detectionLengths) / len(detectionLengths)
        update_csv_file(ID, model.name, model.confidenceThreshold, avg_warm_up_time, avg_inference_time, avg_detection_len, avg_f_beta_score)
        return avg_warm_up_time, avg_inference_time, avg_f_beta_score
    # else:
    #     return None

        
        
        
        
# def test_calc_precision_recall():
#     detections = [(0,0,10,10),(5,0,15,5)]
#     expectedDs = [(0,0,10,10),(4,0,10,5),(0,0,10,5)]
#     calc_precision_recall(detections, expectedDs)
def get_images(folders, extensions=['.jpg', '.jpeg', '.png']):
    images = []
    image_names = []

    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    img_path = os.path.join(root, file)
                    image_name = img_path.split("inputs/")[1]
                    image_names.append(image_name.split(".")[0])
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)

    return images, image_names
def test_model(model: Detector):
    model.setup_interpreter()
    # name = "OTHER/luggage"
    # luggage = str(Path("inputs/OTHER/luggage.jpg").resolve().absolute())
    # luggageImage = cv2.imread(luggage)
    folder = "inputs"
    folders = [f"{folder}/{location}/" for location in ["BASLER", "OAK", "OTHER"]]
    images, image_names  = get_images(folders)
    print(image_names)
    res = run_model(model, images, folders, image_names)
    print(res)
    
if __name__ == "__main__":
    
    # test_calc_precision_recall()
    ml = Path("./models/")
    for i in range(1,20):
        model = EfficientDet(ml, ml, '.tflite', i/20, onEdge=False, doTpu=False)
        test_model(model)
        model = SSDMobileNet(ml, ml, '.tflite', i/20, onEdge=False, doTpu=False)
        test_model(model)
        model = YOLOv8(ml, ml, '.tflite', i/20, onEdge=False, doTpu=False)
        test_model(model)