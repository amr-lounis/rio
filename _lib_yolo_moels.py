from collections import Counter
import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf


class yolov8_OCR_plate:
    array_names_ocr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
                                'T', 'U', 'V', 'W', 'X', 'Y','Z']
    def __init__(self,path_model_ocr):
        self.yolov8_ocr = yolov8_pt( path_model_ocr )

    def read_plate(self,_image,conf=0.5,imgsz=224):
        objects_ocr = self.yolov8_ocr.process(_image,conf,imgsz)
        objects_ocr.sort(key=lambda x: x[0])
        list_char = []
        for o_ocr in objects_ocr:
            id_object = o_ocr[5]
            char = yolov8_OCR_plate.array_names_ocr[id_object]
            list_char.append(char)
        # plate_txt =  ''.join(map(str, list_char))
        plate_txt =  ''.join(list_char)
        return plate_txt       

class yolov8_pt:
    def __init__(self,path_model:str):
        if not os.path.exists(path_model):
            print("not exist file :",path_model)
            exit()
        self.model = YOLO(path_model)

    def process(self,image_in:np.ndarray,conf=0.5,imgsz=640):
        results = self.model.predict(image_in, save=False,save_txt=False,show=False,verbose = False,conf=conf,imgsz = imgsz)
        objects = np.array(results[0].boxes.data)
        list_ = []
        for o in objects:
            x1,y1,x2,y2,p,id_ = int(o[0]), int(o[1]), int(o[2]), int(o[3]), o[4], int(o[5])
            list_.append([x1,y1,x2,y2,p,id_])
        return list_

class yolov8_onox:
    def __init__(self,path_model:str):
        if not os.path.exists(path_model):
            print("not exist file :",path_model)
            exit()
        self.model = cv2.dnn.readNetFromONNX(path_model)

    def process(self,image_in:np.ndarray,conf=0.5,imgsz=640):
        [height, width, _] = image_in.shape
        length = max((height, width))
        image_rect = np.zeros((length, length, 3), np.uint8)
        image_rect[0:height, 0:width] = image_in
        scale = length / imgsz

        blob = cv2.dnn.blobFromImage(image_rect, scalefactor=1 / 255, size=(imgsz, imgsz), swapRB=True)
        self.model.setInput(blob)
        output_data = self.model.forward()

        return yolov8_process.postprocess_output(output_data,scale,conf)

class yolov8_tflite:
    def __init__(self,path_model:str):
        if not os.path.exists(path_model):
            print("not exist file :",path_model)
            exit()
        self.model = tf.lite.Interpreter(model_path=path_model)
        self.model.allocate_tensors()

    def process(self,image_in:np.ndarray,conf=0.5,imgsz=640):
        [height, width, _] = image_in.shape
        length = max((height, width))
        image_square = np.zeros((length, length, 3), np.uint8)
        image_square[0:height, 0:width] = image_in
        scale = length / imgsz

        image_square = cv2.resize(image_square, (imgsz,imgsz))
        image_square = image_square / 255.0
        image_square = np.expand_dims(image_square, axis=0)
        # input_type = input_details[0]['dtype']
        input_type = np.float32
        image_square = image_square.astype(input_type)

        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        self.model.set_tensor(input_details[0]['index'], image_square)
        self.model.invoke()
        output_data = self.model.get_tensor(output_details[0]['index'])
     
        return yolov8_process.postprocess_output(output_data,scale,conf)
        
class yolov8_process:
    def iou(box1, box2):
        # Calculate the intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        # Calculate the IoU (Intersection over Union)
        iou = intersection / union
        return iou

    def nmsIndices(boxes, scores, scoreThreshold, iouThreshold):
        # Sort the boxes based on scores in descending order
        sortedIndices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        selectedIndices = []
        while sortedIndices:
            current = sortedIndices[0]
            selectedIndices.append(current)

            remainingIndices = []
            for i in range(1, len(sortedIndices)):
                if yolov8_process.iou(boxes[current], boxes[sortedIndices[i]]) <= iouThreshold:
                    remainingIndices.append(sortedIndices[i])

            sortedIndices = remainingIndices

        # Filter out boxes with scores below the threshold
        selectedIndices = [i for i in selectedIndices if scores[i] >= scoreThreshold]
        return selectedIndices

    def convert2inputTflit(pImage_rect:np.ndarray,size_input=640):
        pImage_rect = cv2.resize(pImage_rect, (size_input,size_input))
        pImage_rect = pImage_rect / 255.0
        pImage_rect = np.expand_dims(pImage_rect, axis=0)
        # input_type = input_details[0]['dtype']
        input_type = np.float32
        pImage_rect = pImage_rect.astype(input_type)
    
    def postprocess_output_manul(output_data,scale, confidence_threshold=0.2):
        aa = output_data[0].T
        max_scores = np.max(aa[:,4:], axis=1)
        max_index_scores = np.argmax(aa[:,4:], axis=1)
        bb = np.zeros((8400,4))
        bb[:,0] = aa[:,0] - (0.5 * aa[:,2]) 
        bb[:,1] = aa[:,1] - (0.5 * aa[:,3] )
        bb[:,2] = aa[:,2]
        bb[:,3] = aa[:,3]

        mask = max_scores > confidence_threshold
        bb = bb[mask]
        max_scores = max_scores[mask]
        max_index_scores = max_index_scores[mask]

        r = yolov8_process.nmsIndices(bb,max_scores,0.5,0.9)
        detections = []
        for i in range(len(r)):
            index = r[i]
            box = bb[index]
            x1 = round(box[0] * scale)
            y1 = round((box[1]) * scale)
            x2 = round((box[0] + box[2]) * scale)
            y2 = round((box[1] + box[3]) * scale)
            detections.append([x1,y1,x2,y2,round(max_scores[index],2),max_index_scores[index]])
        return detections

    def postprocess_output(output_data,scale,confidence_threshold=0.5):
        output_data = output_data[0].T
        rows = output_data.shape[0]

        boxes = []
        scores = []
        class_ids = []
        for i in range(rows):
            classes_scores = output_data[i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= confidence_threshold:
                    box = [
                        output_data[i][0] - (0.5 * output_data[i][2]),
                        output_data[i][1] - (0.5 * output_data[i][3]),
                        output_data[i][2],
                        output_data[i][3]]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)
            
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            x1 = round(box[0] * scale)
            y1 = round((box[1]) * scale)
            x2 = round((box[0] + box[2]) * scale)
            y2 = round((box[1] + box[3]) * scale)
            detections.append([int(x1),int(y1),int(x2),int(y2),round(scores[index],2),int(class_ids[index])])
        return detections
