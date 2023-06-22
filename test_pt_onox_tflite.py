import os
import cv2
import numpy as np
from ultralytics import YOLO

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

def drow_objects(image:np.ndarray,objects):
    for o in objects:
        x1,y1,x2,y2,p,id_ = o
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(id_), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
def main():
    yolov8 = yolov8_pt('yolov8n.pt')

    cap=cv2.VideoCapture(0)   
    while True:
        ret,frame = cap.read()
        if not ret:
            break

        cc = yolov8.process(frame,0.75,480)

        list_person = []
        for c in cc :
            if c[5] == 0:
                list_person.append(c)

        print("number : ",len(cc))

        drow_objects(frame,list_person)

        cv2.imshow("WINDOW_NAME", frame)
        if cv2.waitKey(1)&0xFF==27:
            break


    cv2.destroyAllWindows()

main()

