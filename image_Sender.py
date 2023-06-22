import time
import cv2, imutils, socket
import numpy as np
import base64

BUFF_SIZE = 65536
# serverAddressPort   = ("127.0.0.1", 9999)
serverAddressPort   = ("192.168.125.5", 9999)

class UDP_Client:
    def __init__(self):
        print("UDP_Client init")
        try:
            self.UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        except:
            print("error init UDP")

    def send(self,frame):
        try:
            frame = imutils.resize(frame,width=400)
            encoded,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
            message = base64.b64encode(buffer)
            self.UDPClientSocket.sendto(message,serverAddressPort)
        except:
            print("error send")

udp = UDP_Client()

vid = cv2.VideoCapture(0) #  replace 'rocket.mp4' with 0 for webcam

while vid.isOpened():
    _,frame = vid.read()

    udp.send(frame)
    time.sleep(0.1)
    # cv2.imshow("ccc",frame)
    # if cv2.waitKey(60) & 0xFF == 27:
    #     break