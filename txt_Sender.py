import time
import cv2, imutils, socket
import numpy as np
import base64

BUFF_SIZE = 65536
serverAddressPort   = ("127.0.0.1", 9999)
# serverAddressPort   = ("192.168.125.5", 9999)

class UDP_Client:
    def __init__(self):
        print("UDP_Client init")
        try:
            self.UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        except:
            print("error init UDP")

    def send(self,text):
        try:
            message = base64.b64encode(str.encode(text))
            self.UDPClientSocket.sendto(message,serverAddressPort)
        except:
            print("error send")

udp = UDP_Client()


udp.send("string ")
