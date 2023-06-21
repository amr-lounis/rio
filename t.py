import cv2
import imutils
import socket
import base64

BUFF_SIZE = 65536

class DDD:
    def __init__(self):
        self.port = 9999
        self.host_ip = '127.0.0.1'  # Replace with the desired IP address
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
        self.socket_address = (self.host_ip, self.port)
        self.server_socket.bind(self.socket_address)
        print('Listening at:', self.socket_address)

    def send(self, frame):
        try:
            msg, client_addr = self.server_socket.recvfrom(BUFF_SIZE)
            print('Got connection from', client_addr)
            WIDTH = 400
            frame = imutils.resize(frame, width=WIDTH)
            encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            message = base64.b64encode(buffer)
            self.server_socket.sendto(message, client_addr)
        except:
            pass

    def receive(self):
        while True:
            try:
                msg, client_addr = self.server_socket.recvfrom(BUFF_SIZE)
                print('Received message from', client_addr)

                # Decode the message
                buffer = base64.b64decode(msg)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

                # Display the received frame
                cv2.imshow('Received Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass

    def close(self):
        self.server_socket.close()

vid = cv2.VideoCapture(0) #  replace 'rocket.mp4' with 0 for webcam


ddd = DDD()

while(vid.isOpened()):
    _,frame = vid.read()
    cv2.imshow("ss",frame)
    ddd.send(frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break



import cv2, imutils, socket
import base64

BUFF_SIZE = 65536
class DDD:
    def __init__(self):
        self.port = 9999
        self.host_ip = '127.0.0.1'#  socket.gethostbyname(host_name)
        self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
        self.host_name = socket.gethostname()
        self.socket_address = (self.host_ip,self.port)
        self.server_socket.bind(self.socket_address)
        print('Listening at:',self.socket_address)

    def send(self,frame):
        try:
            msg,client_addr = self.server_socket.recvfrom(BUFF_SIZE)
            print('GOT connection from ',client_addr)
            WIDTH=400
            frame = imutils.resize(frame,width=WIDTH)
            encoded,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
            message = base64.b64encode(buffer)
            self.server_socket.sendto(message,client_addr)
        except:
             pass
	
    def close(self):
	    self.server_socket.close()

vid = cv2.VideoCapture(0) #  replace 'rocket.mp4' with 0 for webcam


ddd = DDD()

while(vid.isOpened()):
    _,frame = vid.read()
    cv2.imshow("ss",frame)
    ddd.send(frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break



