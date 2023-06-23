import socket
import base64

# serverAddressPort   = ("127.0.0.1", 9999)
serverAddressPort   = ("192.168.125.255", 9999)

class UDP_Client:
    def __init__(self):
        print("UDP_Client init")
        try:
            self.UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        except:
            print("error init UDP")

    def send(self,text):
        message = base64.b64encode(str.encode(text))
        try:
            self.UDPClientSocket.sendto(message,serverAddressPort)
        except:
            print("error send")

udp = UDP_Client()


udp.send("hello world . ")
