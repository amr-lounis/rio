import socket
import cv2
import base64
import numpy as np

BUFF_SIZE = 65536
serverAddressPort = ("127.0.0.1", 9999)

class UDP_Server:
    def __init__(self):
        print("UDP_Server init")
        try:
            self.UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            self.UDPClientSocket.bind(serverAddressPort)
            print("UDP_Server started")
        except Exception as e:
            print("Error starting UDP_Server:", e)

    def receive(self):
        while True:
            try:
                msg, addr = self.UDPClientSocket.recvfrom(BUFF_SIZE)
                print('Received message from', addr)

                # Decode the message
                buffer = base64.b64decode(msg)
                print(buffer)

            except Exception as e:
                print("Error receiving message:", e)
                break

    def close(self):
        self.UDPClientSocket.close()
        print("UDP_Server closed")


# Create an instance of the UDP_Server class
udp_server = UDP_Server()

# Receive and process frames from the client
udp_server.receive()

# Call the close method to clean up resources
udp_server.close()
