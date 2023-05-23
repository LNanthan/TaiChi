
import socket
import sys
import os
import cv2

import numpy as np




def annotate (imageBytes):
    imgArr = np.frombuffer(imageBytes,dtype=np.uint8)
    img = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)

    height, width = img.shape[0:2]
    img = cv2.rectangle(img, (int(width*0.75),int(height*0.5)), (width,int(height*0.9)), (0,0,255,255), -1)

    return cv2.imencode('.jpeg', img)[1].tobytes()

server_address = './uds_socket'
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

try:
    sock.connect(server_address)
except socket.error as e:
    print (e)
    sys.exit(1)


while True:
    try:
        # Recieve size then image from server
        data = sock.recv(4096)  #recieve image size 
        imBytes = data
        while(len(data)!=0):
            data = sock.recv(4096)  #recieve image size 
            imBytes+=data

        # imgSize = int(data.decode())
        # print(imgSize)
        # data = sock.recv(imgSize)  #recieve img from server
        # while len(data) < imgSize:
        #     data += sock.recv(imgSize)

        #send annotated image size and bytes
        annImgBytes = annotate(imBytes)
        # print(len(np.frombuffer(annImgBytes,dtype=np.uint8)))
        annImgSize = len(annImgBytes)

        msg = str(annImgSize)
        while (len(msg)<8):
            msg = '0'+msg
        print(msg)
        sock.sendall(msg.encode())

        sock.sendall(annImgBytes)
            


    finally:
        data = sock.recv(16)
        if(data.decode()=="close socket"):
            print ('closing socket')
            sock.close()
            break


