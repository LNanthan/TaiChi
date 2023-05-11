
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
        # Send data
        data = sock.recv(16)  #recieve image size 
        print(data.decode())
        sock.sendall("got size".encode())
        
        # if str(data).startswith("Size"):
        imgSize = data.split()[1]
        data = sock.recv(int(imgSize.decode()))  #recieve img from server
        while len(data) < int(imgSize.decode()):
            data += sock.recv(int(imgSize.decode()))


        sock.sendall("got image".encode()) #send confirmation

        #
        #send annotated image size and bytes
        annImgBytes = annotate(data)
        print(len(np.frombuffer(annImgBytes,dtype=np.uint8)))
        annImgSize = len(annImgBytes)

        msg = "Size %d" % annImgSize
        print(msg)
        sock.sendall(msg.encode())

        data = sock.recv(16) # confirmation that img size was recieved
        print(data.decode())

        sock.sendall(annImgBytes)

        data = sock.recv(16) # confirmation that img was recieved
        print(data.decode())
            


    finally:
        data = sock.recv(16)
        if(data.decode()=="close socket"):
            print ('closing socket')
            sock.close()
            break


