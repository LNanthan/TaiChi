import socket
import sys
import os
from PIL import Image
import cv2 
import numpy as np

count = 0
server_address = './uds_socket'
# image_path = 'test.jpeg'

vidcap = cv2.VideoCapture('11_forms_demo_4min.mp4')
success,img = vidcap.read()
# success = True
result = cv2.VideoWriter('newVid.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 30, (img.shape[1],img.shape[0]))

try:
    os.unlink(server_address)
except OSError:
    if os.path.exists(server_address):
        raise
# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(server_address)


sock.listen(1)

connection, client_address = sock.accept()

while True:
    # Wait for a connection
    
    try:
        count+=1
        success,img = vidcap.read()
        imageBytes = cv2.imencode('.png', img)[1].tobytes()
        imageSize = len(imageBytes)


        msg = "Size %d" % imageSize
   
        connection.sendall(msg.encode())
        print(msg)
        
        data = connection.recv(16) # confirmation that img size was recieved
        print(data.decode())
        
        connection.sendall(imageBytes)

        data = connection.recv(16) # confirmation that img was recieved
        print(data.decode())
        

        #recieve new img
        data = connection.recv(16) # new img size
        print(data.decode())
        connection.sendall("got new size".encode())

        # if str(data).startswith("Size"):
        annImgSize = data.split()[1]

        data = connection.recv(int(annImgSize.decode()))  #recieve new img
        while len(data) < int(annImgSize.decode()):
            data += connection.recv(int(annImgSize.decode()))
        newImgArr = np.frombuffer(data,dtype=np.uint8)
    #  print(len(newImgArr))
        newImg = cv2.imdecode(newImgArr,cv2.IMREAD_UNCHANGED)
    
        connection.sendall("got new image".encode()) #send confirmation

        result.write(newImg)

        #show live video
        cv2.imshow("window",newImg)
        cv2.waitKey(1)

        

            
    finally:
        # Clean up the connection
        if(count==90):
            connection.sendall("close socket".encode()) #send confirmation
            connection.close()
            break
        else:
            connection.sendall("keep open".encode()) #send confirmation
        
        
vidcap.release()
result.release()