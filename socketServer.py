import socket
import sys
import os
from PIL import Image
import cv2 
import numpy as np
import time
import csv

count = 0
server_address = './uds_socket'
# image_path = 'test.jpeg'

vidcap = cv2.VideoCapture('11_forms_demo_4min.mp4')
success,img = vidcap.read()
# success = True
#result = cv2.VideoWriter('newVid.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 30, (img.shape[1],img.shape[0]))

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
start = time.time()
frames = int(sys.argv[1])
while True:

    try:
        count+=1
        success,img = vidcap.read()
        imageBytes = cv2.imencode('.png', img)[1].tobytes()
        imageSize = len(imageBytes)
   
        connection.sendall(str(imageSize).encode())
        print(imageSize)
        
        data = connection.recv(16) # confirmation that img size was recieved
        print(data.decode())
        
        connection.sendall(imageBytes)
        
        #recieve new img
        data = connection.recv(16) # new img size
        print(data.decode())

        connection.sendall("got new size".encode())
        newImgSize = int(data.decode())

        data = connection.recv(newImgSize)  #recieve new img
        while len(data) < newImgSize:
            data += connection.recv(newImgSize)
        newImgArr = np.frombuffer(data,dtype=np.uint8)

    #  print(len(newImgArr))
        newImg = cv2.imdecode(newImgArr,cv2.IMREAD_UNCHANGED)
    
        # connection.sendall("got new image".encode()) #send confirmation

     #   result.write(newImg)

        #show live video
        print(count)
        # cv2.imshow("window",newImg)
        # cv2.waitKey(1)

        

            
    finally:
        # Clean up the connection
        if(count==frames):
            connection.sendall("close socket".encode()) #send confirmation
            connection.close()
            break
        else:
            connection.sendall("keep open".encode()) #send confirmation
        
        
vidcap.release()
# result.release()
end = time.time()
f = open('times','a+')
writer = csv.writer(f)
writer.writerow(['B',frames,end-start])
print(end-start)