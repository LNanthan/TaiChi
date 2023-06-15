import socket
import sys
import os
from PIL import Image
import cv2
import numpy as np
import time
import csv

def add_caption (caption,img):
    imHeight, imWidth = img.shape[0:2]

    font = cv2.FONT_HERSHEY_DUPLEX
    fontscale = 2
    thickness = 2

    color = (0, 0, 0)

    #org = bottom left corner of text
    textWidth= cv2.getTextSize(caption, font, fontscale, thickness)[0][0]

    org = (int((imWidth-textWidth)/2)), int(imHeight*0.9)

    cv2.putText(img, caption, org, font, 
                   fontscale, color, thickness, cv2.LINE_AA)

    return img
count = 0
caption_size = 20
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

frames = int(sys.argv[1])
print(frames)
sock.listen(1)
print("listening")
clients = 2 
connec = []
for i in range(0,clients):
    # Wait for a connection from all clients
    connection, client_address1 = sock.accept()
    connec.append(connection)
    print(connec)
start = time.time()
while True:
    try:
        count+=1
        success,img = vidcap.read()
        imageBytes = cv2.imencode('.png', img)[1].tobytes()
        imageSize = len(imageBytes)
        
        msg = str(imageSize)

        for i in range (0,len(connec)):
            connec[i].sendall(msg.encode())
            data = connec[i].recv(16) # confirmation that img size was recieved
            print(data.decode())
            connec[i].sendall(imageBytes)

        for i in range (0,len(connec)):
            id = connec[i].recv(1)
            size = int(connec[i].recv(8).decode())
            if (id.decode() =='m'): #m = mask; expecting new image
                connec[i].sendall("got size".encode())
                
                data = connec[i].recv(size)  #recieve img from server
                while len(data) < size:
                    data += connec[i].recv(size)

                maskArr = np.frombuffer(data,dtype=np.uint8)
                mask = cv2.imdecode(maskArr,cv2.IMREAD_UNCHANGED)
                drawing = img.copy()
                drawing = cv2.rectangle(drawing, (0,0), (img.shape[1],img.shape[0]), (0,0,255,255), -1) #filled rect 
                object = cv2.bitwise_and(drawing, drawing, mask=mask) 
                inversion = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask)) 
                img = cv2.bitwise_or(object, inversion)
            elif (id.decode() =='c'): #c = caption
                connec[i].sendall("got size".encode())
                data = connec[i].recv(size)  #recieve img from server
                
                while len(data) < size:
                    data += connec[i].recv(size)
                img = add_caption(data.decode(),img)
            
  

    finally:
        # Clean up the connection
         print(count)
         if(count==frames):
            for i in range (0,len(connec)):
                connec[i].sendall("close socket".encode()) #send confirmation
                connec[i].close()
            break
                
         else:
            for i in range (0,len(connec)):
                connec[i].sendall("keep open".encode()) #send confirmation
        

vidcap.release()
end = time.time()
f = open('times','a+')
writer = csv.writer(f)
writer.writerow(['A',frames,end-start])
print(end-start)