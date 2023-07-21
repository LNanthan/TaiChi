import socket
import sys
import os
from PIL import Image
import cv2
import numpy as np
import time
import csv

def add_caption (caption,img):
    if caption == None:
        return img
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

def skeleton(img,pts_3d):
    orig = img.copy()
    joints = [[]]*4
    joints[0] = [4,3,2,1,5,6,7] #arms
    joints[1] = [23,22,11,24,11,10,9,8,12,13,14,21,14,19,20] #legs  #[11,10,9,8,12,13,14]
    joints[2] = [1,8] #torso
    for r in joints:
        start = 0
        for i in r:
            if(start==0 and pts_3d[i][0]>=0):
                start_pt = (int(pts_3d[i][0]),int(pts_3d[i][1]))
                start = 1
                
            elif(pts_3d[i][0]>=0):
                end_pt = (int(pts_3d[i][0]),int(pts_3d[i][1]))
                img = cv2.line(img, start_pt, end_pt, color=(200, 200, 0,180), thickness=9)  
                img = cv2.circle(img, start_pt, radius=8, color=(255, 255, 255,180), thickness=-1)
                start_pt = end_pt
        if(start==1):   
            img = cv2.circle(img, start_pt, radius=8, color=(255, 255, 255,180), thickness=-1)  #draw last joint 
    img = cv2.addWeighted(orig,0.35,img,0.65,0.0)
    return img



count = 0
caption_size = 20
server_address = './uds_server'
# image_path = 'test.jpeg'

vidcap = cv2.VideoCapture('11_forms_demo_4min.mp4')
success,img = vidcap.read()
for i in range (0,300):
    success,img = vidcap.read()
# success = True
result = cv2.VideoWriter('skel_test.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 20, (img.shape[1],img.shape[0]))

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
clients = 1 
connec = []
for i in range(0,clients):
    # Wait for a connection from all clients
    connection, client_address1 = sock.accept()
    connec.append(connection)
    print(connec)
start = time.time()
import select
while True:
    try:
        
        count+=1
        success,img = vidcap.read()
        caption = None
        imageBytes = cv2.imencode('.png', img)[1].tobytes()
        size_msg = str(len(imageBytes))

        while(len(size_msg)<8):
            size_msg ='0' + size_msg
            
        print(size_msg)

        for i in range (0,len(connec)):
            connec[i].sendall(size_msg.encode())
           # data = connec[i].recv(16) # confirmation that img size was recieved
           # print(data.decode())
            connec[i].sendall(imageBytes)

        for i in range (0,len(connec)):
            id = connec[i].recv(1)
            size = int(connec[i].recv(8).decode())
            if (id.decode() =='m'): #m = mask; expecting new image
                connec[i].sendall("got size".encode())
                data = connec[i].recv(size)  #recieve img from serve
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
                caption = data.decode()

            elif (id.decode() =='h'): #h = hpe
                connec[i].sendall("got size".encode())
                data = connec[i].recv(size)  #recieve img from server
                while len(data) < size:
                    data += connec[i].recv(size)
                pts_3d = np.frombuffer(data)
                pts_3d = pts_3d.reshape(25,2)
                img = skeleton(img,pts_3d)

        img = add_caption(caption,img)

        result.write(img)
            
  

    finally:
        # Clean up the connection
         print(count)
         if(count==frames):
            for i in range (0,len(connec)):
                connec[i].sendall("close...".encode()) #send confirmation
                connec[i].close()
            break
                
         else:
            for i in range (0,len(connec)):
                connec[i].sendall("keepOpen".encode()) #send confirmation
        

vidcap.release()
result.release()
end = time.time()
print(end-start)