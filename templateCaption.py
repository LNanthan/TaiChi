
import socket
import sys
import os
import cv2

import numpy as np

def add_caption (imageBytes):
    imgArr = np.frombuffer(imageBytes,dtype=np.uint8)
    img = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)

    imHeight, imWidth = img.shape[0:2]

    caption = generate_text(img)

    font = cv2.FONT_HERSHEY_DUPLEX
    fontscale = 2
    thickness = 2

    color = (0, 0, 0)

    #org = bottom left corner of text
    textWidth= cv2.getTextSize(caption, font, fontscale, thickness)[0][0]

    org = (int((imWidth-textWidth)/2)), int(imHeight*0.9)

    image = cv2.putText(img, caption, org, font, 
                   fontscale, color, thickness, cv2.LINE_AA)

    return cv2.imencode('.jpeg', image)[1].tobytes()
    #return

def generate_text(image):
  caption = "placeholder"
  return caption


server_address = './uds_socket'
# Create a UDS socket
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
        annImgBytes = add_caption(data)
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

