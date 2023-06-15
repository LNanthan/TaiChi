
import socket
import sys
import os
import cv2

import numpy as np

def add_caption (imageBytes):
    imgArr = np.frombuffer(imageBytes,dtype=np.uint8)
    img = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)
    blank = np.zeros(img.shape()) #black background

    imHeight, imWidth = img.shape[0:2]

    caption = generate_text(img)

    font = cv2.FONT_HERSHEY_DUPLEX
    fontscale = 2
    thickness = 2

    color = (0, 0, 0)

    #org = bottom left corner of text
    textWidth= cv2.getTextSize(caption, font, fontscale, thickness)[0][0]

    org = (int((imWidth-textWidth)/2)), int(imHeight*0.9)

    cv2.putText(blank, caption, org, font, 
                   fontscale, color, thickness, cv2.LINE_AA)

    return cv2.imencode('.jpeg', blank)[1].tobytes()
    #return

def generate_text(imageBytes):
  caption = "placeholder"
  return caption


render_address = './uds_render'
try:
    os.unlink(render_address)
except OSError:
    if os.path.exists(render_address):
        raise

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(render_address)

server_address = './uds_socket'


try:
    sock.connect(server_address)
    print('connected')
except socket.error as e:
    print (e)
    sys.exit(1)

while True:
    try:
         # Recieve size then image from server
        data = sock.recv(8)  #recieve image size 
        print(data.decode())
        imgSize = int(data.decode())
        sock.sendall('got size'.encode())

        data = sock.recv(imgSize)  #recieve img from server
        while len(data) < imgSize:
            data += sock.recv(imgSize)

        #send annotated image size and bytes
        text = generate_text(data)
        
        # print(len(np.frombuffer(annImgBytes,dtype=np.uint8)))
        sock.sendall('c'.encode())
        print("c")
        sock.sendall(str(len(text)).encode())
        print(len(text))
        data = sock.recv(16)
        sock.sendall(text.encode())



    finally:
        data = sock.recv(16)
        if(data.decode()=="close socket"):
            print ('closing socket')
            sock.close()
            break

