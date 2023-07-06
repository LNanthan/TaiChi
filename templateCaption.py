
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

server_address = './uds_server'
caption_address = './uds_caption'

try:
    os.unlink(caption_address)
except OSError:
    if os.path.exists(caption_address):
        raise

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(caption_address)


close = False

try:
    sock.connect(server_address)
    print('connected')
except socket.error as e:
    print (e)
    sys.exit(1)

while True:
    try:
         # Recieve size then image from server
        data = sock.recv(16)
        if(data.decode() == 'close socket....'):
            close = True
            break
        f_num_msg = data
        f_num = f_num_msg.decode()
        while(f_num[0]=='0'):
            f_num = f_num[1:]
        print(f_num)

        size = sock.recv(8).decode()
        while(size[0]=='0'):
            size = size[1:]
        size=int(size)

        data_id = sock.recv(1)

        frame = sock.recv(size)  #recieve img from server
        while len(frame) < size:
            frame += sock.recv(size-len(frame))

        #send annotated image size and bytes
        text = generate_text(frame)


        #send back to server(which sends to render) using this format:
        # sizeOfRenderMsg(8) renderMsg(sizeOfRenderMsg)
        #renderMsg: frameNum(16) sizeOfData(8) data(sizeOfData)  *character id should be prepended to data
        
        renderMsg = f_num_msg
        sizeData = str(len(text))
        while(len(sizeData)<8):
            sizeData ='0' + sizeData
        renderMsg+= sizeData.encode()
        renderMsg+='c'.encode()
        renderMsg += text.encode()

        sizeRenderMsg = str(len(renderMsg))
        while(len(sizeRenderMsg)<8):
            sizeRenderMsg ='0' + sizeRenderMsg

        sock.sendall(sizeRenderMsg.encode())
        sock.sendall(renderMsg)



    finally:
        if(close == True):
            print ('closing socket')
            sock.close()
            break

