
import socket
import sys
import os
import cv2
import time
import csv

import numpy as np

curr_dir = os.getcwd()
os.chdir('..')
os.chdir('..')
print(os.getcwd())
curr_dir = os.getcwd()

#record overall time (incl. time to send/recv data) per frame

f = open('tests/runtime/timeData/caption', 'w+')
writer = csv.writer(f)
header = ['total time per frame']
writer.writerow(header)

row = []


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
        start = time.time()
         # Recieve size then image from server
        data = sock.recv(16)

        if(data.decode() == 'close socket....'):
            close = True
            break
        f_num_msg = data
        f_num = f_num_msg.decode()
        
        while(len(f_num)>0 and f_num[0]=='0'):
            f_num = f_num[1:]
        

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
        row.append(time.time()-start)
        writer.writerow(row)
        row = []



    finally:
        if(close == True):
            print ('closing socket')
            sock.close()
            f.close()
            break

