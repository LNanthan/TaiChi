import socket
import sys
import os
from PIL import Image
import cv2
import numpy as np

numMods = 3

class Frame:

    #meetings --> id: [img,caption]
    def __init__(self,img):
        self.meetings = {}
        self.remaining = numMods
        self.img = img
        self.shape = img.shape
    def addMeeting(self, id):
        self.meetings[id] = [self.img,None]
    def render (self, id, data):
        if (data[0] =='m'): #m = mask; expecting new image
            data = data[1:]
            maskArr = np.frombuffer(data,dtype=np.uint8)
            mask = cv2.imdecode(maskArr,cv2.IMREAD_UNCHANGED)
            img = self.meetings[id][0]
            drawing = img.copy()
            drawing = cv2.rectangle(drawing, (0,0), (self.shape[1],self.shape[0]), (0,0,255,255), -1) #filled rect 
            object = cv2.bitwise_and(drawing, drawing, mask=mask) 
            inversion = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask)) 
            img = cv2.bitwise_or(object, inversion)

        elif (data[0] =='c'): #c = caption
            self.meetings[id][1] = data[1:] # store caption
        
        elif (data[0] =='h'): #h = hpe
            data = data[1:]
            pts_3d = np.frombuffer(data)
            pts_3d = pts_3d.reshape(25,2)
            img = self.meetings[id][0]
            img = skeleton(img,pts_3d)
    def placeCaption(self,id):
        if not self.meetings[id][1] ==  None:
            img = self.meetings[id][0]
            img = add_caption[img]
            



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
render_address = './uds_render'

try:
    os.unlink(render_address)
except OSError:
    if os.path.exists(render_address):
        raise
# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(render_address)

##connect to server as client
server_address = './uds_socket'
try:
    sock.connect(server_address)
except socket.error as e:
    print (e)
    sys.exit(1)


frames = {}

   
while True:
    try:
        f_num = sock.recv(16)
        while(f_num[0]=='0'):
            f_num = f_num[1:]

        size = sock.recv(16)
        while(size[0]=='0'):
            size = size[1:]

        data = sock.recv(size)  #recieve img from server
        while len(data) < size:
            data += sock.recv(size-len(data))
        data = data.decode()
        if data[0] == 'f':
            imgArr = np.frombuffer(data,dtype=np.uint8)
            img = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)
            frames[f_num] = Frame(img)
        else:
            size = sock.recv(16)
            while(size[0]=='0'):
                size = size[1:]
            meetingInfo = sock.recv(size)  #recieve img from server
            while len(meetingInfo) < size:
                meetingInfo += sock.recv(size-len(meetingInfo))
            meetingInfo = meetingInfo.decode().split(',')[1:]  #the first character would be a comma in meetingInfo so first item in split would be ''
            for i in meetingInfo:
                #meeting exists
                if i in frames[f_num].meetings:
                    frames[f_num].render(i,data)
                else:
                    frames[f_num].addMeetings[i]
                    frames[f_num].render(i,data)
                frames[f_num].remaining -= 1
                if(frames[f_num].remaining == 0):
                    for mid in  frames[f_num].meetings:
                        #caption is added at the end so that it's not covered by the other annotations
                        frames[f_num].placeCaption(mid)
                        img = frames[f_num].meetings[mid][0]
                        imgBytes = cv2.imencode('.png', img)[1].tobytes()

                        #send meeting id (4 bytes)
                        id_msg = str(count)
                        while(len(id_msg)<4):
                            id_msg ='0' + id_msg
                        sock.sendall(id_msg.encode())

                        #send size of frame (16 bytes)
                        size_msg = str(len(imgBytes))
                        while(len(size_msg)<16):
                            size_msg ='0' + size_msg
                        sock.sendall(size_msg.encode())

                        #send img bytes
                        sock.sendall(imgBytes)
            
        
    finally:
        data = sock.recv(16)
        print(data.decode())
        if(data.decode()=="close socket"):
            print ('closing socket')
            sock.close()
            break





        