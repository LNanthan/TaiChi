import socket
import sys
import os
from PIL import Image
import cv2
import numpy as np
from collections import deque
from threading import Thread, Lock
import time
import csv

numMods = 3

curr_dir = os.getcwd()
os.chdir('..')
os.chdir('..')
print(os.getcwd())
curr_dir = os.getcwd()

f = open('tests/runtime/timeData/render', 'w+')
writer = csv.writer(f)
header = ['render mask','render skel', 'render caption', 'total time']
writer.writerow(header)

row = []

m_time = 0
c_time = 0 
h_time = 0
#record: time to render each mod & tot time
#overall time to render a frame --> assuming one meeting w/ all 3 configs

class Frame:

    #meetings --> id: [img,caption]
    def __init__(self,img):
        self.caption = None
        self.meetings = {}
        self.remaining = numMods
        self.img = img
        self.shape = img.shape

    def addMeeting(self, id):
        #each meeting has the current frame and caption
        self.meetings[id] = [self.img.copy(),None]

    def render (self, id, data, data_id):
        global m_time, h_time
        if (data_id=='m'): #m = mask; expecting new image
            start_m = time.time()
            maskArr = np.frombuffer(data,dtype=np.uint8)
            mask = cv2.imdecode(maskArr,cv2.IMREAD_UNCHANGED)
            img = self.meetings[id][0]
            drawing = img.copy()

            #color of the annotation
            drawing = cv2.rectangle(drawing, (0,0), (self.shape[1],self.shape[0]), (0,0,255,255), -1) #filled rect 
            object = cv2.bitwise_and(drawing, drawing, mask=mask) 
            inversion = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask)) 
            img = cv2.bitwise_or(object, inversion)
            self.meetings[id][0] = img
            m_time = time.time()-start_m

        elif (data_id =='c'): #c = caption
            self.meetings[id][1] = data.decode() # store caption
        
        elif (data_id =='h'): #h = hpe
            start_h = time.time()
            pts_3d = np.frombuffer(data)
            pts_3d = pts_3d.reshape(25,2)
            img = self.meetings[id][0]
            img = skeleton(img,pts_3d)
            h_time = time.time()-start_h


    def placeCaption(self,id):
        global c_time
        start_c = time.time()
        if not self.meetings[id][1] ==  None:
            img = self.meetings[id][0]
            img = add_caption(self.meetings[id][1],img)
        c_time = time.time()-start_c
            



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

joints = [[]]*4
joints[0] = [4,3,2,1,5,6,7] #arms
joints[1] = [23,22,11,24,11,10,9,8,12,13,14,21,14,19,20] #legs  #[11,10,9,8,12,13,14]
joints[2] = [1,8] #torso
def skeleton(img,pts_3d):
    orig = img.copy()
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
server_address = './uds_server'
try:
    sock.connect(server_address)
except socket.error as e:
    print (e)
    sys.exit(1)


close= False

global renderMsgQ
renderMsgQ = deque()


def readInData(lock):
    while True:
        # frame num
        data = sock.recv(16)
        if(data.decode() == 'close socket....'):
            global close 
            close= True
            return
        f_num = data.decode()
        while(f_num[0]=='0'):
            f_num = f_num[1:]

        # dataSize
        size = sock.recv(8).decode()
        while(size[0]=='0'):
            size = size[1:]
        size=int(size)

        # data id {f,c,h,m}
        data_id = sock.recv(1).decode()

        # data
        data = sock.recv(size)  #recieve img from server
        while len(data) < size:
            data += sock.recv(size-len(data))

        # meeting info size
        mSize = sock.recv(8).decode()
        while(mSize[0]=='0'  and len(mSize)>1):
            mSize = mSize[1:]
        mSize=int(mSize)

        #meeting info
        meetingInfo = sock.recv(mSize)  #recieve img from server
        while len(meetingInfo) < mSize:
            meetingInfo += sock.recv(mSize-len(meetingInfo))
        meetingInfo = meetingInfo.decode().split(',')[1:]

        
        renderMsgQ.append([f_num, data_id, data, meetingInfo])
        

            
            
            


frames = {}

lock= Lock()
t_recieveData= Thread(target=readInData, args=(lock,))
t_recieveData.daemon = True
t_recieveData.start()

start = time.time()
while True:
    try:
        if(len(renderMsgQ)>0):

            msg = renderMsgQ.popleft()
           
            #[frame number, data id, data, meetings]
            if msg[1] == 'f':
                imgArr = np.frombuffer(msg[2],dtype=np.uint8)
                img = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)
                currFrame = Frame(img)
                frames[msg[0]] = currFrame
                # add all meetings for this frame
                for i in msg[3]:
                    currFrame.addMeeting(i)
            else:
                currFrame = frames[msg[0]]
                for i in msg[3]:
                    #meeting exists
                    if i in currFrame.meetings:
                        # startRender = time.time()
                        currFrame.render(i,msg[2],msg[1])
                        # print(time.time()-startRender)
                currFrame.remaining -= 1
                if(currFrame.remaining == 0):
                    del frames[msg[0]]
                    for mid in  currFrame.meetings:
                        #caption is added at the end so that it's not covered by the other annotations
                        currFrame.placeCaption(mid)
                        img = currFrame.meetings[mid][0]
                        imgBytes = cv2.imencode('.png', img)[1].tobytes()

                        #send meeting id (4 bytes)
                        id_msg = str(mid)
                        while(len(id_msg)<4):
                            id_msg ='0' + id_msg
                        sock.sendall(id_msg.encode())

                        #send size of frame (16 bytes)
                        size_msg = str(len(imgBytes))
                        while(len(size_msg)<8):
                            size_msg ='0' + size_msg
                        sock.sendall(size_msg.encode())
                        
                        #send img bytes
                        sock.sendall(imgBytes)
                        row = [m_time,c_time,h_time]
                        writer.writerow(row)
                        
               
                        
        
        
    finally:
        if(close==True and len(renderMsgQ)==0):
            sock.sendall('done'.encode())
            print ('closing socket')
            sock.close()
            writer.writerow(['','','',time.time()-start])
            f.close()
            break





        