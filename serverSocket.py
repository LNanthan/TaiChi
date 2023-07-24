import socket
import sys
import os
from PIL import Image
import cv2 
import numpy as np
import time
import csv
from threading import Thread, Lock
import select

count = 0
server_address = './uds_server'
maxFrames = int(sys.argv[1])
# image_path = 'test.jpeg'

vidcap = cv2.VideoCapture('11_forms_demo_4min.mp4')
for i in range(0,1460):
    success,img = vidcap.read()

meetings = {}

class Meeting:
    def __init__(self,id):
        self.config = {"HPE": False, "caption": False, "mask":False}
        self.output= cv2.VideoWriter('meeting_'+id+'.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 20, (img.shape[1],img.shape[0]))

    def removeConfig(self, annot):
        if annot in self.config:
            self.config[annot] = False
            print("annot removed")
        else:
            print("Not a valid annotation")
    def addConfig(self,annot):
        if annot in self.config:
            self.config[annot] = True
            print("annot added")
        else:
            print("Not a valid annotation")

def updateMeeting(lock): 
    # add meeting {id}
    # rm meeting {id}
    # m {id} add {annot} --> annot = {skeleton, caption, mask}
    # m {id} rm {annot} 

    while True:
        cmd  = input().split()
        # lock.acquire()
        if(cmd[0] == "add"):
            # add meeting
            id  = cmd[2]
            if id in meetings:
                print("The meeting already exists")
            else:
                m = Meeting(id)
                meetings[id] = m
                print("meeting started")

        elif(cmd[0] == "rm"):
            # remove meeting
            id  = cmd[2]
            if id in meetings:
                meetings[id].output.release()
                meetings.pop(id)
                print("meeting removed")
            else:
                print("meeting id does not exist")

        # add annot
        elif(cmd[2] == "add"):
            id  = cmd[1]
            if id in meetings:
                meetings[id].addConfig(cmd[3])
                
            else:
                print("meeting id does not exist")
        # rm annot
        elif(cmd[2] == "rm"):
            id  = cmd[1]
            if id in meetings:
                meetings[id].removeConfig(cmd[3])
            else:
                print("meeting id does not exist")
        # lock.release()
        



try:
    os.unlink(server_address)
except OSError:
    if os.path.exists(server_address):
        raise
# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(server_address)
sock.listen(1)


capAddress  = './uds_caption'
HPEAddress  = './uds_hpe'
maskAddress  = './uds_mask'
renderAddress  = './uds_render'

module_addresses = ['./uds_caption','./uds_hpe', './uds_mask', './uds_render']
annotIdDict = {capAddress:'caption', HPEAddress:'HPE', maskAddress:'mask'}

# connect to all the cllient modules
modules = 4
clients = {capAddress:None, HPEAddress:None, maskAddress:None, renderAddress: None}

i= 0
# Wait for a connection from all clients
print("listening for connections")
while i<modules:
    connection, client_address = sock.accept()
    if client_address in clients:
        i+=1
        clients[client_address] = connection
        print(client_address)
print("done connecting")

start = time.time()
frames = int(sys.argv[1])

# read in from terminal
lock= Lock()
t_meetingHandler= Thread(target=updateMeeting, args=(lock,))
t_meetingHandler.setDaemon(True)
t_meetingHandler.start()



annot_sent = 0
render_sent = 0
renderedAll = False
annotFrames = {capAddress:None, HPEAddress:None, maskAddress:None, renderAddress: None}
remaining_annots = maxFrames*(modules-1)

def sendMeetingInfo(config):
    m_id_msg=''
    for id in meetings:
        if (config=='all' or meetings[id].config[config] == True):
            m_id_msg+=','+str(id)

    size_msg = str(len(m_id_msg))
    while(len(size_msg)<8):
        size_msg ='0' + size_msg
    clients[renderAddress].sendall(size_msg.encode())
    clients[renderAddress].sendall(m_id_msg.encode())

##init meetings

while True:
    try:
        count+=1
        
        #add all configs at start
        if (count ==1):
            m = Meeting('1')
            meetings['1'] = m
            # m.addConfig('mask')
            m.addConfig('HPE')
            # m.addConfig('caption')

        #read frame from vid
        success,img = vidcap.read()
        imageBytes = cv2.imencode('.png', img)[1].tobytes()

        #serv --> mods ~ msg format: frame_number(16) size(8) 'f'frame(size)
        #frame is sent with a 'f' prepending it so render can identify as the original frame

        curr_fnum = str(count)
        #prepend 0's to the beginning of the msg to reach the desired size
        while(len(curr_fnum)<16):
            curr_fnum ='0' + curr_fnum

        frame_size = str(len(imageBytes))
        while(len(frame_size)<8):
            frame_size ='0' + frame_size

        if(count<=maxFrames):
            for address in clients:  
                clients[address].sendall(curr_fnum.encode())
                clients[address].sendall(frame_size.encode())
                clients[address].sendall('f'.encode())
                clients[address].sendall(imageBytes)
                if(address == renderAddress):
                    #send all the meetings that require this frame to render
                    sendMeetingInfo('all')
            

        
        for address in clients:  
            #if data is available to be read
            r=[]
            if(clients[address].fileno()>=0):
                #timeout of 0.0, so it polls the current connection to see if data is available to be read
                r, w, e = select.select([clients[address]], [], [],0.0) 
            while(len(r) > 0):
                if (address ==  renderAddress):
                    data = clients[address].recv(4).decode()
                    #after render processes all frames it sends a msg to server
                    if(data == 'done'):
                        renderedAll = True
                        break
                    else:
                        #get meeting id for curr frame
                        m_id = data
                    
                    while(len(m_id)>0 and m_id[0]=='0'):
                        m_id = m_id[1:]

                    #size of rendered frame
                    size = clients[address].recv(8).decode()
    
                    while(size[0]=='0'):
                        size = size[1:]
                    size = int(size)
  

                    frame = clients[address].recv(size)
                    while(len(frame)<size):
                        frame+=clients[address].recv(size-len(frame))

                    imgArr = np.frombuffer(frame,dtype=np.uint8)
                    newImg = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)

                    #write the new frame to the appropriate meeting
                    meetings[m_id].output.write(newImg)

                    #poll render again and read in more data if available
                    #render is polled again so that the same frames for different meetings can be outputted immediately
                    r, w, e = select.select([clients[address]], [], [],0.0) 
                    
                else:
                    #recieve data from the annot mods and send to render
                    ##whatever the modules send to render needs to be in the same format as what the server sends
                    #annot --> serv ~ msg format: size(8) data(size)
                    # where annot_msg: frame_number(16) size_data(8) {id}data(size_data)
                    #the server also sends render the list of meetings that have this annotation configured
                    size = clients[address].recv(8).decode()
                    if(len(size)==0): #when the socket on the other end is shutdown, it sends a msg with size 0
                        clients[address].close()
                        break
                    
                    while(size[0]=='0'):
                        size = size[1:]
                    size= int(size)

                    render_msg = clients[address].recv(size)
                    while(len(render_msg)<size):
                        render_msg+=clients[address].recv(size-len(render_msg))

                    clients[renderAddress].sendall(render_msg)
                    

                    sendMeetingInfo(annotIdDict[address])
                    remaining_annots-=1
                    r = []
                
           
            


    finally:
    # Clean up the connection
    #close msg needs to be exacty 16
        if(count%10 == 0):
            print(count)
        if(count==maxFrames):
            #tell clients to close but wait until all frames are recieved to end connection
            for address in clients:  
                if not address==renderAddress:
                    clients[address].sendall("close socket....".encode())
        elif(remaining_annots==0):
            clients[renderAddress].sendall("close socket....".encode())
            remaining_annots = -1
        if(renderedAll == True):
            clients[renderAddress].close()
            break

        
vidcap.release()
for id in meetings:
    meetings[id].output.release()
end = time.time()
print(end-start)