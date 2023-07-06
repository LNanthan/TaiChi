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
success,img = vidcap.read()

meetings = {}

class Meeting:
    def __init__(self,id):
        self.config = {"skeleton": False, "caption": False, "clock":False}
        self.output= cv2.VideoWriter('meeting_'+id+'.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 5, (img.shape[1],img.shape[0]))

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
    # m {id} add {annot} --> annot = {skeleton, caption, clock}
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
skelAddress  = './uds_skeleton'
clockAddress  = './uds_clock'
renderAddress  = './uds_render'

module_addresses = ['./uds_caption','./uds_skeleton', './uds_clock', './uds_render']
annotIdDict = {capAddress:'caption', skelAddress:'skeleton', clockAddress:'clock'}

# connect to all the cllient modules
modules = 4
clients = {capAddress:None, skelAddress:None, clockAddress:None, renderAddress: None}
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
# t_meetingHandler.start()
# t_meetingHandler.join()

annot_sent = 0
render_sent = 0

annotFrames = {capAddress:None, skelAddress:None, clockAddress:None, renderAddress: None}
recieved_frames = 0

##init meetings
m = Meeting('1')
meetings['1'] = m
m.addConfig('skeleton')
while True:
    try:
        count+=1
        print(count)

        #read frame from vid
        success,img = vidcap.read()
        imageBytes = cv2.imencode('.png', img)[1].tobytes()

        #serv --> mods ~ msg format: frame_number(16) size(8) 'f'frame(size)
        #frame is sent with a 'f' prepending it so render can identify as the original frame

        curr_fnum = str(count)
        while(len(curr_fnum)<16):
            curr_fnum ='0' + curr_fnum

        frame_size = str(len(imageBytes))
        while(len(frame_size)<8):
            frame_size ='0' + frame_size

        
        # for address in clients:  
        #     r, w, e = select.select([clients[address]], [], [],0.0) #timeout of 0, so it polls the current connection to see if data is available to be read
          
        #     clients[address].sendall(fnum_msg.encode())
        #     clients[address].sendall(size_msg.encode())
        #     clients[address].sendall('f'.encode())
        #     clients[address].sendall(imageBytes)
            


        for address in clients:  
            #if data is available to be read
            print(address)
            r, w, e = select.select([clients[address]], [], [],0.0) #timeout of 0, so it polls the current connection to see if data is available to be read
            while(len(r) > 0):
                
                
                if (address ==  renderAddress):
                    m_id = clients[address].recv(4).decode()
                    
                    while(m_id[0]=='0'):
                        m_id = m_id[1:]

                    size = clients[address].recv(8).decode()
    
                    while(size[0]=='0'):
                        size = size[1:]
                    size = int(size)
  

                    frame = clients[address].recv(size)
                    while(len(frame)<size):
                        frame+=clients[address].recv(size-len(frame))

                    imgArr = np.frombuffer(frame,dtype=np.uint8)
                    newImg = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)

                    meetings[m_id].output.write(newImg)
                    
                else:
                    #recieve data from the annot mods and send to render
                    ##whatever the modules send to render needs to be in the same format as what the server sends
                    #annot --> serv ~ msg format: size(8) data(size)
                    # where annot_msg: frame_number(16) size_data(8) {id}data(size_data)
                    #the server also sends render the list of meetings that have this annotation configured
                    size = clients[address].recv(8).decode()
                    while(size[0]=='0'):
                        size = size[1:]
                    size= int(size)

                    render_msg = clients[address].recv(size)
                    while(len(render_msg)<size):
                        render_msg+=clients[address].recv(size-len(render_msg))

                    clients[renderAddress].sendall(render_msg)
                    m_id_msg=''
                    for id in meetings:
                        if (meetings[id].config[annotIdDict[address]] == True):
                            m_id_msg+=','+str(id)

                    size_msg = str(len(m_id_msg))
                    while(len(size_msg)<8):
                        size_msg ='0' + size_msg

                    clients[renderAddress].sendall(size_msg.encode())
                    clients[renderAddress].sendall(m_id_msg.encode())
                r, w, e = select.select([clients[address]], [], [],0.0)
            clients[address].sendall(curr_fnum.encode())
            clients[address].sendall(frame_size.encode())
            clients[address].sendall('f'.encode())
            r, w, e = select.select([clients[address]], [], [],0.0)
            print(r)
            clients[address].sendall(imageBytes)
            


    finally:
    # Clean up the connection
    #close msg needs to be exacty 16
        if(count==maxFrames):
            print(count)
            for address in clients:  
                clients[address].sendall("close socket....".encode())
                clients[address].close()
            break
            
                        
        

        
                



        
        
vidcap.release()
for id in meetings:
    meetings[id].output.release()
end = time.time()
print(end-start)