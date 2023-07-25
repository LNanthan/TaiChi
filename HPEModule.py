
import socket
import os
import cv2
import sys
import params.cameraParams as cameraParams
import time
import subprocess
import shutil
import numpy as np
import json




server_address = './uds_server'
HPE_address = './uds_hpe'


#intrinsic and extrinsic camera parameters
K=cameraParams.K
pose = cameraParams.pose
Rt1 = np.eye(3,4)
R1 = Rt1[:,0:3]
t1 = Rt1[:,3]
P1 = np.matmul(K,Rt1)
Identity = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
P2 = np.matmul (K, Identity @ pose)


shutil.rmtree('data/images')
shutil.rmtree('data/annots')
data_pth = "data"
os.makedirs(data_pth+"/images")
os.makedirs(data_pth+"/annots")
curr_dir = os.getcwd()

def get_kpts(img):
    kpts = []

    ##read in image
    cv2.imwrite(data_pth+"/images/curr_frame.jpg",img)
    os.chdir(os.path.join(curr_dir,"openpose"))  #chdir to the openpose file

    #--display 0 and --render_pose 0 saves time
    p = subprocess.call(['./build/examples/openpose/openpose.bin', '--image_dir', '../data/images/', '--write_json', '../data/annots/', 
                         '--display', '0', '--render_pose', '0','--net_resolution','-1x368'])

    os.chdir(curr_dir) #change back to main dir
    
    st = time.time()
    kpt_data = json.load(open("data/annots/curr_frame_keypoints.json"))
    for i in kpt_data['people']:
        kpt_arr = np.array(i['pose_keypoints_2d'])
        kpts.append(kpt_arr.reshape(25,3))    

    return kpts


#use spine length to distinguish between mirror and real kpts
def measureJoint(kpts1, kpts2):
    l1 = kpts1[1] - kpts1[8]
    l2 = kpts2[1] - kpts2[8]
    mag1 = np.linalg.norm(l1[0:2])
    mag2 = np.linalg.norm(l2[0:2])

    if(mag1>mag2):
        return kpts1,kpts2
    else:
        return kpts2,kpts1
    
    
def matchKpts(mirror_img):
    r_side = [17,15,2,3,4,9,10,11,24,23,22]
    l_side = [18,16,5,6,7,12,13,14,21,20,19]
    reflected_mirror = np.copy(mirror_img)
    for i in range(0,len(r_side)):
        reflected_mirror[r_side[i]] = mirror_img[l_side[i]]
        reflected_mirror[l_side[i]] = mirror_img[r_side[i]]
    return reflected_mirror



def get3D (real_kpts,mirror_kpts):
    pts1 = np.transpose(np.delete(real_kpts,2,1))
    pts2 = np.transpose(np.delete(mirror_kpts,2,1))

    points_3D = cv2. triangulatePoints(P1, P2, pts1, pts2)
    points_3D = points_3D[0:3] /points_3D[3] #divide x,y,z by w

    real_projected = cv2.projectPoints(points_3D,R1,t1,K,None)[0]
    real_projected = real_projected.reshape(25,2)

    points_3D = np.transpose(points_3D)

    for i in range(0,len(real_kpts)):
        if real_kpts[i][2]==0.0 or mirror_kpts[i][2]==0.0:
            real_projected[i]=[-1,-1]  #kpts w/ confidence of 0 will be negative so it's not drawn on
            points_3D[i] = [np.nan]

    return real_projected


try:
    os.unlink(HPE_address)
except OSError:
    if os.path.exists(HPE_address):
        raise

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(HPE_address)
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

        #send 3d kpts size and data
        imgArr = np.frombuffer(frame,dtype=np.uint8)
        img = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)
        keypoints = get_kpts(img)  
        real,mirror = measureJoint(keypoints[0],keypoints[1])
        mirror = matchKpts(mirror)
        pts_3d = get3D(real,mirror)
        pts_3d_enc = pts_3d.tobytes()



        #send back to server(which sends to render) using this format:
        # sizeOfRenderMsg(8) renderMsg(sizeOfRenderMsg)
        #renderMsg: frameNum(16) sizeOfData(8) data(sizeOfData)  *character id should be prepended to data
        renderMsg = f_num_msg
        sizeData = str(len(pts_3d_enc))
        while(len(sizeData)<8):
            sizeData ='0' + sizeData
        renderMsg+= sizeData.encode()
        renderMsg+='h'.encode()
        renderMsg += pts_3d_enc

        sizeRenderMsg = str(len(renderMsg))
        while(len(sizeRenderMsg)<8):
            sizeRenderMsg ='0' + sizeRenderMsg

        sock.sendall(sizeRenderMsg.encode())
        sock.sendall(renderMsg)
        
    


    finally:
        if(close==True):
            print ('closing socket')
            sock.close()
            break


