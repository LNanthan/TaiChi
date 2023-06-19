
import socket
import os
from glob import glob
from os.path import join
from tqdm import tqdm
import cv2
import sys
import param
import time
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
import json



def get_kpts(img):

    data_pth = "data"
    kpts = []

    ##read in image
    start=time.time()
     
    shutil.rmtree('data/images')
    shutil.rmtree('data/annots')
    os.makedirs(data_pth+"/images")
    os.makedirs(data_pth+"/annots")
    cv2.imwrite(data_pth+"/images/curr_frame.jpg",img)
    curr_dir = os.getcwd()
    os.chdir(os.path.join(curr_dir,"openpose"))  #chdir to the openpose file

    #--display 0 and --render_pose 0 saves time
    p = subprocess.call(['./build/examples/openpose/openpose.bin', '--image_dir', '../data/images/', '--write_json', '../data/annots/', 
                         '--display', '0', '--render_pose', '0','--net_resolution','-1x368'])

    os.chdir(curr_dir) #change back to main dir
    end = time.time()
    print(end-start)   

    kpt_data = json.load(open("data/annots/curr_frame_keypoints.json"))
    for i in kpt_data['people']:
        kpt_arr = np.array(i['pose_keypoints_2d'])
        kpts.append(kpt_arr.reshape(25,3))    
    
    return kpts


#limb lengths vary based on current pose -- the arm is not always larger in the real kpts
#so instead the leftmost joints correspond to the mirror person
def measureJoint(kpts1, kpts2):
    if(kpts1[1][0]>kpts2[1][0]):
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
    K = param.K
    new_R = np.array(param.R1)
    new_t = param.t1
    pose = param.pose
    P1 = np.matmul(K,np.eye(3,4))
    Identity = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    P2 = np.matmul (K, Identity @ pose)
    pts1 = np.transpose(np.delete(real_kpts,2,1))

    pts2 = np.transpose(np.delete(mirror_kpts,2,1))

    points_3D = cv2. triangulatePoints(P1, P2, pts1, pts2)

    x=cv2.decomposeProjectionMatrix(P1)


    real_projected = cv2.projectPoints(points_3D[0:3],x[1],x[2][0:3],np.delete(P1,3,1),None)[0]
    real_projected = real_projected.reshape(25,2)


    points_3D /= points_3D[3]
    points_3D = np.transpose(points_3D)

    for i in range(0,len(real_kpts)):
        if real_kpts[i][2]<=0.3:
            real_projected[i]=[-1,-1]  #kpts w/ confidence of 0 will be negative so it's not drawn on

    return real_projected







# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
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
        
        imgSize = int(data.decode())
        print(imgSize)
        sock.sendall('got size'.encode())

        data = sock.recv(imgSize)  #recieve img from server
        while len(data) < imgSize:
            data += sock.recv(imgSize)

        #send 3d kpts size and data
        imgArr = np.frombuffer(data,dtype=np.uint8)
        img = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)
        keypoints = get_kpts(img)  #get 2D
        real,mirror = measureJoint(keypoints[0],keypoints[1])
        mirror = matchKpts(mirror)
        pts_3d = get3D(real,mirror)
        pts_3d_enc = pts_3d.tobytes()
        pts_size = len(pts_3d_enc)
        
        # print(len(np.frombuffer(annImgBytes,dtype=np.uint8)))
        sock.sendall('h'.encode())
        print("h")
        sock.sendall(str(pts_size).encode())
        print(pts_size)
        data = sock.recv(16)
        sock.sendall(pts_3d_enc)



    finally:
        data = sock.recv(16)
        print(data.decode())
        if(data.decode()=="close socket"):
            print ('closing socket')
            sock.close()
            break

