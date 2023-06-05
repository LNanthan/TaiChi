
import socket
import os
# from EasyMocap.easymocap.estimator.yolohrnet_wrapper import  extract_yolo_hrnet 
from EasyMocap.easymocap.annotator.file_utils import read_json
from EasyMocap.easymocap.estimator.wrapper_base import check_result, create_annot_file, save_annot
# from EasyMocap.easymocap.estimator.openpose_wrapper import FeetEstimatorByCrop
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


#print(sys.path)

shoulder_r = 12
shoulder_l = 14

def initDetection (config_yolo,config_hrnet):
    config_yolo.pop('ext', None)
    import torch
    device = torch.device('cuda')
    from YOLOv4 import YOLOv4
    device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
    detector = YOLOv4(device=device, **config_yolo)
    from HRNet import SimpleHRNet
    estimator = SimpleHRNet(device=device, **config_hrnet)
    return detector,estimator

def extract_yolo_hrnet(image_bytes, detector, estimator):
    imgArr = np.frombuffer(image_bytes,dtype=np.uint8)
    image = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.predict_single(image_rgb)
    points2d = estimator.predict(image_rgb, detections)
    print(points2d)
    return points2d

def get_kpts():
    vidcap = cv2.VideoCapture('11_forms_demo_4min.mp4')
    data_pth = "data"

    ##read in image
    start=time.time()
    for i in range (0,1):  
        shutil.rmtree('data/images')
        shutil.rmtree('data/annots')
        os.makedirs(data_pth+"/images")
        os.makedirs(data_pth+"/annots")
        success,img = vidcap.read()
        cv2.imwrite(data_pth+"/images/curr_frame.jpg",img)
        p = subprocess.call(["python3 EasyMocap/apps/preprocess/extract_keypoints.py data --mode yolo-hrnet"], shell=True, executable='/bin/bash')
        p = subprocess.call(['python3', 'EasyMocap/apps/preprocess/extract_keypoints.py', 'data', '--mode', 'feetcrop'])

    end = time.time()
    print(end-start)


##real person is closer to camera so joint length would be bigger
#compare distance from
def measureJoint(kpts1, kpts2):
    # l1 = kpts1[shoulder_l] - kpts1[shoulder_r]
    # l2 = kpts2[shoulder_l] - kpts2[shoulder_r]

    # mag1 = np.linalg.norm(l1[0:2])
    # mag2 = np.linalg.norm(l2[0:2])
    if(kpts1[1][0]>kpts2[1][0]):
        return kpts1,kpts2
    else:
        return kpts2,kpts1

    # if(mag1>mag2):
    #     return kpts1,kpts2
    # else:
    #     return kpts2,kpts1


def matchKpts(mirror_img):
    r_side = [17,15,2,3,4,9,10,11,24,23,22]
    l_side = [18,16,5,6,7,12,13,14,21,20,19]
    reflected_mirror = np.copy(mirror_img)
    for i in range(0,len(r_side)):
        reflected_mirror[r_side[i]] = mirror_img[l_side[i]]
        reflected_mirror[l_side[i]] = mirror_img[r_side[i]]
    return reflected_mirror

def skeleton(img,image_pts,real):
    orig = img.copy()
    joints = [[]]*4
    joints[0] = [4,3,2,1,5,6,7] #arms
    joints[1] = [23,22,11,24,11,10,9,8,12,13,14,21,14,19,20] #legs  #[11,10,9,8,12,13,14]
    joints[2] = [1,8] #torso
    for r in joints:
        start = 0
        for i in r:
            if(start==0 and real[i][2]>0):
                start_pt = (int(image_pts[i][0]),int(image_pts[i][1]))
                start = 1
            elif(real[i][2]>0):
                end_pt = (int(image_pts[i][0]),int(image_pts[i][1]))
              
                img = cv2.line(img, start_pt, end_pt, color=(200, 200, 0,180), thickness=9)  
                img = cv2.circle(img, start_pt, radius=8, color=(255, 255, 255,180), thickness=-1)
                start_pt = end_pt
            img = cv2.circle(img, start_pt, radius=8, color=(255, 255, 255,180), thickness=-1)
    img = cv2.addWeighted(orig,0.35,img,0.65,0.0)
    return img


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
    return real_projected

get_kpts()
# start = time.time()
# vidcap = cv2.VideoCapture('11_forms_demo_4min.mp4')
# success,img = vidcap.read()
# detec, estim = initDetection(extract_keypoints.config['yolo'],extract_keypoints.config['hrnet'])
# for i in range (0,3690):
#     success,img = vidcap.read()
# for i in range(0,1):
#     success,img = vidcap.read()
#     im_bytes = cv2.imencode('.jpg', img)[1].tobytes()
#     keypoints = extract_yolo_hrnet(im_bytes, detec,estim)
    # real,mirror = measureJoint(keypoints[0],keypoints[1])
    # mirror = matchKpts(mirror)
   

    # pts_3d = get3D(real,mirror)
    # # print(pts_3d)
    # real_mag = np.linalg.norm(real[5][0:2] - real[6][0:2])
    # mag_3d = np.linalg.norm(pts_3d[5][0:2] - pts_3d[6][0:2])
    # scale = (real_mag/mag_3d)
    # pts_3d = pts_3d*scale

    # pts_3d = np.transpose(pts_3d)
    # pts_3d[0]+=img.shape[1]/2
    # pts_3d[1]+=img.shape[0]/2
    # pts_3d = np.transpose(pts_3d[0:2])

    
    
    # print(np.transpose(np.transpose(real)[0:2]))
    # img = skeleton(img,pts_3d,real)
    # # print(pts_3d)
    # cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Window",int(img.shape[1]*0.8),int(img.shape[0]*0.8))
    # cv2.imshow("Window",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()








# server_address = './uds_socket'
# # Create a UDS socket
# sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

# try:
#     sock.connect(server_address)
# except socket.error as e:
#     print (e)
#     sys.exit(1)

# while True:
#     try:
#         # Send data
#         data = sock.recv(16)  #recieve image size 
#         print(data.decode())
#         sock.sendall("got size".encode())
        
#         # if str(data).startswith("Size"):
#         imgSize = data.split()[1]
#         data = sock.recv(int(imgSize.decode()))  #recieve img from server
#         while len(data) < int(imgSize.decode()):
#             data += sock.recv(int(imgSize.decode()))


#         sock.sendall("got image".encode()) #send confirmation

#         #
#         #send annotated image size and bytes
#         annImgBytes = add_caption(data)
#         print(len(np.frombuffer(annImgBytes,dtype=np.uint8)))
#         annImgSize = len(annImgBytes)

#         msg = "Size %d" % annImgSize
#         print(msg)
#         sock.sendall(msg.encode())

#         data = sock.recv(16) # confirmation that img size was recieved
#         print(data.decode())

#         sock.sendall(annImgBytes)

#         data = sock.recv(16) # confirmation that img was recieved
#         print(data.decode())
            


#     finally:
#         data = sock.recv(16)
#         if(data.decode()=="close socket"):
#             print ('closing socket')
#             sock.close()
#             break
