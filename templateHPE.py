
import socket
import os
from EasyMocap.apps.preprocess  import extract_keypoints  
from EasyMocap.easymocap.annotator.file_utils import read_json
from EasyMocap.easymocap.estimator.wrapper_base import check_result, create_annot_file, save_annot
from glob import glob
from os.path import join
from tqdm import tqdm
import cv2
import sys
import param

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),"EasyMocap","easymocap","estimator"))
sys.path.append(os.path.join(os.path.dirname(__file__),"EasyMocap"))

shoulder_r = 2
shoulder_l = 5


def extract_yolo_hrnet(image_bytes, annot_root, ext, config_yolo, config_hrnet):
    config_yolo.pop('ext', None)
    import torch
    device = torch.device('cuda')
    from YOLOv4 import YOLOv4
    device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
    detector = YOLOv4(device=device, **config_yolo)
    from HRNet import SimpleHRNet
    estimator = SimpleHRNet(device=device, **config_hrnet)

    imgArr = np.frombuffer(image_bytes,dtype=np.uint8)
    image = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)
    #image = cv2.imread(imgname)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.predict_single(image_rgb)
    # forward_hrnet
    points2d = estimator.predict(image_rgb, detections)
    return points2d



##real person is closer to camera so joint length would be bigger
#compare distance from
def measureJoint(kpts1, kpts2):
    l1 = kpts1[shoulder_l] - kpts1[shoulder_r]
    l2 = kpts2[shoulder_l] - kpts2[shoulder_r]

    mag1 = np.linalg.norm(l1[0:2])
    mag2 = np.linalg.norm(l2[0:2])

    if(mag1>mag2):
        return kpts1,kpts2
    else:
        return kpts2,kpts1


def matchKpts(mirror_img):
    r_side = [17,15,2,3,4,9,10,11,24,23,22]
    l_side = [18,16,5,6,7,12,13,14,21,20,19]
    reflected_mirror = mirror_img
    for i in range(0,len(r_side)):
        reflected_mirror[r_side[i]] = mirror_img[l_side[i]]
        reflected_mirror[l_side[i]] = mirror_img[r_side[i]]
    return reflected_mirror

vidcap = cv2.VideoCapture('11_forms_demo_4min.mp4')
success,img = vidcap.read()
im_bytes = cv2.imencode('.jpg', img)[1].tobytes()



keypoints = extract_yolo_hrnet(im_bytes,"data/annots/test",'jpg', extract_keypoints.config['yolo'],extract_keypoints.config['hrnet'])
real,mirror = measureJoint(keypoints[0],keypoints[1])
mirror = matchKpts(mirror)

K = param.K
new_R = param.R1
new_t = param.t1
pose = param.pose
P1 = np.matmul(K,np.eye(3,4))
Identity = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
P2 = np.matmul (K, Identity @ pose)
pts1 = np.transpose(np.delete(real,2,1))
pts2 = np.transpose(np.delete(mirror,2,1))


#points_3D = get_linear_triangulated_points (pose, features1, features2, K)
points_3D = cv2. triangulatePoints(P1, P2, pts1, pts2)

points_3D /= points_3D[3]
points_3D = np.transpose(points_3D)


