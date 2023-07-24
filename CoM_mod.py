import socket
import os
from glob import glob
from os.path import join
from tqdm import tqdm
import cv2
import sys
import params.bodySegParams as bodySegParams
import time
import subprocess
import shutil
import numpy as np
import json
import params.cameraParams as cameraParams

#camera parameters
K = cameraParams.K
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
Rt2 = Identity @ pose
R2 = Rt2[:,0:3]
t2 = Rt2[:,3]
#if one foot is 2e-2 higher than the other, it is in the air
foot_in_air_thresh  = 0.0105

#if one foot is at least 10% closer to the CoM than the other, it is supporting
CoM_foot_thresh = 0.05

def getCoM (real_kpts,mirror_kpts,img):
    h,w = img.shape[0:2]
    pts1 = np.transpose(np.delete(real_kpts,2,1))
    pts2 = np.transpose(np.delete(mirror_kpts,2,1))

    points_3D = cv2. triangulatePoints(P1, P2, pts1, pts2)
    points_3D = points_3D[0:3] /points_3D[3] #divide x,y,z by w
    points_3D = np.transpose(points_3D)

    for i in range(0,len(real_kpts)):
        if real_kpts[i][2]==0.0 or mirror_kpts[i][2]==0.0:
            points_3D[i] = [np.nan]


    #body segment joints (proximal,distal)
    # upperarm r, upperarm l, forearm r, forearm l, thigh r, thigh l, shank r, shank l, feet r, feet l, trunk, head
    joint_kpts = [[2,3],[5,6],[3,4],[6,7],[9,10],[12,13],[10,11],[13,14],[11,22],[14,19],[8,1],[0,0]]

    rel_CoM = bodySegParams.relative_CoM
    rel_mass = bodySegParams.relative_mass
    segment_CoM_points = []
    tot_mass = 0
  
    #for each segment find absolute CoM and multiply by its relative mass
    # divide the sum of all segments by total relative mass
    for i in range(0,len(joint_kpts)):  
        joint1 = joint_kpts[i][0]  
        joint2 = joint_kpts[i][1]
        pt_3d1 = points_3D[joint1]
        pt_3d2 = points_3D[joint2]
        #0 confidence would have the joint coords be at (0,0,0) which would mess up CoM calc
        if not (np.isnan(pt_3d1[0])or np.isnan(pt_3d2[0])):
            
            abs_CoM = ((pt_3d2-pt_3d1)*rel_CoM[i] + pt_3d1)
            segment_CoM_points.append(abs_CoM*rel_mass[i])
            tot_mass += rel_mass[i]


            
    segment_CoM_points = np.array(segment_CoM_points)
    #overall center of mass
    CoM = segment_CoM_points.sum(axis=0, dtype=np.float64)/tot_mass
    
    CoM_projected = cv2.projectPoints(CoM,R1,t1,K,None)[0]
    CoM_projected = CoM_projected.reshape(2)

    img = cv2.circle(img, (int(CoM_projected[0]),int(CoM_projected[1])), radius=8, color=(180, 255, 0,200), thickness=-1)

    l_heel = points_3D[21]
    l_toe=points_3D[19]
    r_heel = points_3D[24]
    r_toe = points_3D[22]
    left_foot = {
        "ground": True,
        "supp": True
    }
    right_foot = {
        "ground": True,
        "supp": True
    }

    #get midpoint of feet
    lfoot_mid = (l_toe+l_heel)/2
    rfoot_mid = (r_toe+r_heel)/2
    #distance b/w feet
    lr_feet_dist = rfoot_mid-lfoot_mid

    #y-axis for the mirrored view (perpendicular to the ground plane)
    y_mirror = np.dot(R2,[0,1,0])
    y_mirror[2]*=-1 #reflect z axis since it was intially reflected
    y_mirror_norm = np.sqrt(sum(y_mirror**2))    

    #project the distance vector onto the mirror y-axis to get the actual vertical distance
    proj_of_dist_on_y_mirr = (np.dot(lr_feet_dist, y_mirror)/y_mirror_norm**2)*y_mirror
    vertical_dist = np.linalg.norm(proj_of_dist_on_y_mirr)
    
    
    # camera coordinate system's origin is at the center of the image
    # pixel coordinate system's origin is at the top left so any part of the image above the origin is negative
    # higher up = more negative 3D y-coord
    if vertical_dist>=foot_in_air_thresh:
        if (lfoot_mid[1]<rfoot_mid):
            #left foot off the ground, right supporting & on ground
            left_foot["ground"] = False
            left_foot["supp"] = False
            img = cv2.circle(img, (int(w*0.1+30),int(h*0.8)), radius=12, color=(0, 120, 255,200), thickness=-1)
        else:
            #right foot off the gorund, left supporting
            right_foot["ground"] = False
            right_foot["supp"] = False
            img = cv2.circle(img, (int(w*0.1),int(h*0.8)), radius=12, color=(0, 120, 255,200), thickness=-1)

    return img