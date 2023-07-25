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
#if one foot is 0.0105 higher than the other, it is in the air
foot_in_air_thresh  = 0.0105

#if one foot is at least 5 closer to the CoM than the other, it is supporting
CoM_foot_thresh = 0.05

def getCoM (points_3D):
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

    return CoM
    
def feetStates(CoM,points_3D):
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
    lr_feet_dist = lfoot_mid-rfoot_mid


    #y-axis for the mirrored view (perpendicular to the ground plane)
    y_mirror = np.dot(R2,[0,1,0])
    y_mirror[2]*=-1 #reflect z axis since it was intially reflected
  
    y_mirror_norm = np.sqrt(sum(y_mirror**2))    

    #project the distance vector onto the mirror y-axis to get the actual vertical distance
    proj_of_dist_on_y_mirr = (np.dot(lr_feet_dist, y_mirror)/y_mirror_norm**2)*y_mirror
    vertical_dist = np.linalg.norm(proj_of_dist_on_y_mirr)

    #cosine of vertical distance vector & y_mirror to get direction
    direc = np.dot(proj_of_dist_on_y_mirr,y_mirror)/y_mirror*vertical_dist


    #CoM displacement
    rfoot_CoM_dist = rfoot_mid - CoM
    lfoot_CoM_dist = lfoot_mid - CoM

    #project onto x-z plane by subtracting perpendicular component
    rfoot_CoM_dist_proj = rfoot_CoM_dist - ((np.dot(rfoot_CoM_dist, y_mirror)/y_mirror_norm**2)*y_mirror)
    lfoot_CoM_dist_proj = lfoot_CoM_dist - ((np.dot(lfoot_CoM_dist, y_mirror)/y_mirror_norm**2)*y_mirror)

    #get magnitude of projection
    l_CoM_dist = np.linalg.norm(lfoot_CoM_dist_proj)
    r_CoM_dist = np.linalg.norm(rfoot_CoM_dist_proj)
    

    if vertical_dist>=foot_in_air_thresh:
        if (direc>0):
            #left foot off the ground, right supporting & on ground
            left_foot["ground"] = False
            left_foot["supp"] = False
        else:
            #right foot off the ground, left supporting
            right_foot["ground"] = False
            right_foot["supp"] = False

    #both feet on ground
    else:
        if(r_CoM_dist>=(l_CoM_dist*(1+CoM_foot_thresh))):
             #left foot is supporting weight
            right_foot["supp"] = False
        elif(r_CoM_dist>=(l_CoM_dist*(1+CoM_foot_thresh))):
            #right foot is supporting weight
            left_foot["supp"] = False

    return left_foot, right_foot
