#https://www.sciencedirect.com/science/article/pii/S1350453318301504#sec0009

### CoM of Body Segments####
### % of segment length from proximal end ####
##proximal_distal##
shoulder_elbow = 0.398
elbow_wrist = 0.428
hip_knee = 0.431
knee_ankle = 0.372
hip_neck = 0.513
ankle_toe = 0.442

#0.6 from base of neck to vertex but body25 model does not
#have a keypoint for the vertex of the head
#the 0th joint (for the nose), is ~60% of the distance
head_CoM = 1

### Relative Mass of Body Segments####
### % body weight ####

# the hands are exlcuded (not a part of body25 model)
# but they make up only 0.71% of bodyweight
upperarm = 0.0308
forearm = 0.0168
thigh = 0.1076
shank = 0.0419
trunk = 0.4959
feet = 0.0064
head = 0.0826

relative_CoM = [shoulder_elbow,shoulder_elbow,elbow_wrist,elbow_wrist,hip_knee,hip_knee,knee_ankle,knee_ankle,ankle_toe,ankle_toe,hip_neck,head_CoM]
relative_mass = [upperarm,upperarm,forearm,forearm,thigh,thigh,shank,shank,feet,feet,trunk,head]
## for the head it's 0.5 from the vertex (top of the head) to the base of the neck
## openpose outputs a point in the middle of the head (kpt 0) but no vertex