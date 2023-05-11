
import socket
import sys
import os
import cv2

import numpy as np

from IPython import display
from PIL import Image
import tensorflow.compat.v1 as tf
import sys
sys.path.insert(0, 'tpu/models/official')
sys.path.insert(0, 'tpu/models/official/detection')
sys.path.insert(0, 'tpu/models/official/detection/utils')
from utils.object_detection import visualization_utils
from evaluation import coco_utils
from google.cloud import storage


session = tf.Session(graph=tf.Graph())
#output_image_path = 'maskImage.png'

storage_client = storage.Client.create_anonymous_client()
bucket = storage_client.bucket("cloud-tpu-checkpoints")

if not os.path.exists("shapemask"):
    os.makedirs("shapemask")
    blob = bucket.blob("shapemask/1571767330/saved_model.pb")
    blob.download_to_filename("shapemask/saved_model.pb")

if not os.path.exists("shapemask/variables"):
    os.makedirs("shapemask/variables")  
    blob = bucket.blob("shapemask/1571767330/variables/variables.data-00000-of-00001")
    blob.download_to_filename("shapemask/variables/variables.data-00000-of-00001")

    blob = bucket.blob("shapemask/1571767330/variables/variables.index")
    blob.download_to_filename("shapemask/variables/variables.index")

saved_model_dir = 'shapemask' #@param {type:"string"}
_ = tf.saved_model.load(session, ['serve'], saved_model_dir)


def mask_image (imageBytes):
    imgArr = np.frombuffer(imageBytes,dtype=np.uint8)
    img = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)

    np_image_string = np.array([imageBytes])
    height, width = img.shape[0:2]


    num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, detection_outer_boxes, image_info = session.run(
        ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0', 'DetectionOuterBoxes:0', 'ImageInfo:0'],
        feed_dict={'Placeholder:0': np_image_string})

    num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
    detection_boxes = np.squeeze(detection_boxes / min(image_info[0, 2]), axis=(0,))[0:num_detections]
    detection_outer_boxes = np.squeeze(detection_outer_boxes / min(image_info[0, 2]), axis=(0,))[0:num_detections]
    detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
    detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections]
    instance_masks = np.squeeze(detection_masks, axis=(0,))[0:num_detections]
    # Use outer boxes 
    ymin, xmin, ymax, xmax = np.split(detection_outer_boxes, 4, axis=-1)
    processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
    segmentations = coco_utils.generate_segmentation_from_masks(instance_masks, processed_boxes, height, width) 

    filter_arr = []
    for c in detection_classes:
        if c == 1:
            filter_arr.append(True)
        else:
            filter_arr.append(False)


    detection_boxes = detection_boxes[filter_arr]
    detection_classes= detection_classes[filter_arr]
    detection_scores = detection_scores[filter_arr]
    segmentations = segmentations[filter_arr]


    image = img.copy()
    image = cv2.rectangle(image, (int(width*0.75),int(height*0.5)), (width,int(height*0.9)), (0,0,255,255), -1) #filled rect 

    
    mask = cv2.bitwise_or(segmentations[0].reshape(height,width,1),segmentations[1].reshape(height,width,1))
    mask = cv2.convertScaleAbs(mask, alpha=255, beta=0)  

    person = cv2.bitwise_and(img, img, mask=mask) #mask of person
    inversion = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)) #picture w/ drawing w/o person
    masked = cv2.bitwise_or(person, inversion)

    #cv2.imwrite(output_image_path, masked)

    return cv2.imencode('.png', masked)[1].tobytes()



server_address = './uds_socket'
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

try:
    sock.connect(server_address)
except socket.error as e:
    print (e)
    sys.exit(1)


while True:
    try:
        # Send data
        data = sock.recv(16)  #recieve image size 
        print(data.decode())
        sock.sendall("got size".encode())
        
        # if str(data).startswith("Size"):
        imgSize = data.split()[1]
        data = sock.recv(int(imgSize.decode()))  #recieve img from server
        while len(data) < int(imgSize.decode()):
            data += sock.recv(int(imgSize.decode()))


        sock.sendall("got image".encode()) #send confirmation

        #
        #send annotated image size and bytes
        annImgBytes = mask_image(data)
        print(len(np.frombuffer(annImgBytes,dtype=np.uint8)))
        annImgSize = len(annImgBytes)

        msg = "Size %d" % annImgSize
        print(msg)
        sock.sendall(msg.encode())

        data = sock.recv(16) # confirmation that img size was recieved
        print(data.decode())

        sock.sendall(annImgBytes)

        data = sock.recv(16) # confirmation that img was recieved
        print(data.decode())
            


    finally:
        data = sock.recv(16)
        if(data.decode()=="close socket"):
            print ('closing socket')
            sock.close()
            break