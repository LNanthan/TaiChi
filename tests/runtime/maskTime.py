
import socket
import sys
import os
import cv2

import numpy as np
import time
import csv

from IPython import display
from PIL import Image
import tensorflow.compat.v1 as tf
import sys

curr_dir = os.getcwd()
os.chdir('..')
os.chdir('..')
curr_dir = os.getcwd()


sys.path.insert(0, curr_dir+'/tpu/models/official')
sys.path.insert(0, curr_dir+'/tpu/models/official/detection')

from evaluation import coco_utils
from google.cloud import storage

#record: overall time, time to run shapemask, process model output, create drawing mask
#overall time includes time taken to recieve and send data between sockets

f = open('tests/runtime/timeData/mask', 'w+')
writer = csv.writer(f)
header = ['run shapemask', 'format & filter model output', 'create drawing mask', 'total time per frame']
writer.writerow(header)

row = []


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
session = tf.Session(graph=tf.Graph(),config=tf.ConfigProto(gpu_options=gpu_options))


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
_ = tf.saved_model.load(session,['serve'] ,saved_model_dir)
print(tf.config.list_physical_devices('GPU'))


def mask_image (imageBytes):
    imgArr = np.frombuffer(imageBytes,dtype=np.uint8)
    img = cv2.imdecode(imgArr,cv2.IMREAD_UNCHANGED)

    np_image_string = np.array([imageBytes])
    height, width = img.shape[0:2]

    start_mod = time.time()

    print("running model")
    num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, detection_outer_boxes, image_info = session.run(
        ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0', 'DetectionOuterBoxes:0', 'ImageInfo:0'],
        feed_dict={'Placeholder:0': np_image_string})
    row.append((time.time()-start_mod))

    start_format= time.time()

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
    for i in range(0,len(detection_classes)):
        if detection_classes[i] == 1 and detection_scores[i]>0.6:  #class 1 --> person
            filter_arr.append(True)
        else:
            filter_arr.append(False)

    segmentations = segmentations[filter_arr]   #segmentations is an array of 1's & 0's denoting where the person is in the img
    row.append((time.time()-start_format))
    
    start_drawing = time.time()
    #replace with clock
    drawing = np.zeros((height,width),dtype = np.uint8) #black background
    drawing = cv2.rectangle(drawing, (int(width*0.45),int(height*0.4)), (int(width*0.9),int(height*0.95)), 255, -1) #filled rect 

    person = cv2.bitwise_or(segmentations[0].reshape(height,width,1),segmentations[1].reshape(height,width,1))

    ##segmentations is represented by 1's & 0's but for bitwise_or need 0XFF --> scale by 255
    person = cv2.convertScaleAbs(person, alpha=255, beta=0)  
    
    mask = cv2.subtract(drawing,person)
    row.append((time.time()-start_drawing))

    return cv2.imencode('.png', mask)[1].tobytes()


server_address = './uds_server'
mask_address = './uds_mask'

try:
    os.unlink(mask_address)
except OSError:
    if os.path.exists(mask_address):
        raise

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(mask_address)

close = False
try:
    sock.connect(server_address)
    print('connected')
except socket.error as e:
    print (e)
    sys.exit(1)


while True:
    try:
        start = time.time()
         # Recieve size then image from server
        data = sock.recv(16)
        if(data.decode() == 'close socket....'):
            print('recv close socket')
            close = True
            break
        f_num_msg = data
        f_num = f_num_msg.decode()
        while(len(f_num)>0 and f_num[0]=='0'):
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

        annImgBytes = mask_image(frame)

        #send back to server(which sends to render) using this format:
        # sizeOfRenderMsg(8) renderMsg(sizeOfRenderMsg)
        #renderMsg: frameNum(16) sizeOfData(8) data(sizeOfData)  *character id should be prepended to data
    
        renderMsg = f_num_msg
        sizeData = str(len(annImgBytes))
        while(len(sizeData)<8):
            sizeData ='0' + sizeData
        renderMsg+= sizeData.encode()
        renderMsg+='m'.encode()
        renderMsg += annImgBytes

        sizeRenderMsg = str(len(renderMsg))
        while(len(sizeRenderMsg)<8):
            sizeRenderMsg ='0' + sizeRenderMsg

        sock.sendall(sizeRenderMsg.encode())
        sock.sendall(renderMsg)
        row.append((time.time()-start))
        writer.writerow(row)
        row = []


    finally:
        if(close==True):
            print ('closing socket')
            sock.close()
            f.close()
            break