import pickle
import os
import sys
import numpy as np
from PIL import Image
import cv2

# print(sys.path)
# sys.path.append('/home/lin/Documents/defense_project/AttieCode')
from Config.config import CONF
# from utils.read_pgm import pgmread

image_width, image_height = 960, 540

scan_id = '0ad2d3a3-79e2-2212-9a51-9094be707ec2'
views_root = os.path.join(CONF.PATH.R3SCAN_DATA_OUT, "views")

#Open a scene folder
with open( os.path.join(views_root, scan_id+'_object2image.pkl'), 'rb' ) as f:
    data = pickle.load(f)
    for objectInfo in data.items():  
        for object_in_frame in objectInfo[1]:  
        #objectInfo[0] is the local id of objects; 
        #objectInfo[1] is a list contain the info of the object in different frame.
            #So, object_in_frame is a frame contain the qualified info about the object.
            jpg_file_name_string = object_in_frame[0]
            frame_id_string = jpg_file_name_string.split('.')[0] 
            aabb = object_in_frame[3]   #axis-aligned bounding box of the object. This is a tuple of 4 numbers.(xmin,ymin,xmax,ymax)
            
            RGB_path = os.path.join(CONF.PATH.R3SCAN_RAW, scan_id, 'sequence', jpg_file_name_string)
            Depth_path = os.path.join(CONF.PATH.R3SCAN_RAW, scan_id, 'sequence', frame_id_string + '.depth.pgm')
            
            #resize depth image
            image_dim = np.array([image_width, image_height])
            depth = pgmread(Depth_path).reshape(-1)
            depth = depth.reshape(image_dim[::-1])/1000

            #visialize color image
            img = Image.open(RGB_path)
            img.show() 
            img = Image.open(Depth_path)
            img.show() 

            break
        break