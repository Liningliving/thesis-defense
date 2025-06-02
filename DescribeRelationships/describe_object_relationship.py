# SPDX-License-Identifier: AGPL-3.0-only

#Select top k frame todescribe an object 
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

import sys
sys.path.append('/root/autodl-tmp/AttieCode')
from Config.config import CONF
# from DataProcess.get_object_frame import get_object_frame
from Model.spatialBot2describe import describe_object, describe_relationship
# from Model.FLAN_t5_summarize import summarize_object_descriptions, summarize_relationship_descriptions
import pickle
# from PIL import Image
import numpy as np
from NLP.sentence_analyse import retrieve_info#, extract_relationship_with_dependency
from NLP.delete_image_list import delete_image_from_list
import json
from Model.kimi_summarize import extract_relationships_via_function
# import warnings
# warnings.simplefilter("ignore")


def construct_relationship2frame(object2frame): 
    object2frame_copy = object2frame.copy()
    relationship2frame = {}
    for idx_i,objectFrameInfo_i in zip( range( len(object2frame.items()) ), object2frame.items() ):
        # print(objectFrameInfo_i)
        i_frame_set = set()
        for frame in objectFrameInfo_i[1]:
            i_frame_set.add(frame[0])

        del object2frame_copy[objectFrameInfo_i[0]]
        for idx_j, objectFrameInfo_j in zip( range( len(object2frame_copy.items()) ), object2frame_copy.items() ):
            j_frame_set = set()
            for frame in objectFrameInfo_j[1]:
                j_frame_set.add(frame[0])
            
            shared_frames = i_frame_set.intersection(j_frame_set)
            if bool(shared_frames):
                key_string = str( objectFrameInfo_i[0])+','+ objectFrameInfo_i[1][0][5]\
                +','+str(objectFrameInfo_j[0])+',' + objectFrameInfo_j[1][0][5]
                if not key_string in relationship2frame:
                    relationship2frame[key_string] = []
                for shared_frame in shared_frames:
                    for frame in objectFrameInfo_i[1]:
                        if frame[0] == shared_frame:
                            bb_i = frame[3]  
                            break

                    for frame in objectFrameInfo_j[1]:
                        if frame[0] == shared_frame:
                            bb_j = frame[3] 
                            break
                    # print(bb_j)
                    # print("shared_frames",shared_frames )
                    relationship2frame[ key_string ].append((shared_frame, (bb_i, bb_j), \
                    np.concatenate((objectFrameInfo_j[1][0][4], objectFrameInfo_i[1][0][4]), axis=0)) )
    return relationship2frame

if __name__ == "__main__":
    views_root = os.path.join(CONF.PATH.R3SCAN_DATA_OUT, 'views')
    for view in os.listdir(views_root):
        if "0cac75b1-8d6f-2d13-8c17-9099db8915bc" not in view:
            continue
        scan_id = view.split('_')[0]

        main_dict_object = {}
        main_relationship_descriptions = {}
        with open( os.path.join(views_root, view), 'rb' ) as f:
            data = pickle.load(f)
           
            #Descibe a single object.
            for objectInfo in data.items():

                #select top_k frame:
                sorted_objectInfo = sorted(objectInfo[1], key=lambda x: x[1], reverse=True)
                
                if len(sorted_objectInfo) > CONF.DESCRIBE_TOPK:
                    object_frames = sorted_objectInfo[:CONF.DESCRIBE_TOPK]
                else:
                    object_frames = objectInfo[1]
                object_description_list = []  ; object_description_sentences = ""
                frame_num = len(objectInfo[1])
                for idx, object_in_frame in zip( range( frame_num ), object_frames ): 
                #objectInfo[0] is the local id of objects; 
                #objectInfo[1] is a list contain the info of the object in different frame.
                    #So, object_in_frame is a frame contain the qualified info about the object.
                    jpg_file_name_string = object_in_frame[0]
                    frame_id_string = jpg_file_name_string.split('.')[0] 
                    bb = object_in_frame[3]   #axis-aligned bounding box of the object. This is a tuple of 4 numbers.(xmin,ymin,xmax,ymax)
                    unique_mapping = object_in_frame[4]
                    object_label = object_in_frame[5]

                    #get the path of image
                    Depth_path = os.path.join(CONF.PATH.R3SCAN_RAW, scan_id, 'sequence', frame_id_string.zfill(6) + '.depth.pgm')
                    aligned_RGB_path = os.path.join( CONF.PATH.R3SCAN_DATA_OUT, 'aligned_RGB' , scan_id,  frame_id_string.zfill(6)
                                                    +'.aligned_rgb.jpg')

                    object_description = describe_object(aligned_RGB_path, Depth_path, bb,unique_mapping, object_label)
                    object_description_sentences += object_description
                    # f.write( image_description )
                    # object_description_list.append(image_description)
                    # print(object_description_list)

                    if idx ==  frame_num-1:
                        # summary = summarize_object_descriptions(object_description_sentences, object_label)
                        # i = retrieve_info(summary)
                        info = extract_relationships_via_function(object_description_sentences)
                        info = delete_image_from_list(info)
                        info_dict = {f"id:{objectInfo[0]}":info}
                        main_dict_object.update(info_dict)
                        
 

            #dealing relationship in a scene
            relationship2frame = construct_relationship2frame(data)
            relationships_info = ""
            for relationship in  relationship2frame.items():
                obj0_id, obj0_label, obj1_id, obj1_label = relationship[0].split(',')
                relationship_description_dict = { obj0_id + ',' + obj0_label +',' + obj1_id +',' + obj1_label:[],\
                                                  obj1_id + ',' +obj1_label +',' + obj0_id + ',' + obj0_label:[], "non_directional": []}
                frame_size = len(relationship[1])
                relationship_descriptions = ""

                #select top_k frames:
                sorted_objectInfo = sorted(relationship[1], key=lambda x: x[1], reverse=True)
                
                if len(sorted_objectInfo) > CONF.DESCRIBE_TOPK:
                    relationship_frames = sorted_objectInfo[:CONF.DESCRIBE_TOPK]
                else:
                    relationship_frames = relationship[1]

                for idx, frame in zip( range( frame_size ), relationship_frames ):
                    #frame is a tuple contain ('frame_id', (bb), 2D_points_nparray)
                    jpg_file_name_string = frame[0]
                    frame_id_string = jpg_file_name_string.split('.')[0] 
                    aabbs = frame[1]   #axis-aligned bounding box of the object. This is a tuple of 4 numbers.(xmin,ymin,xmax,ymax)
                    points = frame[2]

                    #get the path of image
                    Depth_path = os.path.join(CONF.PATH.R3SCAN_RAW, scan_id, 'sequence', frame_id_string.zfill(6) + '.depth.pgm')
                    aligned_RGB_path = os.path.join( CONF.PATH.R3SCAN_DATA_OUT, 'aligned_RGB' , scan_id,  frame_id_string.zfill(6)
                                                    +'.aligned_rgb.jpg')

                    relationship_description = describe_relationship(aligned_RGB_path, Depth_path, aabbs, points, relationship[0])
                    relationship_descriptions += relationship_description

                    #delete same relationship description:
                    if idx == frame_size-1:
                        # summary = summarize_relationship_descriptions(relationship_descriptions, obj0_label+","+obj1_label)
                        info = extract_relationships_via_function(relationship_descriptions)
                        info = delete_image_from_list(info)
                        info_dict = {f"{relationship[0]}:":info}
                        main_relationship_descriptions.update(info_dict)


        #create a path for store object files.
        if not os.path.isdir(os.path.join(CONF.PATH.R3SCAN_DATA_OUT, "object") ):
            os.makedirs( os.path.join(CONF.PATH.R3SCAN_DATA_OUT, "object") )
        #save the object_info of a scene in json file
        with open( os.path.join(CONF.PATH.R3SCAN_DATA_OUT, "object", scan_id+'.json'),'w') as f:
            json.dump( main_dict_object, f,  indent=4)
        # delete_image_from_lists(os.path.join(CONF.PATH.R3SCAN_DATA_OUT, "object", scan_id+'.json'))
        
        #create a path for store relationship files.
        if not os.path.isdir(os.path.join(CONF.PATH.R3SCAN_DATA_OUT, "relationship") ):
            os.makedirs( os.path.join(CONF.PATH.R3SCAN_DATA_OUT, "relationship") )
        #save the object_info of a scene in json file
        # Replace single quotes with double quotes to make it a valid JSON string
        with open( os.path.join(CONF.PATH.R3SCAN_DATA_OUT, "relationship", scan_id+'.json'),'w') as f:
            json.dump( main_relationship_descriptions, f,  indent=4)
        delete_image_from_list(os.path.join(CONF.PATH.R3SCAN_DATA_OUT, "relationship", scan_id+'.json'))