#unzip sequences, to save disk space. 

import sys
sys.path.append('/home/lin/Documents/defense_project/AttieCode')

import os
import zipfile
from Config.config import CONF
import tqdm 
import cv2

for folder in tqdm.tqdm( os.listdir(CONF.PATH.R3SCAN_RAW) ):
    current_folder = os.path.join(CONF.PATH.R3SCAN_RAW, folder)
    if os.path.isdir( current_folder ):
        for file_or_folder in os.listdir( current_folder ):
            if file_or_folder.endswith("sequence") and os.path.isdir( os.path.join( current_folder, file_or_folder ) ):
                # List all files in the directory
                sequence_folder = os.path.join( current_folder, file_or_folder )
                for filename in os.listdir(sequence_folder):
                    file_path = os.path.join(sequence_folder, filename)
                    
                    # Check if it is a file (not a subdirectory)
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # Remove the file
                        # print(f"Deleted file: {filename}")

                os.rmdir(os.path.join( current_folder, file_or_folder))
