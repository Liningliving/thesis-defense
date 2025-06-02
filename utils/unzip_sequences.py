# SPDX-License-Identifier: AGPL-3.0-only
#unzip sequences 

import sys
sys.path.append( '/home/aa/文档/AttieLin/AttieCode')
# sys.path.extend( '/home/aa/文档/AttieLin' )
print(sys.path)

import os
import zipfile
from Config.config import CONF
import tqdm 
import cv2

for folder in tqdm.tqdm( os.listdir(CONF.PATH.R3SCAN_RAW) ):
    current_folder = os.path.join(CONF.PATH.R3SCAN_RAW, folder)
    if os.path.isdir( current_folder ):
        for file in os.listdir( current_folder ):
            # print(file)
            if file.endswith("sequence.zip"):
                # print(folder)
                with zipfile.ZipFile( os.path.join( current_folder, file), 'r') as zip_ref:
                    zip_ref.extractall( os.path.join( current_folder, "sequence") )