# SPDX-License-Identifier: AGPL-3.0-only

import os
import sys
import path
from easydict import EasyDict

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/root/autodl-tmp/"  # Base directory
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE , "data")  # Root path for datasets
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE , "models")
# sys.path.extend([CONF.PATH.HOME, CONF.PATH.BASE, CONF.PATH.DATA] )

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# Original Datasets
CONF.PATH.R3SCAN_RAW = os.path.join(CONF.PATH.DATA, "3RScan4small_test")  # 3RScan original dataset directory

CONF.PATH.R3SCAN_DATA_OUT = os.path.join(CONF.PATH.DATA, "3RScan4small_test_dataout")  # Output directory for processed datasets
# CONF.PATH.R3SCAN = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_3RScan") 

# for _, path in CONF.PATH.items():
#     print(path)
#     assert os.path.exists(path), f"{path} does not exist"

#paras

CONF.RGB_EMBEDDING = EasyDict()
CONF.RGB_EMBEDDING.USING_VGG = True
if CONF.RGB_EMBEDDING.USING_VGG:
    CONF.RGB_EMBEDDING.USING_BLIP2 = False
else:
    CONF.RGB_EMBEDDING.USING_BLIP2 = True
# CONF.RGB_EMBEDDING.SAVE_AS_CSV = True

CONF.RGB_EMBEDDING.LAPLACIAN_VARIANCE_THRESHOLD = 200
CONF.RGB_EMBEDDING.SIMILARITY_THERSHOLD = 0.7
CONF.RGB_EMBEDDING.K = 5

CONF.DESCRIBE_TOPK = 5