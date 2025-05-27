# This source code is from 3DSSG
#   (https://github.com/ShunChengWu/3DSSG/tree/cvpr21)
# Copyright (c) 2021 3DSSG authors
# This source code is licensed under the BSD 2-Clause found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import json
import random
import numpy as np
# from open3dsg.util import util_label
# from open3dsg.util.util_label import scannet_label_ids, scannet_3rscan_label_mapping


def rand_24_bit():
    """Returns a random 24-bit integer"""
    return random.randrange(0, 16**6)


def color_dec():
    """Alias of rand_24 bit()"""
    return rand_24_bit()


def color_hex(num=rand_24_bit()):
    """Returns a 24-bit int in hex"""
    return "%06x" % num


def color_rgb(num=rand_24_bit()):
    """Returns three 8-bit numbers, one for each channel in RGB"""
    hx = color_hex(num)
    barr = bytearray.fromhex(hx)
    return (barr[0], barr[1], barr[2])


def set_random_seed(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_file_exist(path):
    if not os.path.exists(path):
        raise RuntimeError('Cannot open file. (', path, ')')


def read_txt_to_list(file):
    output = []
    with open(file, 'r') as f:
        for line in f:
            entry = line.rstrip().lower()
            output.append(entry)
    return output


def read_relationships(read_file):
    relationships = []
    with open(read_file, 'r') as f:
        for line in f:
            relationship = line.rstrip().lower()
            relationships.append(relationship)
    return relationships

#Added by Attie
def pgmread_p5(filename, dtype=np.uint16):
    """Reads a PGM image file and returns a numpy array."""
    with open(filename, 'rb') as f:
        width, height = 0, 0
        magicNum = ''
        maxVal = 0
        count = 0

        # Read header lines, ignoring comments
        while count < 3:
            line = f.readline().decode('ascii', errors='ignore').strip()
            print(line)
            if line.startswith('#'):
                continue
            count += 1
            if count == 1:
                magicNum = line
                if magicNum != 'P5':
                    raise ValueError("Unsupported format (must be P5)")
            elif count == 2:
                width, height = map(int, line.split())
            elif count == 3:
                maxVal = int(line)

        # Skip any remaining whitespace after the header
        # f.read(1)  # Skip the newline after maxVal

        # Read pixel data
        pixel_data = np.fromfile(f, dtype, count=width * height)
        pixel_data = pixel_data.reshape((height, width))

    return np.array(pixel_data)