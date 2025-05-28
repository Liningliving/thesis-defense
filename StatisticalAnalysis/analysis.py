# SPDX-License-Identifier: AGPL-3.0-only

import os
import sys
sys.path.append('/root/autodl-tmp/AttieCode')
import argparse
from utils.util_misc import read_txt_to_list
from Config.config import CONF
import json

def num_of_relationship(root, split="train", scan_id="02b33df9-be2b-2d54-9062-1253be3ce186"):
    """
    Reads a json file and returns the number of labelled relationships in the scan.
    """
    selected_scans = set()
    selected_scans = selected_scans.union(read_txt_to_list(os.path.join(CONF.PATH.R3SCAN_RAW, "3DSSG_subset", f'{split}_scans.txt')))
    # selected_scans = selected_scans.union(read_txt_to_list('/home/lin/Documents/defense_project/AttieCode/data/3RScan/3DSSG_subset/test_scans.txt'))
    with open(os.path.join(CONF.PATH.R3SCAN_RAW, "3DSSG_subset", f"relationships_{split}.json"), "r") as read_file:
    # with open("/home/lin/Documents/defense_project/data/3RScan/3DSSG_subset/relationships_train.json", "r") as read_file:
        data = json.load(read_file)

    # convert data to dict('scene_id': {'obj': [], 'rel': []})
    scene_data = dict()
    for i in data['scans']:
        if i['scan'] not in scene_data.keys():
            scene_data[i['scan']] = {'obj': dict(), 'rel': list()}
        scene_data[i['scan']]['obj'].update(i['objects'])
        scene_data[i['scan']]['rel'].extend(i['relationships'])

    return len( scene_data[scan_id]['rel'] )


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('--mode', type=str, default='test', choices=["train", "test", "validation"], help='train, test, validation')
    
    root = os.path.join(CONF.PATH.R3SCAN_RAW, "3DSSG_subset")
    print( num_of_relationship(root) )