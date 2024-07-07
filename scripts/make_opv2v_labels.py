import pdb
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
from PIL import Image
from progressbar import ProgressBar
import pickle
import yaml
import open3d as o3d
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from src.utils.configs import get_default_configuration
from src.data.utils import get_visible_mask, encode_binary_labels
from src.data.opv2v.utils import *


def load_data_info(path):
    with open(path, 'rb') as file:
        data_info = pickle.load(file)
    return data_info


def process_scene(split, scene, config, data_info):
    for vehicle in data_info.keys():
        process_vehicle(split, scene, vehicle, config, data_info[vehicle])


def process_vehicle(split, scene, vehicle, config, data_info):
    for timestamp in data_info.keys():
        process_timestamp(split, scene, vehicle,timestamp, config, data_info[timestamp])


def process_timestamp(split, scene, vehicle, timestamp, config, data_info):
    for camera_path in data_info["cameras"]:

        camera = camera_path[-11:-4]

        with open(data_info["yaml"]) as file:
            yaml_file = yaml.safe_load(file)

        masks = get_car_masks(split, scene, vehicle, timestamp, camera, config, data_info, yaml_file)

        intrinsic = np.array(yaml_file["camera0"]["intrinsic"])
        masks[-1] |= ~get_visible_mask(intrinsic, 800, config)
        

        labels = encode_binary_labels(masks)
        plt.imshow(np.flipud(labels))
        plt.show()
        pdb.set_trace()
        

        output_path = os.path.join(config.opv2v.label_root ,split, scene, vehicle, camera, f'{timestamp}_{camera}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)
        
        
if __name__ == '__main__':

    config = get_default_configuration()
    config.merge_from_file('configs/datasets/opv2v.yml')

    data_info = load_data_info("utility_files/data_info.pkl")
    
    for split in tqdm(list(data_info.keys()), "split"):
        for scene in tqdm(list(data_info[split].keys()), "scene"):
            process_scene(split, scene, config, data_info[split][scene])























































