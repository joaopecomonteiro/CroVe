import os
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch 
import pdb
from tqdm import tqdm
import pickle
import yaml



def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0


def load_camera_files(cav_path, timestamp):
    """
    Retrieve the paths to all camera files.

    Parameters
    ----------
    cav_path : str
        The full file path of current cav.

    timestamp : str
        Current timestamp

    Returns
    -------
    camera_files : list
        The list containing all camera png file paths.
    """
    camera0_file = os.path.join(cav_path,
                                timestamp + '_camera0.png')
    camera1_file = os.path.join(cav_path,
                                timestamp + '_camera1.png')
    camera2_file = os.path.join(cav_path,
                                timestamp + '_camera2.png')
    camera3_file = os.path.join(cav_path,
                                timestamp + '_camera3.png')

    cameras = [camera0_file, camera1_file, camera2_file, camera3_file]
    # Return list of existing camera files
    return [camera for camera in cameras if os.path.exists(camera)]

def load_camera_data(camera_files, preload=True):
    """
    Args:
        camera_files: list, 
            store camera path
        shape : tuple
            (width, height), resize the image, and overcoming the lazy loading.
    Returns:
        camera_data_list: list,
            list of Image, RGB order
    """
    camera_data_list = []
    for camera_file in camera_files:
        camera_data = np.array(Image.open(camera_file).convert('L'))
        if preload:
            camera_data = camera_data.copy()
        camera_data_list.append(camera_data)
    return camera_data_list

def load_data_info(path):
    with open(path, 'rb') as file:
        data_info = pickle.load(file)
    return data_info

def get_data_info(data_path):

    data_info = OrderedDict()
    stages = ["train", "validate"]
    for stage in stages:
        stage_dict = OrderedDict()
        scenarios = os.listdir(data_path+"/"+stage)
        for scenario in tqdm(scenarios):
            vehicles = os.listdir(data_path+"/"+stage+"/"+scenario)
            vehicles = [item for item in vehicles if os.path.isdir(data_path+"/"+stage+"/"+scenario+"/"+item)]
            scenario_dict = OrderedDict()

            for vehicle in vehicles:
                tmp_files = os.listdir(data_path+"/"+stage+"/"+scenario+"/"+vehicle)
                timestamps = []
                for tmp_file in tmp_files:
                    if tmp_file[:6] not in timestamps:
                        timestamps.append(tmp_file[:6])
                vehicle_dict = OrderedDict()

                for timestamp in timestamps:
                    if check_yaml(data_path+"/"+stage+"/"+scenario+"/"+vehicle+"/"+timestamp+".yaml"):
                        image_paths = load_camera_files(data_path+"/"+stage+"/"+scenario+"/"+vehicle, timestamp)
                        timestamp_dict = OrderedDict()
                        timestamp_dict["yaml"] = data_path+"/"+stage+"/"+scenario+"/"+vehicle+"/"+timestamp+".yaml"
                        timestamp_dict["lidar"] = data_path+"/"+stage+"/"+scenario+"/"+vehicle+"/"+timestamp+".pcd"
                        timestamp_dict["cameras"] = image_paths
                        vehicle_dict[timestamp] = timestamp_dict
                
                scenario_dict[vehicle] = vehicle_dict
            
            stage_dict[scenario] = scenario_dict
        data_info[stage] = stage_dict
    return data_info


def get_camera_name(path):
    return path.split('/')[-1].split('_')[-1].split('.')[0]


def get_camera_angle(x, y):
    return np.degrees(np.arccos((x**2-y**2)/(x**2+y**2)))



def get_valid_indices(lidar_points_camera, cam_id):
    valid_indices = []
    i = 0
    
    for point in lidar_points_camera:
        if point[0] >= 0:
            if (point[1] <= 400) and (point[1] >= -400):
                valid_indices.append(i)
            else:
                if point[1] < -400:
                    angle = get_camera_angle(point[0], point[1]+400)
                elif point[1] > 400:
                    angle = get_camera_angle(point[0], point[1]-400)
                if angle < 110:
                    valid_indices.append(i)
        i += 1
    
    return valid_indices


def check_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            # Try to load the YAML content
            yaml.safe_load(file)
            return True  # YAML file opened successfully
    except FileNotFoundError:
        return False  # File not found
    except Exception:
        return False 