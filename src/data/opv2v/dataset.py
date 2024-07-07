import math
from collections import OrderedDict
import cv2
import numpy as np
import torch
import icecream as ic
import os
import pickle as pkl
from torch.utils.data import Dataset
from tqdm import tqdm
import yaml
from torchvision.transforms.functional import to_tensor
import torch
import matplotlib.pyplot as plt
import skimage


from src.data.opv2v.dataset_utils import *
from src.utils.configs import get_default_configuration 
from src.data.opv2v.utils import x_to_world, get_car_bboxs, render_polygon, x1_to_x2
from src.models.model_factory import build_model, build_criterion




class Opv2vDataset(Dataset):
    """
    Dataset used for single camera training
    """
    def __init__(self, labels_root, dataset_dir="/data/v2x/OPV2V", split="train", image_size=[960, 600], return_camera_path=False):
        self.labels_root = labels_root
        self.image_size = image_size

        self.return_camera_path = return_camera_path

        self.split = split.lower()
        
        self.scenario_database = OrderedDict()      

        self.images_paths = []
        self.yaml_paths = []
        self.labels_paths = []

        self.data_path = dataset_dir
        data_info_path = "utility_files/data_info.pkl"
        self.data_info = load_data_info(data_info_path)

        for scene in self.data_info[self.split].keys():
            for cav in self.data_info[self.split][scene].keys():
                for timestamp in self.data_info[self.split][scene][cav].keys():

                    yaml_path = self.data_info[self.split][scene][cav][timestamp]["yaml"]
                    cameras_path = sorted(self.data_info[self.split][scene][cav][timestamp]["cameras"])

                    for camera_path in cameras_path:
                        camera_name = camera_path[-11:-4]

                        self.images_paths.append(camera_path)   
                        self.yaml_paths.append(yaml_path)
                        labels_path = os.path.join(self.labels_root, self.split, scene, cav, camera_name, f"{timestamp}_{camera_name}.png")
                        self.labels_paths.append(labels_path)

    def __len__(self):
        return len(self.images_paths)


    def __getitem__(self, idx):
        yaml_path = self.yaml_paths[idx]
        labels_path = self.labels_paths[idx]
        camera_path = self.images_paths[idx]
        image = self.load_image(camera_path)

        calib = self.load_calib(yaml_path, camera_path[-11:-4])

        labels, mask = self.load_labels(labels_path)
        if self.return_camera_path:
            return image, calib, labels, mask, camera_path
        else:
            return image, calib, labels, mask

    def load_image(self, camera_path):
        image = Image.open(camera_path)
        image = image.resize(self.image_size)
        return to_tensor(image)
        
    def load_calib(self, yaml_path, camera):
        with open(yaml_path) as file:
            yaml_file = yaml.safe_load(file)

        calib = np.array(yaml_file[camera]["intrinsic"])
        calib[0] *= self.image_size[0] / 800
        calib[1] *= self.image_size[1] / 600


        return torch.from_numpy(calib)


    def load_labels(self, labels_path):
        encoded_labels = to_tensor(Image.open(labels_path)).long()

        labels = decode_binary_labels(encoded_labels, 2)
        labels, mask = labels[:-1], ~labels[-1]
        return labels, mask



class CollabIntermidiateFusionOpv2vDataset(Dataset):
    """
    Dataset used for intermidiate fusion training
    """
    def __init__(self, labels_root, config,dataset_dir="/data/v2x/OPV2V", split="train", image_size=[960, 600]):
        
        self.config = config

        self.labels_root = labels_root
        self.image_size = image_size

        self.split = split.lower()
        
        self.scenario_database = OrderedDict()      

        self.images_paths = []
        self.yaml_paths = []
        self.labels_paths = []

        self.data_path = dataset_dir
        data_info_path = "utility_files/data_info.pkl"
        self.data_info = load_data_info(data_info_path)
        self.pairs_scene_ego_timestamp = []


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        for scene in self.data_info[self.split].keys():
            scene_dict = OrderedDict()
            vehicles = list(self.data_info[self.split][scene].keys()) 

            if len(vehicles) > 1:
                vehicles = vehicles[:2]

            ego = vehicles[0]

            ego_dict = OrderedDict()
            for timestamp in self.data_info[split][scene][ego].keys():
                self.pairs_scene_ego_timestamp.append((scene, ego, timestamp))
                timestamp_dict = OrderedDict()
                for cur_vehicle in vehicles:
                    vehicle_dict = OrderedDict()
                    if timestamp in self.data_info[self.split][scene][cur_vehicle].keys():
                        vehicle_dict["yaml"] = self.data_info[self.split][scene][cur_vehicle][timestamp]["yaml"]
                        vehicle_dict["cameras"] = sorted(self.data_info[self.split][scene][cur_vehicle][timestamp]["cameras"])
                        labels = []
                        for camera_path in vehicle_dict["cameras"]:
                            camera_name = camera_path[-11:-4]
                            labels_path = os.path.join(self.labels_root, self.split, scene, cur_vehicle, camera_name, f"{timestamp}_{camera_name}.png")
                            labels.append(labels_path)
                        vehicle_dict["labels"] = labels
                        timestamp_dict[cur_vehicle] = vehicle_dict

                    ego_dict[timestamp] = timestamp_dict
                scene_dict[ego] = ego_dict
            self.scenario_database[scene] = scene_dict




    def __len__(self):
        return len(self.pairs_scene_ego_timestamp)


    def __getitem__(self, idx):
        scene, ego, timestamp = self.pairs_scene_ego_timestamp[idx]

        map_extents = self.config.map_extents
        map_resolution = self.config.map_resolution

        x1, y1, x2, y2 = map_extents
        cam_len = (y2 - y1)
        scene_len = cam_len * 4
        size_side = int(scene_len / map_resolution)

        full_labels = np.zeros((1, size_side, size_side), dtype=np.uint8)
        full_mask = np.zeros((size_side, size_side), dtype=np.uint8)
        cam_images = []
        cam_positions = []
        cam_calibs = []

        with open(self.scenario_database[scene][ego][timestamp][ego]["yaml"]) as file:
            ego_yaml_file = yaml.safe_load(file)
        
        ego_position = ego_yaml_file["predicted_ego_pos"]

        ego_to_world = x_to_world(ego_position)
        world_to_ego = np.linalg.inv(ego_to_world)


        for vehicle in self.scenario_database[scene][ego][timestamp].keys():
            with open(self.scenario_database[scene][ego][timestamp][vehicle]["yaml"]) as file:
                yaml_file = yaml.safe_load(file)

            vehicle_position = yaml_file["predicted_ego_pos"]
            distance = np.sqrt((ego_position[0]-vehicle_position[0])**2 + (ego_position[1]-vehicle_position[1])**2)

            if distance <= cam_len + cam_len/2:

                # BBoxes
                bboxs = np.array(get_car_bboxs(self.config, yaml_file))
                if len(bboxs) > 0:

                    bb = np.concatenate((bboxs, np.ones((bboxs.shape[0],bboxs.shape[1], 1))), -1)
                    
                    for bbox in bb:

                        bbox = bbox.T
                        bbox_ego = world_to_ego @ bbox
                        bbox_ego = bbox_ego[:2, :4]
                        bbox_ego += (scene_len/2)
                        bbox_ego_tmp = bbox_ego.copy()
                        bbox_ego[0, :] = bbox_ego_tmp[1, :]
                        bbox_ego[1, :] = bbox_ego_tmp[0, :]
                        render_polygon(full_labels, bbox_ego.T, map_extents, map_resolution, size_side) 

                # Images and Mask
                for idx in range(len(self.scenario_database[scene][ego][timestamp][vehicle]["labels"])):
                    calib = torch.tensor(ego_yaml_file[f"camera{idx}"]["intrinsic"])

                    camera_path = self.scenario_database[scene][ego][timestamp][vehicle]["cameras"][idx]

                    camera_image = self.load_image(camera_path)
                    camera_position = yaml_file[f"camera{idx}"]["cords"]
                    
                    cam_calibs.append(calib)
                    cam_positions.append(camera_position)
                    cam_images.append(camera_image)
                    
                    labels_path = self.scenario_database[scene][ego][timestamp][vehicle]["labels"][idx]

                    labels, mask = self.load_labels(labels_path)

                    cam_to_world = x_to_world(yaml_file[f"camera{idx}"]["cords"])

                    y, x = np.arange(y1, y2, map_resolution), np.arange(x1, x2, map_resolution)

                    true_ids = np.where(mask>0)
                    true_x = x[true_ids[0]]
                    true_y = y[true_ids[1]]
                    
                    if len(true_ids[0]) > 0:
                        mask_coordinates = np.array([[true_x[i], true_y[i], 0, 1] 
                            for i in range(len(true_x))]).T
                        mask_coordinates_world = cam_to_world @ mask_coordinates

                        mask_coordinates_ego = world_to_ego @ mask_coordinates_world

                        for p_idx in range(mask_coordinates_ego.shape[1]):
                                point = mask_coordinates_ego[:, p_idx].copy()

                                p_x = point[1]
                                p_y = point[0]
                                p_x = round((p_x+(scene_len/2)) / map_resolution)
                                p_y = round((p_y+(scene_len/2)) / map_resolution)

                                if (p_x>=0) and (p_x<size_side) and (p_y>=0) and (p_y<size_side):
                                    full_mask[p_y, p_x] = 1

        for x in range(full_mask.shape[0]):
            for y in range(full_mask.shape[0]):
                if x>0 and x<size_side-1 and y>0 and y<size_side-1:
                    if full_mask[x][y]==0 and full_mask[x-1][y]==1 \
                        and full_mask[x+1][y]==1 and full_mask[x][y-1]==1 \
                        and full_mask[x][y+1]==1:
                        full_mask[x][y] = 1

        cam_images = torch.stack(cam_images)
        cam_positions = torch.from_numpy(np.array(cam_positions))
        cam_calibs = torch.stack(cam_calibs)
        ego_position = torch.from_numpy(np.array(ego_position))
        full_labels = torch.from_numpy(full_labels)
        full_mask = torch.from_numpy(full_mask)

        return cam_images, cam_calibs, cam_positions, ego_position, full_labels, full_mask

    def load_image(self, camera_path):
        image = Image.open(camera_path)
        image = image.resize(self.image_size)
        return to_tensor(image)

    def load_labels(self, labels_path):
        encoded_labels = to_tensor(Image.open(labels_path)).long()

        labels = decode_binary_labels(encoded_labels, 2)
        
        labels, mask = labels[:-1], ~labels[-1]
        return labels, mask


class CollabLateFusionOpv2vDataset(Dataset):
    """
    Dataset used for obtaining late fusion predicions
    """
    def __init__(self, labels_root, config,dataset_dir="/data/v2x/OPV2V", split="train", image_size=[960, 600]):

        self.config = config

        self.labels_root = labels_root
        self.image_size = image_size

        self.split = split.lower()
        
        self.scenario_database = OrderedDict()      

        self.images_paths = []
        self.yaml_paths = []
        self.labels_paths = []

        self.data_path = dataset_dir
        data_info_path = "utility_files/data_info.pkl"
        self.data_info = load_data_info(data_info_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pairs_scene_ego_timestamp = []
        

        self.model = build_model(self.config.model, self.config)

        # Path with single camera pretrained model
        state_dict = torch.load("model.pth", map_location=self.device)
        self.model.load_state_dict(state_dict["model"])

        for scene in self.data_info[self.split].keys():
            scene_dict = OrderedDict()
            vehicles = list(self.data_info[self.split][scene].keys()) 

            if len(vehicles) > 1:
                vehicles = vehicles[:2]

            ego = vehicles[0]

            ego_dict = OrderedDict()
            for timestamp in self.data_info[split][scene][ego].keys():
                self.pairs_scene_ego_timestamp.append((scene, ego, timestamp))
                timestamp_dict = OrderedDict()
                for cur_vehicle in vehicles:
                    vehicle_dict = OrderedDict()
                    if timestamp in self.data_info[self.split][scene][cur_vehicle].keys():
                        vehicle_dict["yaml"] = self.data_info[self.split][scene][cur_vehicle][timestamp]["yaml"]
                        vehicle_dict["cameras"] = sorted(self.data_info[self.split][scene][cur_vehicle][timestamp]["cameras"])
                        labels = []
                        for camera_path in vehicle_dict["cameras"]:
                            camera_name = camera_path[-11:-4]
                            labels_path = os.path.join(self.labels_root, self.split, scene, cur_vehicle, camera_name, f"{timestamp}_{camera_name}.png")
                            labels.append(labels_path)
                        vehicle_dict["labels"] = labels
                        timestamp_dict[cur_vehicle] = vehicle_dict

                    ego_dict[timestamp] = timestamp_dict
                scene_dict[ego] = ego_dict
            self.scenario_database[scene] = scene_dict


    def __len__(self):
        return len(self.pairs_scene_ego_timestamp)
 


    def __getitem__(self, idx):
        scene, ego, timestamp = self.pairs_scene_ego_timestamp[idx]
        print(scene, ego, timestamp)
        map_extents = self.config.map_extents
        map_resolution = self.config.map_resolution

        x1, y1, x2, y2 = map_extents

        cam_len = (y2 - y1)
        scene_len = cam_len * 4
        size_side = int(scene_len / map_resolution)


        full_labels = torch.zeros((1, size_side, size_side), device=self.device)
        full_logits = torch.zeros((size_side, size_side), device=self.device)
        full_mask = torch.zeros((size_side, size_side), device=self.device)

        with open(self.scenario_database[scene][ego][timestamp][ego]["yaml"]) as file:
            ego_yaml_file = yaml.safe_load(file)

        ego_position = torch.tensor(ego_yaml_file["predicted_ego_pos"]).to(self.device)

        ego_to_world = torch.from_numpy(x_to_world(ego_position.cpu())).to(self.device)
        world_to_ego = torch.linalg.inv(ego_to_world).to(self.device)

        for vehicle in self.scenario_database[scene][ego][timestamp].keys():
            with open(self.scenario_database[scene][ego][timestamp][vehicle]["yaml"]) as file:
                yaml_file = yaml.safe_load(file)

            vehicle_position = torch.tensor(yaml_file["predicted_ego_pos"]).to(self.device)
            distance = torch.sqrt((ego_position[0]-vehicle_position[0])**2 + (ego_position[1]-vehicle_position[1])**2)

            if distance <= cam_len + cam_len/2:

                # BBoxes
                bboxs = np.array(get_car_bboxs(self.config, yaml_file))
                if len(bboxs) > 0:

                    bb = torch.from_numpy(np.concatenate((bboxs, np.ones((bboxs.shape[0],bboxs.shape[1], 1))), -1)).to(self.device)
                    
                    for bbox in bb:
                        
                        bbox = bbox.T
                        bbox_ego = world_to_ego @ bbox
                        bbox_ego = bbox_ego[:2, :4]
                        bbox_ego += (scene_len/2)
                        bbox_ego_tmp = bbox_ego.clone()
                        bbox_ego[0, :] = bbox_ego_tmp[1, :]
                        bbox_ego[1, :] = bbox_ego_tmp[0, :]
                        render_polygon(full_labels, bbox_ego.T.cpu().numpy(), map_extents, map_resolution, size_side) 

                for idx in range(len(self.scenario_database[scene][ego][timestamp][vehicle]["labels"])):
                    camera_path = self.scenario_database[scene][ego][timestamp][vehicle]["cameras"][idx]


                    calib = torch.unsqueeze(torch.tensor(yaml_file[f"camera{idx}"]["intrinsic"]), axis=0).to(self.device)
                    image = torch.unsqueeze(self.load_image(camera_path), axis=0).to(self.device)


                    labels_path = self.scenario_database[scene][ego][timestamp][vehicle]["labels"][idx]

                    labels, mask = self.load_labels(labels_path)
                    mask = mask.to(self.device)

                    cam_to_world = torch.from_numpy(x_to_world(yaml_file[f"camera{idx}"]["cords"])).to(self.device)


                    coordinates = torch.tensor([[x, y, 0, 1] for x in torch.arange(x1, x2, map_resolution) for y in torch.arange(y1, y2, map_resolution)], dtype=torch.double).T.to(self.device)
                    coordinates_world = cam_to_world @ coordinates
                    coordinates_ego = world_to_ego @ coordinates_world
                    coordinates_ego = (coordinates_ego + (scene_len/2))/map_resolution
                    coordinates_ego = torch.round(coordinates_ego).to(torch.int)
                    ix = (coordinates_ego[0] >= 0) & (coordinates_ego[0] < size_side) & (coordinates_ego[1] >= 0) & (coordinates_ego[1] < size_side)
                    coordinates_ego = coordinates_ego[:, ix]
                    ids = torch.stack(torch.meshgrid(torch.arange(200), torch.arange(196), indexing='xy')).reshape(2, -1).to(self.device)

                    ids = ids[:, ix]
                    full_mask[coordinates_ego[0], coordinates_ego[1]] += torch.tensor(mask[ids[1], ids[0]])
                    with torch.no_grad():
                        logits = self.model(image, calib)

                    threshold = 0.5
                    logits[logits < threshold] = 0
                    logits[logits >= threshold] = 1

                    full_logits[coordinates_ego[0], coordinates_ego[1]] += logits[:, :, ids[1], ids[0]].flatten()


        full_logits[full_logits>=1] = 1
        full_mask[full_mask>=1] = 1

        full_mask = full_mask.to("cpu")
        full_logits = full_logits.to("cpu")
        full_labels = full_labels.to("cpu")


        for x in range(full_mask.shape[0]):
            for y in range(full_mask.shape[0]):
                if x>0 and x<size_side-1 and y>0 and y<size_side-1:
                    if full_mask[x][y]==0 and full_mask[x-1][y]==1 \
                        and full_mask[x+1][y]==1 and full_mask[x][y-1]==1 \
                        and full_mask[x][y+1]==1:
                        full_mask[x][y] = 1
                    if full_logits[x][y]==0 and full_logits[x-1][y]==1 \
                        and full_logits[x+1][y]==1 and full_logits[x][y-1]==1 \
                        and full_logits[x][y+1]==1:
                        full_logits[x][y] = 1

        return full_labels, full_logits, full_mask

    def load_image(self, camera_path):
        image = Image.open(camera_path)
        image = image.resize(self.image_size)
        return to_tensor(image)

    def load_labels(self, labels_path):
        encoded_labels = to_tensor(Image.open(labels_path)).long()

        labels = decode_binary_labels(encoded_labels, 2)
        
        labels, mask = labels[:-1], ~labels[-1]
        return labels, mask


