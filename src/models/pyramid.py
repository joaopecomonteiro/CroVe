import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from time import time
import pickle


from src.data.opv2v.utils import x_to_world, x1_to_x2

class PyramidOccupancyNetwork(nn.Module):


    def __init__(self, frontend, transformer, topdown, classifier):
        super().__init__()


        self.frontend = frontend
        self.transformer = transformer
        self.topdown = topdown
        self.classifier = classifier
    

    def forward(self, image, calib, *args):

  
        # Extract multiscale feature maps
        feature_maps = self.frontend(image)

        # Transform image features to birds-eye-view
        bev_feats = self.transformer(feature_maps, calib)

        # Apply topdown network
        td_feats = self.topdown(bev_feats)

        # Predict individual class log-probabilities
        logits = self.classifier(td_feats)
        return logits
    


class PyramidOccupancyNetworkIntermidiateFusion(nn.Module):


    def __init__(self, frontend, transformer, topdown, classifier):
        super().__init__()


        self.frontend = frontend
        self.transformer = transformer
        self.topdown = topdown
        self.classifier = classifier

        # Read logits_coordinates from file so it is faster
        if torch.cuda.is_available():
            filename = 'logits_coordinates.pkl'
        else:
            filename = "logits_coordinates_cpu.pkl"
        
        with open(filename, 'rb') as f:
            self.logits_coordinates = pickle.load(f)

    def forward(self, cam_images, cam_calibs, cam_positions, ego_position, single_mask, map_extents, map_resolution):

        x1, y1, x2, y2 = map_extents
        scene_len = (y2 - y1) * 4
        size_side = int(scene_len / map_resolution)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_logits = torch.zeros((cam_images.shape[0], 1, size_side, size_side), device=device)

        for batch_id in range(cam_images.shape[0]):

            image = cam_images[batch_id].clone()
            calib = cam_calibs[batch_id].clone()
          
            for index in range(image.shape[0]):
                if torch.count_nonzero(image[index]) == 0:
                    break
                if index == image.shape[0]-1:
                    index += 1

            image = image[:index]        
            calib = calib[:index]

            feature_maps = self.frontend(image)
            bev_feats = self.transformer(feature_maps, calib)
            td_feats = self.topdown(bev_feats)

            full_td_feats = torch.zeros((256, size_side, size_side), device=device)
            times_mask = torch.zeros((size_side, size_side), device=device)

            for cam_id in range(image.shape[0]):
                
                logits_coordinates = self.logits_coordinates.clone()
                cam_to_ego = torch.from_numpy(x1_to_x2(cam_positions[batch_id, cam_id].cpu(), ego_position[batch_id].cpu())).float().to(device)

                logits_coordinates_ego = cam_to_ego @ logits_coordinates
                
                logits_coordinates_ego = torch.round((logits_coordinates_ego+(scene_len/2)) / map_resolution).long()
                ids = torch.stack(torch.meshgrid(torch.arange(200, device=device), torch.arange(196, device=device), indexing='ij')).reshape(2, -1)
                ids = ids[:, single_mask.flatten()]

                logits_coordinates_ego = logits_coordinates_ego[:, single_mask.flatten()]
               
                ix = (logits_coordinates_ego[0] >= 0) & (logits_coordinates_ego[0] < size_side) & (logits_coordinates_ego[1] >= 0) & (logits_coordinates_ego[1] < size_side)

                logits_coordinates_ego = logits_coordinates_ego[:, ix]

                ids = ids[:, ix]

                full_td_feats[:, logits_coordinates_ego[0], logits_coordinates_ego[1]] += td_feats[cam_id, :, ids[1], ids[0]]


                times_mask[logits_coordinates_ego[0], logits_coordinates_ego[1]] += 1
            full_td_feats[:, times_mask >= 1] /= times_mask[times_mask >= 1]

            logits = self.classifier(full_td_feats.to(device))

            batch_logits[batch_id, 0] = logits

        return batch_logits.to(device)     