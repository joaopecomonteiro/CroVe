import os

from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch

import numpy as np

from .augmentation import AugmentedMapDataset


from .opv2v.dataset import Opv2vDataset, CollabIntermidiateFusionOpv2vDataset


def build_opv2v_datasets(config):
    print('==> Loading OPV2V dataset...')
    dataroot = os.path.expandvars(config.dataroot)
    if config.fusion == "late":
        train_data = Opv2vDataset(config.label_root, dataroot, "train")
        val_data = Opv2vDataset(config.label_root, dataroot, "validate")
    elif config.fusion == "intermidiate":
        train_data = CollabIntermidiateFusionOpv2vDataset(config.label_root, split="train", config=config)
        val_data = CollabIntermidiateFusionOpv2vDataset(config.label_root, split="validate", config=config)
    return train_data, val_data


def build_datasets(dataset_name, config):
    if dataset_name == "opv2v":
        return build_opv2v_datasets(config)
    else:
        raise ValueError(f"Unknown dataset option '{dataset_name}'")



def build_trainval_datasets(dataset_name, config):

    # Construct the base dataset
    train_data, val_data = build_datasets(dataset_name, config)

    # Add data augmentation to train dataset
    #train_data = AugmentedMapDataset(train_data, config.hflip, config.fusion)

    return train_data, val_data


def custom_collate(batch):
    # Extract individual elements from the batch
    cam_images, cam_calibs, cam_positions, ego_position, full_labels, full_mask = zip(*batch)
    

    # Pad cam_images to ensure they have the same shape within a batch
    padded_cam_images = pad_sequence(cam_images, batch_first=True, padding_value=0)
    
    # Pad cam_calibs to ensure they have the same shape within a batch
    padded_cam_calibs = pad_sequence(cam_calibs, batch_first=True, padding_value=0)
    
    # Pad cam_positions to ensure they have the same shape within a batch
    padded_cam_positions = pad_sequence(cam_positions, batch_first=True, padding_value=0)
    
    # Convert other elements to tensors
    ego_position_tensor = torch.from_numpy(np.array(ego_position))
    full_labels_tensor = torch.from_numpy(np.array(full_labels))
    full_mask_tensor = torch.from_numpy(np.array(full_mask))
    
    return padded_cam_images, padded_cam_calibs, padded_cam_positions, ego_position_tensor, full_labels_tensor, full_mask_tensor





def build_dataloaders(dataset_name, config, batch_size=None):
    if batch_size is None:
        batch_size = config.batch_size
    
    # Build training and validation datasets
    train_data, val_data = build_trainval_datasets(dataset_name, config)


    if config.fusion == "late":
        train_loader = DataLoader(train_data, batch_size,  shuffle=True,
                                num_workers=config.num_workers)

        # Create validation dataloader
        val_loader = DataLoader(val_data, batch_size, 
                                num_workers=config.num_workers)
    
    elif config.fusion == "intermidiate":
        train_loader = DataLoader(train_data, batch_size, shuffle=True,
                            num_workers=config.num_workers, 
                            collate_fn=custom_collate)

        # Create validation dataloader
        val_loader = DataLoader(val_data, batch_size, 
                                num_workers=config.num_workers, 
                                collate_fn=custom_collate)
    

    return train_loader, val_loader

    


    

