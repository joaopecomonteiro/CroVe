import os
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler

from src.models.model_factory import build_model, build_criterion
from src.data.data_factory import build_dataloaders, custom_collate
from src.utils.configs import get_default_configuration, load_config
from src.utils.confusion import BinaryConfusionMatrix
# from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES
# from src.data.argoverse.utils import ARGOVERSE_CLASS_NAMES
from src.utils.visualise import colorise


from src.data.opv2v.dataset import *

parser = ArgumentParser()
parser.add_argument('--tag', type=str, default='run',
                    help='optional tag to identify the run')
parser.add_argument('--fusion', choices=['intermediate', 'late'],
                    default='intermediate', help='type of fusion')
parser.add_argument('--path', default=None,
                    help='path with the model to be evaluated')
args = parser.parse_args()
print(f"TAG -> {args.tag}")

config = get_default_configuration()
config["fusion"] = args.fusion

# Load dataset options
config.merge_from_file(f'configs/datasets/opv2v.yml')

# Load model options
config.merge_from_file(f'configs/models/pyramid.yml')

# Load experiment options
config.merge_from_file(f'configs/experiments/test.yml')



# Finalise config
config.freeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(config.model, config)
criterion = build_criterion(config.model, config)

# Path with the trained model
state_dict = torch.load(args.path, map_location=device)
model.load_state_dict(state_dict["model"])

# model.eval()

single_mask = torch.from_numpy(np.load("/home/up202108347/opv2v-mono-semantic-segmentation/single_mask.npy")).to(device)

#dataset = Opv2vDataset("/home/up202108347/labels/opv2v_cars/", split="test")

dataset = CollabIntermidiateFusionOpv2vDataset(labels_root="/home/up202108347/labels/opv2v_cars/", 
                                       config=config,
                                       split="test"
                                       )

# sampler = RandomSampler(dataset, True, config.epoch_size)
# loader = DataLoader(dataset, 12, sampler=sampler, num_workers=config.num_workers)
loader = DataLoader(dataset, 1, shuffle=False,
            num_workers=config.num_workers, 
            collate_fn=custom_collate)

confusion = BinaryConfusionMatrix(config.num_class)
for i, batch in enumerate(loader):
# for idx in range(len(dataset)):
    # pdb.set_trace()
    if torch.cuda.is_available():
       batch = [t.cuda() for t in batch]

    # Predict class occupancy scores and compute loss
    image, calib, cam_positions, ego_position, labels, mask = batch

    with torch.no_grad():
        logits = model(image, calib, cam_positions, ego_position, single_mask, config.map_extents, config.map_resolution)
        loss = criterion(logits, labels, mask)

    # labels, logits, mask = dataset[idx]
    # labels = labels.unsqueeze(0).to(torch.int)
    # logits = logits.unsqueeze(0).to(torch.int)
    # mask = mask.unsqueeze(0).to(torch.int)
    # breakpoint()

    # Update confusion matrix
    scores = logits.cpu().sigmoid()  
    confusion.update(scores > config.score_thresh, labels, mask)

    if i % 100 == 0:
        print(f"idx: {i}, mean iou: {confusion.mean_iou*100} % ")

print("Final ---- ")
print(confusion.mean_iou)
print(confusion.mean_iou * 100)


