import torch
from torch.utils.data import Dataset

class AugmentedMapDataset(Dataset):

    def __init__(self, dataset, hflip=True, intermidiate_fusion="late"):
        self.dataset = dataset
        self.hflip = hflip
        self.intermidiate_fusion = intermidiate_fusion

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if self.intermidiate_fusion == "late":

            image, calib, labels, mask = self.dataset[index]

            # Apply data augmentation
            if self.hflip:
                image, labels, mask = random_hflip(image, labels, mask)

            return image, calib, labels, mask

        elif self.intermidiate_fusion == "intermidiate":
            cam_images, cam_calibs, cam_positions, ego_position, full_labels, full_mask = self.dataset[index]


            if self.hflip:
                cam_images, full_labels, full_mask = random_hflip(cam_images, full_labels, full_mask)

            return cam_images, cam_calibs, cam_positions, ego_position, full_labels, full_mask
            
def random_hlip_late(images, labels, mask):
    images = torch.flip(images, dims=[2, 3])
    labels = torch.flip(labels.int(), (-1,)).bool()
    mask = torch.flip(mask.int(), (-1,)).bool()
    return images, labels, mask
    
def random_hflip(image, labels, mask):
    image = torch.flip(image, (-1,))
    labels = torch.flip(labels.int(), (-1,)).bool()
    mask = torch.flip(mask.int(), (-1,)).bool()
    return image, labels, mask