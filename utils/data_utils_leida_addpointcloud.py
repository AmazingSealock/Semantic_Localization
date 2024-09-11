

from torch.utils.data import Dataset
from typing import Any
import glob 
import os 
import torch 
from torchvision import transforms
from utils.joint_transforms import Compose, RandomHorizontallyFlip, Resize
from PIL import Image
import numpy as np 
import random 
from .mask_extract import extract_masks_and_classes
# from .coco_utils import get_all_images, get_mask_from_image


data_dir = "/home/xingzhaohu/point_cloud_project/data/"
image_root = os.path.join(data_dir, "img_5k")
mask_root = "/home/xingzhaohu/point_cloud_project/data/mask_3.8k_png"
bin_root = os.path.join(data_dir, "bin")

all_mask_paths = glob.glob(f"{mask_root}/*.png")
train_mask_paths = all_mask_paths[:3000]
test_mask_paths = all_mask_paths[3000:]

# data_dir = "/home/xingzhaohu/point_cloud_project/data/2024_0405_data"
# image_root = os.path.join(data_dir, "frames")
# mask_root = "/home/xingzhaohu/point_cloud_project/data/2024_0405_data/labels"

# all_mask_paths = glob.glob(f"{mask_root}/*.png")
# split_pos = int(len(all_mask_paths) * 0.8)

# train_mask_paths = all_mask_paths[:split_pos]
# test_mask_paths = all_mask_paths[split_pos:]

class MirrorDatasetVideo(Dataset):
    def __init__(self, is_train=True, image_size=384) -> None:
        super().__init__()
        if is_train:
            
            self.all_masks = train_mask_paths 
        else :
            self.all_masks = test_mask_paths

        self.image_size = image_size
        self.joint_transform = Compose([
            RandomHorizontallyFlip(),
            Resize((image_size, image_size))
        ])
        self.val_joint_transform = Compose([
            Resize((image_size, image_size))
        ])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.is_train = is_train

    def process_image_mask(self, image, mask, manual_random=None):
       
        if self.is_train:
            image, mask = self.joint_transform(image, mask, manual_random)
        else :
            image, mask = self.val_joint_transform(image, mask)

        image_raw = image
        image_raw = transforms.ToTensor()(image_raw)
        image = self.img_transform(image)
        mask = torch.from_numpy(np.array(mask))
        # mask = self.target_transform(mask)

        return image, mask, image_raw

    def __getitem__(self, index) -> Any:

        mask = self.all_masks[index]

        image_file = mask.split("/")[-1].split(".")[0] + ".png"
        bin_file = mask.split("/")[-1].split(".")[0] + ".bin"

        image_file = os.path.join(image_root, image_file)
        bin_file = os.path.join(bin_root, bin_file)

        # mask, classes = extract_masks_and_classes(mask)
        # print(np.unique(mask))
        image = Image.open(image_file).convert('RGB')
        mask = Image.open(mask).convert("L")

        w, h = image.size 

        image, mask, image_raw = self.process_image_mask(image, mask)

        return {
            "image": image,
            "image_raw": image_raw,
            "mask": mask,
            "size": (h, w),
            "bin_file_path": bin_file,
        } 
        
    def __len__(self):
        return len(self.all_masks)


def get_train_val_dataset(image_size=384): 
    train_dataset = MirrorDatasetVideo(is_train=True, image_size=image_size)
    val_dataset = MirrorDatasetVideo(is_train=False, image_size=image_size)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    dataset = MirrorDatasetVideo(data_dir="/home/zhaohu/mirror_detection/VMD/train")

    import matplotlib.pyplot as plt 
    for d in dataset:
        image = d["image"]
        mask = d["mask"]

        
