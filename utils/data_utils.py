

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
from .coco_utils import get_all_images, get_mask_from_image

all_images = get_all_images()
train_images = all_images
test_images = all_images[-10:]
image_root = "/home/xingzhaohu/jiuding_code/point_cloud_project/KINECT-11-13_Seg/train"
class MirrorDatasetVideo(Dataset):
    def __init__(self, is_train=True, image_size=384) -> None:
        super().__init__()
        if is_train:
            
            self.all_images = train_images 
        else :
            self.all_images = test_images

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
        # image = Image.open(image).convert('RGB')
        # mask = Image.open(mask).convert('L')

        # transformation
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

        image = self.all_images[index]
        image_file = image["file_name"]
        image_file = os.path.join(image_root, image_file)
        mask = get_mask_from_image(image)
        # print(np.unique(mask))
        image = Image.open(image_file).convert('RGB')
        mask = Image.fromarray(mask)

        w, h = image.size 

        image, mask, image_raw = self.process_image_mask(image, mask)

        return {
            "image": image,
            "image_raw": image_raw,
            "mask": mask,
            "size": (h, w),
        } 
        
    def __len__(self):
        return len(self.all_images)


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

        
