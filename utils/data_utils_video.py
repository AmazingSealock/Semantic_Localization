

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


class MirrorDatasetVideo(Dataset):
    def __init__(self, data_dir, is_train=True, image_size=384) -> None:
        super().__init__()
        self.all_images = sorted(glob.glob(f"{data_dir}/*/JPEGImages/*.jpg"))
        self.all_masks = sorted(glob.glob(f"{data_dir}/*/SegmentationClassPNG/*.png"))
        
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

    def process_image_mask(self, image, mask, manual_random):
        image = Image.open(image).convert('RGB')
        mask = Image.open(mask).convert('L')

        # transformation
        if self.is_train:
            image, mask = self.joint_transform(image, mask, manual_random)
        else :
            image, mask = self.val_joint_transform(image, mask)

        image = self.img_transform(image)
        mask = self.target_transform(mask)

        return image, mask 

    def __getitem__(self, index) -> Any:
        manual_random = random.random()  # random for transformation

        # return super().__getitem__(index)
        image = self.all_images[index]
        mask = self.all_masks[index]
        case_identifier = image.split("/")[-3]

        w, h = Image.open(image).size 

        # print(image_size)
        ## We should get the image's name to save the prediction using the same name.
        case_name = image.split("/")[-1].split(".")[0]

        # image depth shape is 1, 3, 384, 384, we only need to get the preprocess image in this section
        image_depth = preprocess(image, image_size=self.image_size)
        image, mask = self.process_image_mask(image, mask, manual_random)

        if index - 1 == -1:
            index = 1
        image_last = self.all_images[index-1]
        mask_last = self.all_masks[index-1]
        case_identifier_last = image_last.split("/")[-3]
        if case_identifier_last == case_identifier:
            image_last_depth = preprocess(image_last, image_size=self.image_size)

            image_last, mask_last = self.process_image_mask(image_last, mask_last, manual_random)

        else :
            image_last, mask_last = image, mask 
            image_last_depth = image_depth
        

        if (index + 1) == len(self.all_images):
            index = index - 1
        image_next = self.all_images[index+1]
        mask_next = self.all_masks[index+1]
        case_identifier_next = image_next.split("/")[-3]
        if case_identifier_next == case_identifier:
            image_next_depth = preprocess(image_next, image_size=self.image_size)
            image_next, mask_next = self.process_image_mask(image_next, mask_next, manual_random)

        else :
            image_next, mask_next = image, mask 
            image_next_depth = image_depth

        image = torch.cat([image_last[None], image[None], image_next[None]], dim=0)

        ## image depth B, T, 3, W, H
        image_depth = torch.cat([image_last_depth[None], image_depth[None], image_next_depth[None]], dim=0)

        # print(image_depth.shape)
        # print(image.shape)
        # print(mask.shape)

        return {
            "image": image,
            "mask": mask,
            "video_name": case_identifier,
            "case_name": case_name,
            "size": (h, w),
            "image_depth": image_depth
        } 
        
    def __len__(self):
        return len(self.all_images)


def get_train_val_dataset(image_size=384):
    pass 
    train_dataset = MirrorDatasetVideo("./VMD/train", is_train=True, image_size=image_size)
    val_dataset = MirrorDatasetVideo("./VMD/test", is_train=False, image_size=image_size)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    dataset = MirrorDatasetVideo(data_dir="/home/zhaohu/mirror_detection/VMD/train")

    import matplotlib.pyplot as plt 
    for d in dataset:
        image = d["image"]
        mask = d["mask"]

        
