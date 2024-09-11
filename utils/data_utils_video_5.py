

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
        self.t = 3

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

        image, mask = self.process_image_mask(image, mask, manual_random)

        images = []
        masks = []
        half = (self.t - 1) // 2
        for i in range(half):
            index_i = index - self.t + i 
            if index_i < 0:
                index_i = 0
            
        
            image_last = self.all_images[index_i]
            mask_last = self.all_masks[index_i]
            case_identifier_last = image_last.split("/")[-3]
            if case_identifier_last == case_identifier:
                image_last, mask_last = self.process_image_mask(image_last, mask_last, manual_random)
                images.append(image_last[None])
                masks.append(mask_last)
            else :
                image_last, mask_last = image, mask 
                images.append(image_last[None])
                masks.append(mask_last)
        
        images.append(image[None])
        masks.append(mask)

        for i in range(half):
            index_i = index + 1 + i 
            if index_i > len(self.all_images) - 1:
                index_i = -1
            
            image_last = self.all_images[index_i]
            mask_last = self.all_masks[index_i]
            case_identifier_last = image_last.split("/")[-3]
            if case_identifier_last == case_identifier:
                image_last, mask_last = self.process_image_mask(image_last, mask_last, manual_random)
                images.append(image_last[None])
                masks.append(mask_last)
            else :
                image_last, mask_last = image, mask 
                images.append(image_last[None])
                masks.append(mask_last)

        image = torch.cat(images, dim=0)

        return {
            "image": image,
            "mask": mask,
            "video_name": case_identifier,
            "case_name": case_name,
            "size": (h, w),
        } 
        
    def __len__(self):
        return len(self.all_images)


def get_train_val_dataset(image_size=384):
    pass 
    train_dataset = MirrorDatasetVideo("./VMD/train", is_train=True, image_size=image_size)
    val_dataset = MirrorDatasetVideo("./VMD/test", is_train=False, image_size=image_size)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    dataset = MirrorDatasetVideo(data_dir="./VMD/train")

    import matplotlib.pyplot as plt 
    for d in dataset:
        image = d["image"]
        mask = d["mask"]

        
