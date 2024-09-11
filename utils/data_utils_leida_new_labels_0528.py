

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
image_root1 = os.path.join(data_dir, "img_5k")
mask_root1 = "/home/xingzhaohu/point_cloud_project/data/img_5k_new_labels"

all_mask_paths1 = glob.glob(f"{mask_root1}/*.png")
train_mask_paths = all_mask_paths1[:4000]
test_mask_paths = all_mask_paths1[4000:]

data_dir = "/home/xingzhaohu/point_cloud_project/data/2024_0405_data"
image_root2 = os.path.join(data_dir, "frames")
mask_root2 = "/home/xingzhaohu/point_cloud_project/data/2024_0405_data/segb5_labels"

all_mask_paths2 = glob.glob(f"{mask_root2}/*.png")
# split_pos = int(len(all_mask_paths) * 0.8)

train_mask_paths.extend(all_mask_paths2)
# test_mask_paths = all_mask_paths[split_pos:]

# Load model directly
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import numpy as np 

# processor = AutoImageProcessor.from_pretrained("/home/xingzhaohu/point_cloud_project/weight/segformer-b3-finetuned-ade-512-512")
# model = SegformerForSemanticSegmentation.from_pretrained("/home/xingzhaohu/point_cloud_project/weight/segformer-b3-finetuned-ade-512-512")
processor = AutoImageProcessor.from_pretrained("/home/xingzhaohu/point_cloud_project/weight/segformer-b0")
# model = SegformerForSemanticSegmentation.from_pretrained("/home/xingzhaohu/point_cloud_project/weight/segformer_b5")


# print(image.shape)

class MirrorDatasetVideo(Dataset):
    def __init__(self, is_train=True, image_size=384) -> None:
        super().__init__()
        if is_train:
            
            self.all_masks = train_mask_paths 
        else :
            self.all_masks = test_mask_paths

        print(f"data length is {len(self.all_masks)}")
        
        self.image_size = image_size
        self.joint_transform = Compose([
            # RandomHorizontallyFlip(),
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

        from .augmentation import get_training_transforms, get_validation_transforms
        self.train_trans = get_training_transforms(mirror_axes=[0,1])
        self.val_trans = get_validation_transforms()


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

        if "KIN_" in image_file:
            image_root = image_root1
        else :
            image_root = image_root2

        image_file = os.path.join(image_root, image_file)
        # mask, classes = extract_masks_and_classes(mask)
        # print(np.unique(mask))
        image = Image.open(image_file).convert('RGB')
        mask = Image.open(mask).convert("L")

        w, h = image.size 

        image, mask, image_raw = self.process_image_mask(image, mask)


        image = image[None].numpy()
        mask = mask[None, None,].numpy()

        # print(image.shape, mask.shape)
        if self.is_train:
            # print("train*********")
            data = self.train_trans(**{
                "data": image,
                "seg": mask,
            })
        else :
            # print("validation********")
            data = self.val_trans(**{
                "data": image,
                "seg": mask,
            })

        image = data["data"][0]
        mask = data["seg"][0, 0]

        return {
            "image": image,
            "image_raw": image_raw,
            "mask": mask,
            "size": (h, w),
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

        
