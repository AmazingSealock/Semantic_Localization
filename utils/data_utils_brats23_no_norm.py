

from torch.utils.data import Dataset
from typing import Any
import glob 
import os 
import numpy as np 
import torch
from timm.data import create_transform
from torchvision import transforms
from utils.joint_transforms import Compose, RandomHorizontallyFlip, Resize
from PIL import Image
import random 

img_paths = sorted(glob.glob("/home/xingzhaohu/data/Paired_600_CFPFFA/CFP/*/*.png"))

def get_loader_folder(train_rate=0.7, image_size=[768, 768]):
    pass
    dir_list = ["0", "1", "3", "4", "5"]
    all_paths_train = []
    all_paths_test = []
    dir_base = "/home/xingzhaohu/data/Paired_600_CFPFFA/CFP/"

    for each_dir in dir_list:
        ps = []
        for p in os.listdir(os.path.join(dir_base, each_dir)):
            # print(p)
            ps.append(p)

        ps = sorted(ps, key=lambda x: int(x.split('.')[0]))

        train_number = int(len(ps) * train_rate)

        for i in range(train_number):
            all_paths_train.append(os.path.join(dir_base, each_dir, ps[i]))
        
        for j in range(train_number, len(ps)):
            all_paths_test.append(os.path.join(dir_base, each_dir, ps[j]))

    train_ds = MyDataset(all_paths_train, is_train=True, image_size=image_size)
    test_ds = MyDataset(all_paths_test, is_train=False, image_size=image_size)

    loader = [train_ds, test_ds]

    return loader


def get_loader(train_rate=0.7, seed=42, image_size=[768, 768]):
    random.seed(seed)
    all_paths = img_paths

    train_number = int(len(all_paths) * train_rate)

    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    test_datalist = all_paths[train_number:] 

    print(f"training data is {len(train_datalist)}")
    print(f"test data is {len(test_datalist)}")

    train_ds = MyDatasetUWAFA(train_datalist, is_train=True, image_size=image_size)
    test_ds = MyDatasetUWAFA(test_datalist, is_train=False, image_size=image_size)

    loader = [train_ds, test_ds]

    return loader


class MyDataset(Dataset):
    def __init__(self, imgs, is_train=True, image_size=[768, 768]) -> None:
        super().__init__()

        self.imgs = imgs
        
        print(len(self.imgs))
        
        self.image_size = image_size

        noise_level=0
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size[0], image_size[1]), antialias=True),
            # transforms.RandomAffine(degrees=2*noise_level, translate=[0.04*noise_level, 0.04*noise_level], 
            #                         scale=[1-0.04*noise_level, 1+0.04*noise_level], fill=-1),
            # transforms.Normalize(mean=0.5, std=0.5)
            ])
        
        # self.joint_transform = Compose([
        #     RandomHorizontallyFlip(),
        #     Resize((image_size[0], image_size[1]))
        # ])
        # self.val_joint_transform = Compose([
        #     Resize((image_size[0], image_size[1]))
        # ])
        # self.img_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
        self.target_transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.is_train = is_train

        from utils.augmentation import get_training_transforms, get_validation_transforms
        self.train_trans = get_training_transforms(mirror_axes=[0,1])
        self.val_trans = get_validation_transforms()

    def __getitem__(self, index) -> Any:
        manual_random = random.random()  # random for transformation
        cur_img = self.imgs[index]

        filename = cur_img.split("/")[-1]
        dir_name = cur_img.split("/")[-2]

        class_index = int(dir_name)

        # dirs = os.path.dirname(cur_img)
        # identifier = dirs.split("/")[-1]

        # cur_index = cur_img.split("/")[-1].split("_")[0]

        cur_label = cur_img.replace("/CFP/", "/FFA/")
        
        cur_image = Image.open(cur_img).convert("RGB").resize(self.image_size)
        cur_label = Image.open(cur_label).convert("L").resize(self.image_size)
        
        size = cur_image.size

        cur_image = np.array(cur_image).astype(np.float32) / 255.0
        cur_label = np.array(cur_label).astype(np.float32) / 255.0

        # cur_image = cur_image.transpose(2, 0, 1)
        # cur_label = cur_label[None]
        

        cur_image = cur_image.transpose(2, 0, 1)[None]
        cur_label = cur_label[None, None]

        if self.is_train:
            # print("train*********")
            data = self.train_trans(**{
                "data": cur_image,
                "seg": cur_label,
            })
        else :
            # print("validation********")
            data = self.val_trans(**{
                "data": cur_image,
                "seg": cur_label,
            })

        
        cur_image = data["data"][0]
        cur_label = data["seg"][0]
        
        # cur_image = self.img_transform(cur_image)
        # cur_label = self.img_transform(cur_label)

        # print(cur_image.shape, cur_label.shape)

        # if self.is_train:
        #     cur_image, cur_label = self.joint_transform(cur_image, cur_label, manual_random=manual_random)
        # else :
        #     cur_image, cur_label = self.val_joint_transform(cur_image, cur_label, manual_random=manual_random)

        # cur_image = self.img_transform(cur_image)
        # cur_label = self.img_transform(cur_label)

        # cur_image = self.transformer(cur_image)
        # cur_label = self.transformer(cur_label)

        return {
            "class_index": class_index,
            "image": cur_image,
            "label": cur_label,
            "dir_name": dir_name, 
            "file_name": filename,
            "raw_size": size
        } 
        
    def __len__(self):

        return len(self.imgs)


class MyDatasetUWAFA(Dataset):
    def __init__(self, imgs, is_train=True, img_size=[768, 768]) -> None:
        super().__init__()

        self.imgs = imgs
        
        print(len(self.imgs))
        
        noise_level=0

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.RandomAffine(degrees=2*noise_level, translate=[0.04*noise_level, 0.04*noise_level], 
                                    scale=[1-0.04*noise_level, 1+0.04*noise_level], fill=-1),
            # transforms.Normalize(mean=0.5, std=0.5)
            ])
        
        self.transformer_mini = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0]//2, img_size[1]//2)),
            # transforms.Normalize(mean=0.5, std=0.5)
            ])    
        
        self.is_train = is_train
    
    def convert_to_resize(self, X):
        y1 = self.transformer(X)
        y2 = self.transformer_mini(X)
        return y1, y2
    
    def __getitem__(self, index) -> Any:
        manual_random = random.random()  # random for transformation
        cur_img = self.imgs[index]

        filename = cur_img.split("/")[-1]
        dir_name = cur_img.split("/")[-2]

        # dirs = os.path.dirname(cur_img)
        # identifier = dirs.split("/")[-1]

        # cur_index = cur_img.split("/")[-1].split("_")[0]

        cur_label = cur_img.replace("/CFP/", "/FFA/")
        
        cur_image = Image.open(cur_img).convert("RGB")
        cur_label = Image.open(cur_label).convert("L")
        
        XReal_A, XReal_A_half = self.convert_to_resize(cur_image)
        XReal_B, XReal_B_half = self.convert_to_resize(cur_label)

        size = cur_image.size
        # if self.is_train:
        #     cur_image, cur_label = self.joint_transform(cur_image, cur_label, manual_random=manual_random)
        # else :
        #     cur_image, cur_label = self.val_joint_transform(cur_image, cur_label, manual_random=manual_random)

        # cur_image = self.img_transform(cur_image)
        # cur_label = self.img_transform(cur_label)
        
        return {
            "X_realA": XReal_A,
            "X_realA_half": XReal_A_half,
            "X_realB": XReal_B,
            "X_realB_half": XReal_B_half,
            "dir_name": dir_name, 
            "file_name": filename,
            "raw_size": size
        } 
        
    def __len__(self):

        return len(self.imgs)