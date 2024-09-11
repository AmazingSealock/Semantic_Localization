import numpy as np
from utils.data_utils_leida_addpointcloud import get_train_val_dataset
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from einops import rearrange
import random 
import os 
from matplotlib.colors import ListedColormap
from utils.pointpainting import PointPainter
from utils.KittiCalibration import KittiCalibration
import matplotlib.pyplot as plt 
from PIL import Image
from torchvision import transforms
from utils.joint_transforms import Compose, RandomHorizontallyFlip, Resize

def calculate_oa(pred_labels, gt_labels):
    """
    Calculate the Overall Accuracy (OA) for point cloud segmentation.

    Parameters:
    pred_file (str): Path to the file containing predicted labels.
    gt_file (str): Path to the file containing ground truth labels.

    Returns:
    float: Overall Accuracy (OA).
    """
    # Load predicted and ground truth labels
    # pred_labels = np.load(pred_file)
    # gt_labels = np.load(gt_file)

    # Ensure the shapes of prediction and ground truth are consistent
    assert pred_labels.shape == gt_labels.shape, "Shape of prediction and ground truth must be the same."

    # Calculate the number of correctly predicted points
    correct_predictions = np.sum(pred_labels == gt_labels)

    # Calculate Overall Accuracy (OA)
    oa = correct_predictions / len(gt_labels)

    return oa

from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_f1_score(pred_labels, gt_labels):
    """
    Calculate the F1 Score for point cloud segmentation.

    Parameters:
    pred_file (str): Path to the file containing predicted labels.
    gt_file (str): Path to the file containing ground truth labels.

    Returns:
    float: F1 Score.
    """

    # Load predicted and ground truth labels
    # pred_labels = np.load(pred_file)
    # gt_labels = np.load(gt_file)

    # Ensure the shapes of prediction and ground truth are consistent
    assert pred_labels.shape == gt_labels.shape, "Shape of prediction and ground truth must be the same."

    # Flatten the arrays if they are not already flat
    pred_labels = pred_labels.flatten()
    gt_labels = gt_labels.flatten()

    # Calculate F1 Score
    f1 = f1_score(gt_labels, pred_labels, average='weighted')
    precision = precision_score(gt_labels, pred_labels, average="weighted")
    recall = recall_score(gt_labels, pred_labels, average="weighted", )

    prediction = pred_labels / 255.
    gt = gt_labels / 255.

    mae = np.mean(np.abs(prediction - gt))


    return f1, precision, recall, mae


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# set_determinism(123)
set_random_seed(42)
from utils.misc import cal_Jaccard, cal_precision_recall_mae, cal_precision, cal_fmeasure, fscore
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
data_dir = "./data/fullres/train"
fold = 0

logdir = f"./logs_gpu4/segformer-m3"
env = "pytorch"
model_save_path = os.path.join(logdir, "model")
max_epoch = 30
batch_size = 8
val_every = 1
num_gpus = 1
device = "cuda:0"
image_size = 512

class MirrorTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        # from networks.segformer_modify.a_segformer_adapter_extra_embedding_depth_temporal_transformer import SegFormerAdapterExtraDepthMono
        # from networks.segformer_modify.segformer_adapter_extra_embedding_depth_mononet_mae import SegFormerAdapterExtraDepthMono
        from networks.segformer import SegFormer
        # from networks.mask2former_modify_v2.TPDM import TPDM
        self.model = SegFormer()
        self.load_state_dict("/home/xingzhaohu/point_cloud_project/logs_new_labels_0528/segformer-b0-all_data-ep100-20240620/model/final_model_1.0126.pt")

        self.index = 0
        self.colors = np.random.rand(150, 3)  # 生成一个10x3的随机数组，每行对应一个RGB颜色

    def training_step(self, batch):
        import time 
        s = time.time()
        image, label = self.get_input(batch)

        pred = self.model(image)

        # print(np.unique(label.cpu().numpy()))
        loss_lovasz = self.loss_func(pred, label)

        self.log("loss_ce", loss_lovasz, step=self.global_step)

        return loss_lovasz 

    # for image, label in data_loader:
    def get_input(self, batch):
        image = batch["image"]
        label = batch["mask"]
        image_raw = batch["image_raw"]
        bin_file_path = batch["bin_file_path"]
        # image_depth = batch["image_depth"]

        # label = label[]

        # print(label.shape)

        # print(np.unique(label.cpu().numpy()))
        label = label.long()    

        return image, label, image_raw, bin_file_path[0]

    def cal_metric(self, pred, gt):
        if pred.sum() > 0 and gt.sum() > 0:
            d = cal_Jaccard(pred, gt)
            # pre = cal_precision(pred, gt)
            f1 = fscore(pred, gt)
            acc = (pred == gt).mean()

            return d, f1, acc
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0, 1.0, 1.0
        
        else:
            return 0.0, 0.0, 0.0
    
    def read_pcd(self, pcd_path):
        lines = []
        num_points = None

        with open(pcd_path, 'r') as f:
            for line in f:
                lines.append(line.strip())
                if line.startswith('POINTS'):
                    num_points = int(line.split()[-1])
        assert num_points is not None

        points = []
        for line in lines[-num_points:]:
            x, y, z = list(map(float, line.split()))
            points.append((np.array([x, y, z])))

        return np.array(points)

    def run_single_file(self, image_path, pc_path):
        
        image_name = image_path.split("/")[-1].split(".")[0]
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB').resize((image_size, image_size))
        image = self.img_transform(image)[None,]

        self.model.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)

        output = torch.nn.functional.interpolate(output, size=(512, 1024), mode="bilinear")
        output = output.argmax(dim=1).cpu().numpy()

        colors = np.concatenate([np.zeros((1, 3)), np.random.rand(8, 3)])   # 生成一个10x3的随机数组，每行对应一个RGB颜色
        cmap = ListedColormap(colors)
        plt.imshow(output[0], cmap=cmap)
        plt.savefig("./test_colored.png")
        plt.clf()

        painter = PointPainter()
        # 'door': [1, 0, 0],  # red
        # 'ceiling': [0, 1, 0],  # green
        # 'wall': [0, 0, 1],  # blue
        # 'staircase': [1, 1, 0],  # yellow
        # 'railing': [1, 0, 1],  # magenta
        # 'column': [0, 1, 1],  # cyan
        # 'floor': [0, 0, 0],  # black
        # 'window': [0.75, 0.25, 0.75],  # pink
        # 'nothing': [1, 1, 1],  # white

        # labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
        colors = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                   [0, 0, 0], [0.75, 0.25, 0.75], [0.5, 0.5, 0.5]]
        semantic = output[0]
        # pointcloud = np.fromfile(pc_path, dtype=np.float32)
        pointcloud = self.read_pcd(pc_path)

        print(f"pointcloud is {pointcloud.shape}", pointcloud)

        pointcloud = pointcloud.reshape((-1, 3))
        calib = KittiCalibration("/home/xingzhaohu/point_cloud_project/kinect/record/record/calib/calib_kitti.txt")

        painted_pointcloud = painter.paint(pointcloud, semantic, calib)
        print(f"painted_pointcloud shape is {painted_pointcloud.shape}")
        os.makedirs("save_painted_pc", exist_ok=True)
        # np.save(os.path.join(pointcloud_prediction_root, f"{self.index}.bin"), painted_pointcloud)
        with open(os.path.join("./save_painted_pc", f"{image_name}.txt"), mode="a+") as f:
            for row in painted_pointcloud:
                if row[0] == 0 and row[1] == 0 and row[2] == 0 or row[-1] == 255.0:
                    continue

                for item in row[:3]:
                    f.write(str(item / 1000.0) + " ")

                
                color = colors[int(row[-1])]
                for c in color:
                    f.write(str(c) + " ")
                
                f.write(str(int(row[-1])))
                # f.write(str(colors[int(row[-1])]))

                f.write("\n")


    def validation_step(self, batch):
        image, label, image_raw, bin_file_path = self.get_input(batch)

        self.index += 1
        output = self.model(image)
        output = torch.nn.functional.interpolate(output, size=(512, 1024), mode="bilinear")
        output = output.argmax(dim=1).cpu().numpy()
        
        save_image_root = "./visualization_0620/images"
        os.makedirs(save_image_root, exist_ok=True)
        save_predictions_root = "./visualization_0620/predictions"
        os.makedirs(save_predictions_root,  exist_ok=True)
        save_masks_root = "./visualization_0620/masks"
        os.makedirs(save_masks_root,  exist_ok=True)

        pointcloud_prediction_root = "./visualization_0620/pointcloud_prediction"
        os.makedirs(pointcloud_prediction_root,  exist_ok=True)
        pointcloud_mask_root = "./visualization_0620/pointcloud_mask"
        os.makedirs(pointcloud_mask_root,  exist_ok=True)

        image_save = image_raw[0].permute(1, 2, 0).cpu().numpy()
        image_save = image_save * 255
        image_save = image_save.astype(np.uint8)
        # print(image_save.shape)
        image_save = Image.fromarray(image_save).convert("RGB")
        image_save.save(os.path.join(save_image_root, f"{self.index}.png"))
        
        painter = PointPainter()
        # 'door': [1, 0, 0],  # red
        # 'ceiling': [0, 1, 0],  # green
        # 'wall': [0, 0, 1],  # blue
        # 'staircase': [1, 1, 0],  # yellow
        # 'railing': [1, 0, 1],  # magenta
        # 'column': [0, 1, 1],  # cyan
        # 'floor': [0, 0, 0],  # black
        # 'window': [0.75, 0.25, 0.75],  # pink
        # 'nothing': [1, 1, 1],  # white

        # labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
        colors = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                   [0, 0, 0], [0.75, 0.25, 0.75], [0.5, 0.5, 0.5]]
        semantic = output[0]
        pointcloud = np.fromfile(bin_file_path, dtype=np.float32).reshape((-1, 4))
        calib = KittiCalibration("/home/xingzhaohu/point_cloud_project/kinect/record/record/calib/calib_kitti.txt")

        painted_pointcloud = painter.paint(pointcloud, semantic, calib)
        print(f"painted_pointcloud shape is {painted_pointcloud.shape}")
        # np.save(os.path.join(pointcloud_prediction_root, f"{self.index}.bin"), painted_pointcloud)
        with open(os.path.join(pointcloud_prediction_root, f"{self.index}.txt"), mode="a+") as f:
            for row in painted_pointcloud:
                if row[0] == 0 and row[1] == 0 and row[2] == 0 or row[-1] == 255.0:
                    continue

                for item in row[:3]:
                    f.write(str(item / 1000.0) + " ")

                
                color = colors[int(row[-1])]
                for c in color:
                    f.write(str(c) + " ")
                
                f.write(str(int(row[-1])))
                # f.write(str(colors[int(row[-1])]))

                f.write("\n")

        # (os.path.join(pointcloud_prediction_root, f"{self.index}.bin"), painted_pointcloud)
        painted_pointcloud_metric = painted_pointcloud[:, 3]

        mask = label[0].cpu().numpy()
        painted_pointcloud_mask = painter.paint(pointcloud, mask, calib)
        painted_pointcloud_mask_metric = painted_pointcloud_mask[:, 3]

        np.save(os.path.join(pointcloud_mask_root, f"{self.index}.bin"), painted_pointcloud_mask)

        oa = calculate_oa(painted_pointcloud_metric, painted_pointcloud_mask_metric)

        f1, precision, recall, mae = calculate_f1_score(painted_pointcloud_metric, painted_pointcloud_mask_metric)

        print(f" oa is {oa}, f1 is {f1}, precision is {precision}, recall is {recall}, mae is {mae}")

        return [oa, f1, precision, recall, mae]

        # exit(0)

        # prediction = output[0]
        # prediction_save = Image.fromarray(prediction.astype(np.uint8)).convert("L")
        # prediction_save.save(os.path.join(save_predictions_root, f"{self.index}.png"))

        # mask = label[0].cpu().numpy()
       
        # mask_save = Image.fromarray(mask.astype(np.uint8)).convert("L")
        # mask_save.save(os.path.join(save_masks_root, f"{self.index}.png"))


        # import matplotlib.pyplot as plt 
        # plt.subplot(1, 3, 1)
        # plt.imshow(image_raw[0].permute(1,2,0).cpu().numpy())
        # plt.subplot(1, 3, 2)
        # cmap = plt.get_cmap('tab10')
        # plt.imshow(output[0], cmap=cmap)
        # plt.subplot(1, 3, 3)
        # plt.imshow(label[0].cpu().numpy(), cmap=cmap)
        # plt.show()

        # if self.index == 10:
        #     self.break_validation = True
        # res = self.cal_metric(output == 1, label.cpu().numpy() == 1)
        # res1 = self.cal_metric(output == 2, label.cpu().numpy() == 2)
        # res2 = self.cal_metric(output == 3, label.cpu().numpy() == 3)
        # res3 = self.cal_metric(output == 4, label.cpu().numpy() == 4)
        # print(f"res1 is {res}")
        # print(f"res2 is {res1}")
        # print(f"res3 is {res2}")
        # print(f"res4 is {res3}")
        return 0
       

if __name__ == "__main__":
    trainer = MirrorTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17752,
                            training_script=__file__)
    
    trainer.run_single_file("/home/xingzhaohu/point_cloud_project/2024_0624_data_test/img/KIN_3828.png", "/home/xingzhaohu/point_cloud_project/2024_0624_data_test/pcd/KIN_3828.pcd")
    

    # trainer.run_single_file("/home/xingzhaohu/point_cloud_project/2024_0624_data_test/img/KIN_1283.png", "/home/xingzhaohu/point_cloud_project/data/bin/KIN_0.bin")
    # train_ds, val_ds = get_train_val_dataset(image_size=image_size)

    # import matplotlib.pyplot as plt 
    # for data in train_ds:
    #     image = data["image"]
    #     mask = data["mask"]
    #     print(image.shape, mask.shape)
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image.permute(1, 2, 0).numpy())
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(mask[0].numpy(), cmap="gray")
    #     plt.show()
    #     break

    # outputs = trainer.validation_single_gpu(val_ds)
    # print(outputs.shape)

    # print(outputs.mean(dim=0))
# 

    # trainer.train(train_dataset=train_ds, val_dataset=val_ds)

    # import matplotlib.pyplot as plt 
    # index = 0

    # def process(image, mask):
    #     image = image.permute(1, 2, 0)
    #     # mask = mask[0]
    #     return image, mask 
    
    # for d in train_ds:
    #     index += 1

    #     if index < 120:
    #         continue
    #     image = d["image"]
    #     mask = d["mask"]

    #     print(image.shape)
    #     print(mask.shape)
    #     image_1 = image[0]
    #     image_2 = image[1]
    #     image_3 = image[2]
    #     mask_1 = mask[0]
    #     mask_2 = mask[0]
    #     mask_3 = mask[0]

    #     # print()
    #     # print(f"mask shape is {mask.shape}")
    #     # mask = mask[0]
    #     # image = image.permute(1, 2, 0)
    #     image_1, mask_1 = process(image_1, mask_1)
    #     image_2, mask_2 = process(image_2, mask_2)
    #     image_3, mask_3 = process(image_3, mask_3)
    #     plt.subplot(3, 2, 1)
    #     plt.imshow(image_1)
    #     plt.subplot(3, 2, 2)
    #     plt.imshow(mask_1)
    #     plt.subplot(3, 2, 3)
    #     plt.imshow(image_2)
    #     plt.subplot(3, 2, 4)
    #     plt.imshow(mask_2)
    #     plt.subplot(3, 2, 5)
    #     plt.imshow(image_3)
    #     plt.subplot(3, 2, 6)
    #     plt.imshow(mask_3)
    #     plt.show()
    #     if index == 130:
    #         break