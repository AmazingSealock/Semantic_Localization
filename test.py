import numpy as np
from utils.data_utils_leida import get_train_val_dataset
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

import matplotlib.pyplot as plt 
from PIL import Image
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
        # self.load_state_dict("/home/xingzhaohu/point_cloud_project/logs_gpu4/segformer-b3-3.8k-ep100/model/final_model_0.9880.pt")
        # self.load_state_dict("/home/xingzhaohu/point_cloud_project/logs_gpu4/segformer-b0-5k-ep100/model/best_model_0.9620.pt")
        self.load_state_dict("/home/xingzhaohu/point_cloud_project/logs_gpu4/segformer-b0-all_data-ep100-20240418/model/final_model_0.9104.pt")
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
        # image_depth = batch["image_depth"]

        # label = label[]

        # print(label.shape)

        # print(np.unique(label.cpu().numpy()))
        label = label.long()    

        return image, label, image_raw

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

    @torch.no_grad
    def mirror_prediction(self, x):
        prediction = self.model(x)
        
        prediction += torch.flip(self.model(torch.flip(x, (2,)) ), (2,))
        prediction += torch.flip(self.model(torch.flip(x, (3,)) ), (3,))
        prediction += torch.flip(self.model(torch.flip(x, (2, 3)) ), (2, 3))
        
        return prediction
    
    def validation_step(self, batch):
        image, label, image_raw = self.get_input(batch)

        self.index += 1
        output = self.model(image)
        # output = self.mirror_prediction(image)

        output = output.argmax(dim=1).cpu().numpy()
        
        label = label.cpu().numpy()
        save_root = "visualization_0426_nomirror"
        save_image_root = f"./{save_root}/images"
        os.makedirs(save_image_root, exist_ok=True)
        save_predictions_root = f"./{save_root}/predictions_L"
        os.makedirs(save_predictions_root,  exist_ok=True)
        # save_masks_root = f"./{save_root}/masks_L"
        # os.makedirs(save_masks_root,  exist_ok=True)

        # image_save = image_raw[0].permute(1, 2, 0).cpu().numpy()
        # image_save = image_save * 255
        # image_save = image_save.astype(np.uint8)
        # # print(image_save.shape)
        # image_save = Image.fromarray(image_save).convert("RGB")
        # image_save.save(os.path.join(save_image_root, f"{self.index}.png"))

        prediction = output[0]
        cmap = ListedColormap(self.colors)

        # prediction_save = Image.fromarray(prediction.astype(np.uint8)).convert("L")
        # prediction_save.save(os.path.join(save_predictions_root, f"{self.index}.png"))

        # mask = label[0].cpu().numpy()

        plt.imshow(prediction, cmap=cmap)
        plt.savefig(os.path.join(save_predictions_root, f"{self.index}_colored.png"))

        # mask_save = Image.fromarray(mask.astype(np.uint8)).convert("L")
        # mask_save.save(os.path.join(save_masks_root, f"{self.index}.png"))

        number_labels = 150
        res = []
        dices = 0.0
        f1s = 0.0
        accs = 0.0

        for i in range(1, number_labels):
            output_i = output == i 
            target_i = label == i
            d, f1, acc = self.cal_metric(output_i, target_i)

            dices += d 
            f1s += f1 
            accs += acc 

            # res.extend([d, f1, acc])
        
        dices = dices / number_labels
        f1s = f1s / number_labels
        accs = accs / number_labels

        return [dices, f1s, accs]
    
    
    
    def validation_end(self, val_outputs):
        res = val_outputs
        res_mean = []
        for r in res:
            res_mean.append( r.mean())

        # # dices = dices.mean()
        # # f1 = f1.mean()
        # # acc = acc.mean()
        # # val_loss = val_loss.mean()
        # print(f"dices is {dices}, f1 is {f1}, acc is {acc}, val_loss is {val_loss}")
        print(f"res is {res_mean}")
        # self.log("dices", dices, step=self.epoch)
        # self.log("f1", f1, step=self.epoch)
        # self.log("acc", acc, step=self.epoch)
        # self.log("val_loss", val_loss, step=self.epoch)

        # if dices > self.best_mean_dice:
        #     self.best_mean_dice = dices
        #     save_new_model_and_delete_last(self.model, 
        #                                     os.path.join(model_save_path, 
        #                                     f"best_model_{dices:.4f}.pt"), 
        #                                     delete_symbol="best_model")

        # save_new_model_and_delete_last(self.model, 
        #                                 os.path.join(model_save_path, 
        #                                 f"final_model_{dices:.4f}.pt"), 
        #                                 delete_symbol="final_model")

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
    
    train_ds, val_ds = get_train_val_dataset(image_size=image_size)

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

    res = trainer.validation_single_gpu(val_ds)
    print(res)

    print(res.mean(dim=0))

    print(res.shape)
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