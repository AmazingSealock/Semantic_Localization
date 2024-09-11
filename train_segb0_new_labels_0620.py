import numpy as np
from utils.data_utils_leida_new_labels_0620 import get_train_val_dataset
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
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
fold = 0

# logdir = f"./logs_gpu4/segformer-b0-5k-ep100"
logdir = f"./logs_new_labels_0528/segformer-b0-all_data-ep100-20240620"
env = "DDP"
model_save_path = os.path.join(logdir, "model")
max_epoch = 100
batch_size = 8
val_every = 1
num_gpus = 4
device = "cuda:0"
image_size = 512

class MirrorTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        from networks.segformer import SegFormer
        self.model = SegFormer()
       
        # self.load_state_dict("/home/xingzhaohu/point_cloud_project/logs_gpu4/segformer-b0-5k-ep100/model/final_model_0.9620.pt")

        self.best_metric = 0.0

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=3e-5, eps=1e-8)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=3e-5, eps=1e-8)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                    # momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        # self.warmup = 0.1
        
        self.loss_func = nn.CrossEntropyLoss()
 
    def training_step(self, batch):
        import time 
        s = time.time()
        image, label = self.get_input(batch)

        
        pred = self.model(image)

        # print(pred.shape, label.shape)

        # print(np.unique(label.cpu().numpy()))
        loss_lovasz = self.loss_func(pred, label)

        self.log("loss_ce", loss_lovasz, step=self.global_step)

        return loss_lovasz 

    # for image, label in data_loader:
    def get_input(self, batch):
        image = batch["image"]
        label = batch["mask"]
    
        label = label.long()    

        return image, label

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
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output = self.model(image)

        output = output.argmax(dim=1).cpu().numpy()
        target = label.cpu().numpy()
        all_labels = np.unique(target)

        print(all_labels, "~~~", np.unique(output))
        dices = 0.0
        f1s = 0.0
        accs = 0.0

        for label in all_labels:
            if label == 0:
                continue
            
            output_i = output == label
            target_i = target == label
            d, f1, acc = self.cal_metric(output_i, target_i)
            dices += d 
            f1s += f1 
            accs += acc 
        
        
        dices = dices / (len(all_labels) - 1)
        f1s = f1s / (len(all_labels) - 1)
        accs = accs / (len(all_labels) - 1)
        
        return [dices, f1s, accs]
     
    
        # dice, f1, acc = self.cal_metric(output, target)
        
        # return dice, f1, acc, val_loss
    
    def validation_end(self, val_outputs):
        res = val_outputs
        res_mean = []
        for r in res:
            res_mean.append(r.mean())

        print(f"res is {res_mean}")
        self.log("dices", res_mean[0], step=self.epoch)
        self.log("f1", res_mean[1], step=self.epoch)
        self.log("acc", res_mean[2], step=self.epoch)

        f1 = res_mean[1]
        if f1 > self.best_metric:
            self.best_metric = f1
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{f1:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{f1:.4f}.pt"), 
                                        delete_symbol="final_model")

if __name__ == "__main__":

    trainer = MirrorTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17753,
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


    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

    


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