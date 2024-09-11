from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

# from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
#     Convert3DTo2DTransform
import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import numpy as np 


def get_training_transforms(
            mirror_axes=[0,1],
    ) -> AbstractTransform:
        tr_transforms = []
        angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        # if do_dummy_2d_data_aug:
        #     ignore_axes = (0,)
        #     tr_transforms.append(Convert3DTo2DTransform())
        #     patch_size_spatial = patch_size[1:]
        # else:
        #     patch_size_spatial = patch_size
        #     ignore_axes = None
        
        tr_transforms.append(SpatialTransform(
                None, patch_center_dist_from_border=None,
                do_elastic_deform=True, alpha=(0, 100), sigma=(10, 13),
                do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
                p_rot_per_axis=1,  # todo experiment with this
                do_scale=True, scale=(0.7, 1.4),
                border_mode_data="constant", border_cval_data=0, order_data=3,
                border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
                random_crop=False,  # random cropping is part of our dataloaders
                p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
                independent_scale_for_each_axis=False  # todo experiment with this
            ))
        

        # if do_dummy_2d_data_aug:
        #     tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=None))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

def get_validation_transforms() -> AbstractTransform:
        val_transforms = []
        # val_transforms.append(RemoveLabelTransform(-1, 0))

        # val_transforms.append(RenameTransform('seg', 'target', True))

        val_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms