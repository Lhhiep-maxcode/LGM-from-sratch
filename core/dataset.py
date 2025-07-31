# main.py
# opengl/blender -> colmap style
# use opengl for Plucker Embedding
# OpenGL (x=Right, y=Up, z=Backward (camera looks along −Z))
# Colmap (x=Right, y=Down, z=Forward (camera looks along +Z))


import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from scipy.stats import gamma
from typing import Tuple, Literal, Dict, Optional



import kiui
from core.model_config import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ObjaverseDataset(Dataset):
    def __init__(self, data_path, cfg: Options, type: Literal['train', 'test', 'val']='train'):
        
        self.data_path = data_path
        self.cfg = cfg
        self.type = type if type in ['train', 'test', 'val'] else 'train'

        # TODO: load the list of objects for training
        self.items = [name for name in os.listdir(data_path)
                      if os.path.isdir(os.path.join(data_path, name))]

        # naive split
        if self.type == 'val':
            self.items = self.items[-self.cfg.val_size * len(self.items):]
        elif self.type == 'test':
            self.items = self.items[-(self.cfg.val_size + self.cfg.test_size) * len(self.items):-self.cfg.val_size * len(self.items)]
        else:
            self.items = self.items[:self.cfg.train_size * len(self.items)]

        # default camera intrinsics
        self.tan_half_fovy = np.tan(np.deg2rad(self.cfg.fovy / 2))
        self.projection_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.projection_matrix[0, 0] = 1 / self.tan_half_fovy
        self.projection_matrix[1, 1] = 1 / self.tan_half_fovy
        self.projection_matrix[2, 2] = (self.cfg.zfar + self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[3, 2] = - (self.cfg.zfar * self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[2, 3] = 1

        self.input_view_ids = [0, 2, 4, 6,         # L1
                               9, 11, 13, 15,      # L2
                                                   # L3
                               24,]                # L4
        
        self.test_view_ids = [i for i in range(cfg.num_views_total) if i not in self.input_view_ids]

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        #  NEED TO PROCESS DATA IN .OBJ FORMAT TO (IMAGE-CAMERA POSE) PAIRS
        # your_dataset/
            # ├── uid/
            # │   ├── rgb/
            # │   │   ├── 000.png
            # │   │   ├── 001.png
            # │   ├── pose/
            # │   │   ├── 000.txt
            # │   │   ├── 001.txt


        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        view_ids = np.array(self.input_view_ids) + np.random.permutation(self.test_view_ids)
        view_ids = view_ids[:self.cfg.num_views_used]

        for view_id in view_ids:
        
            # data path: /kaggle/input/objaverse-subset/archive_4
            image_path = os.path.join(self.data_path, uid, 'rgb', f'{view_id:03d}.png')
            camera_path = os.path.join(self.data_path, uid, 'pose', f'{view_id:03d}.txt') 

            try:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # shape: [512, 512, 4]
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image)  # shape: [H, W, C]
                
                with open(camera_path, 'r') as f:
                    lines = f.readlines()
                    
                    # OpenGL camera matrix: [4, 4]
                    c2w = torch.tensor([list(map(float, line.strip().split())) for line in lines]).reshape(4, 4)
            except Exception as e:
                # print(f'[WARN] dataset {uid} {vid}: {e}')
                continue

            # scale up radius to make model make scale predictions
            c2w[:3, 3] *= self.cfg.cam_radius / 1.5 # 1.5 is the default scale of the dataset
        
            # Background removing
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

        
        view_cnt = len(images)
        if view_cnt < self.cfg.num_views_used:
            print(f'[WARN] dataset {uid}: not enough valid views, only {view_cnt} views found!')
            # Padding to be enough views
            n = self.cfg.num_views_used - view_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n

        images = torch.stack(images, dim=0)     # [V, C, H, W]
        masks = torch.stack(masks, dim=0)       # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0)  # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.cfg.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        # resize input images
        images_input = F.interpolate(images[:len(self.input_view_ids)].clone(), size=(self.cfg.input_size, self.cfg.input_size), mode='bilinear', align_corners=False)   # [V, C, H, W]
        cam_poses_input = cam_poses[:len(self.input_view_ids)].clone()
        
        # data augmentation
        if self.type == 'train':
            # if random.random() < self.cfg.prob_grid_distortion:
            #     images_input[1:] = grid_distortion(images_input[1:])
            if random.random() < self.cfg.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # build rays for input views
        rays_embeddings = []
        for i in range(len(self.input_view_ids)):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.cfg.input_size, self.cfg.input_size, self.cfg.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V=9, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=9, 9, H, W]

        results['input'] = final_input
        results['cam_poses_input'] = cam_poses_input

        # resize ground-truth images, still in range [0, 1]
        results['images_output'] = F.interpolate(images, (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2)     # World-to-camera matrix: [V, 4, 4] (row-vector)
        cam_view_proj = cam_view @ self.projection_matrix     # world-to-clip matrix: [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        # results = {
        #     [C, H, W]
        #     'input': ...,             (processed input images 25x9x256x256)
        #     'cam_poses_input': ...,   
        #     'images_output': ...,     (25x3x512x512)
        #     'masks_output': ...,      (.......)
        #     'cam_view': ...,          (colmap coordinate)
        #     'cam_view_proj': ...,     (colmap coordinate)
        #     'cam_pos': ...,           (colmap coordinate)
        # }
        return results
