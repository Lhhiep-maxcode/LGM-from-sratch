# main.py

import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.model_config import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting (location: core\dataset.py)! (search keyword TODO)')

    def __init__(self, cfg: Options, training=True):
        
        self.cfg = cfg
        self.training = training

        # TODO: remove this barrier
        self._warn()

        # TODO: load the list of objects for training
        self.items = []
        with open('TODO: file containing the list', 'r') as f:
            for line in f.readlines():
                self.items.append(line.strip())

        # naive split
        if self.training:
            self.items = self.items
        else:
            # debug mode
            self.items = self.items[-self.opt.batch_size:]

        # default camera intrinsics
        self.tan_half_fovy = np.tan(np.deg2rad(self.cfg.fovy / 2))
        self.projection_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.projection_matrix[0, 0] = 1 / self.tan_half_fovy
        self.projection_matrix[1, 1] = 1 / self.tan_half_fovy
        self.projection_matrix[2, 2] = (self.cfg.zfar + self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[3, 2] = - (self.cfg.zfar * self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[2, 3] = 1

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        #  NEED TO PROCESSING DATA IN .OBJ FORMAT TO (IMAGE-CAMERA POSE) PAIRS
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
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        # Selects a set of camera viewpoints (frames) for this object
        if self.training:
            # input views are in (36, 72), other views are randomly selected
            vids = np.random.permutation(np.arange(36, 73))[:self.cfg.num_input_views].tolist() + np.random.permutation(100).tolist()
        else:
            # fixed views
            vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()

        for vid in vids:
        
            image_path = os.path.join(uid, 'rgb', f'{vid:03d}.png')     # (Vd: uid/rgb/070.png)
            camera_path = os.path.join(uid, 'pose', f'{vid:03d}.txt')   # (Vd: uid/pose/070.txt)

            try:
                # TODO: load data (modify self.client here)
                image = np.frombuffer(self.client.get(image_path), np.uint8)
                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                c2w = [float(t) for t in self.client.get(camera_path).decode().strip().split(' ')]
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
            except Exception as e:
                # print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # TODO: you may have a different camera system
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.cfg.cam_radius / 1.5 # 1.5 is the default scale
        
            # Background removing
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.cfg.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            # Padding to be enough views
            n = self.opt.num_views - vid_cnt
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
        images_input = F.interpolate(images[:self.cfg.num_input_views].clone(), size=(self.cfg.input_size, self.cfg.input_size), mode='bilinear', align_corners=False)   # [V, C, H, W]
        cam_poses_input = cam_poses[:self.cfg.num_input_views].clone()
        
        # data augmentation
        if self.training:
            if random.random() < self.cfg.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            if random.random() < self.cfg.prob_cam_jitter:
                cam_poses[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize ground-truth images, still in range [0, 1]
        results['images_output'] = F.interpolate(images, (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)
        results['masks_output'] = F.interpolate(masks.unsqueeze(0), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)

        # Plucker Embedding for input views
        rays_embeddings = []
        for i in range(self.cfg.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input, self.cfg.input_size, self.cfg.input_size, self.cfg.fovy, opengl=True)    # (h, w, 3)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)     # (h, w, 6)
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()      # (V=4, 6, h, w)
        final_input = torch.cat([images_input, rays_embeddings], dim=-1)        # (v=4, 9, h, w)
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2)     # World-to-camera matrix: [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix     # world-to-clip matrix: [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        # results = {
        #     'input': ...,
        #     'images_output': ...,
        #     'masks_output': ...,
        #     'cam_view': ...,
        #     'cam_view_proj': ...,
        #     'cam_pos': ...,
        # }
        return results

