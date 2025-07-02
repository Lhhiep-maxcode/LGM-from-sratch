# core/model.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from core.model_config import Options

import kiui

class GaussianRenderer:
    def __init__(self, cfg: Options):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.bg = torch.tensor([1, 1, 1]).to(self.device)

        # projection matrix
        # self.tan_half_fovy = np.tan(np.deg2rad(cfg.fovy / 2))
        # self.projection_matrix = torch.zeros((4, 4)).to(self.device)
        # self.projection_matrix[0, 0] = 1 / self.tan_half_fovy
        # self.projection_matrix[1, 1] = 1 / self.tan_half_fovy
        # self.projection_matrix[2, 2] = - (cfg.zfar + cfg.znear) / (cfg.zfar - cfg.znear)
        # self.projection_matrix[2, 3] = - 2 * cfg.zfar * cfg.znear / (cfg.zfar - cfg.znear)
        # self.projection_matrix[3, 2] = -1

        self.tan_half_fovy = np.tan(np.deg2rad(cfg.fovy / 2))
        self.projection_matrix = torch.zeros(4, 4).to(self.device)
        self.projection_matrix[0, 0] = 1 / self.tan_half_fovy
        self.projection_matrix[1, 1] = 1 / self.tan_half_fovy
        self.projection_matrix[2, 2] = (cfg.zfar + cfg.znear) / (cfg.zfar - cfg.znear)
        self.projection_matrix[3, 2] = - (cfg.zfar * cfg.znear) / (cfg.zfar - cfg.znear)
        self.projection_matrix[2, 3] = 1

    def save_ply(self, gaussians: torch.Tensor, path, compatible=True):
        # Target Gaussians example:
        # ------------------------------
        # property float x
        # property float y
        # property float z
        # property float f_dc_0
        # property float f_dc_1
        # property float f_dc_2
        # property float opacity
        # property float scale_0
        # property float scale_1
        # property float scale_2
        # property float rot_0
        # property float rot_1
        # property float rot_2
        # property float rot_3
        # ------------------------------

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement

        # gaussians: [1, N, 14]
        means3D = gaussians[0, :, 0:3].contiguous().float()     # (N, 3)
        opacity = gaussians[0, :, 3:4].contiguous().float()     # (N, 1)
        scales = gaussians[0, :, 4:7].contiguous().float()      # (N, 3)
        rotations = gaussians[0, :, 7:11].contiguous().float()  # (N, 4)
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # filter out Gaussian with low opacity value
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    