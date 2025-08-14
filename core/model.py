# main.py

import kiui.vis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.model_config import Options
from core.unet import UNet
from core.gs import GaussianRenderer
from kiui.lpips import LPIPS
from core.utils import get_rays
from diffusers import AutoencoderKL


class LGM(nn.Module):
    def __init__(self, cfg: Options):
        super().__init__()

        self.cfg = cfg

        # UNet
        self.unet = UNet(
            10, 14,                                                        
            down_channels=self.cfg.down_channels,
            down_attention=self.cfg.down_attention,
            mid_attention=self.cfg.mid_attention,
            up_channels=self.cfg.up_channels,
            up_attention=self.cfg.up_attention,
        )

        self.vae = AutoencoderKL.from_pretrained(                           
            cfg.vae_model, 
            subfolder="vae",
        ).to("cuda") 
        self.vae.requires_grad_(False)
        
        # last conv
        # self.conv = nn.Conv2d(14, 14, kernel_size=1)
        self.conv = nn.Conv2d(14, 4 * 14, kernel_size=1)                     

        # Gaussian Renderer
        self.gs = GaussianRenderer(cfg)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)     # Dense Gaussians
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.cfg.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k or 'vae' in k: 
                del state_dict[k]
        return state_dict
    
    def prepare_default_rays(self, device, elevation=0):
        # prepare Plucker embedding for 4 input images

        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.cfg.cam_radius),
            orbit_camera(elevation, 90, radius=self.cfg.cam_radius),
            orbit_camera(elevation, 180, radius=self.cfg.cam_radius),
            orbit_camera(elevation, 270, radius=self.cfg.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.cfg.input_size, self.cfg.input_size, self.cfg.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
    
    def forward_gaussians(self, images):
        # images: [B, 13, 9, H, W]
        # return: Gaussians: [B, num_gauss * 14]

        B, V, C, H, W = images.shape
        rgb_images = images[:, :, :3, :, :]   # [B, 13, 3, H, W]
        plucker = images[:, :, 3:, :, :]      # [B, 13, 6, H, W]

        rgb_images_reshaped = rgb_images.reshape(B*V, 3, H, W)

        
        with torch.no_grad():
            latents = self.vae.encode(rgb_images_reshaped).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # [B*V, 4, H/8, W/8], = [B*13, 4, 32, 32]
        _, latent_C, latent_H, latent_W = latents.shape

        # Downsample plucker embedding for same size with latent
        plucker_reshaped = plucker.reshape(B*V, 6, H, W)
        plucker_downsampled = F.interpolate(plucker_reshaped, size=(latent_H, latent_W), mode='bilinear', align_corners=False)

        unet_input = torch.cat([latents, plucker_downsampled], dim=1) # [B*13, 10, 32, 32]
        
        x = self.unet(unet_input)   # [B*13, 14*4, H, W]
        x = self.conv(x)        # [B*13, 14*4, H, W]

        _B_times_V, _C, _H, _W = x.shape 
        # x = x.reshape(B, V, 14, _H, _W)
        # x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)    # [B, 13, splat_size, splat_size, 14] --> [B, N, 14]
        k = _C // 14
        x = x.view(B, V, 14, k, _H, _W)
        x = x.permute(0, 1, 4, 5, 3, 2).contiguous() # [B, V, H, W, k, 14]
        x = x.view(B, -1, 14) # [B, V * H * W * k, 14]                                                # replace
        
        pos = self.pos_act(x[..., 0:3])     # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4]) # [B, N, 1]
        scale = self.scale_act(x[..., 4:7]) # [B, N, 3]
        rotation = self.rot_act(x[..., 7:11])   # [B, N, 3]
        rgbs = self.rgb_act(x[..., 11:])    # [B, N, 4]

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)    # [B, N, 14]
        return gaussians
    
    def forward(self, data):
        # data: output of the dataloader
        # data = {
        #     [C, H, W]
        #     'input': ...,             (processed input images 13x9x256x256)
        #     'cam_poses_input': ...,   
        #     'images_output': ...,     (17x3x512x512)
        #     'masks_output': ...,      (.......)
        #     'cam_view': ...,          (colmap coordinate)
        #     'cam_view_proj': ...,     (colmap coordinate)
        #     'cam_pos': ...,           (colmap coordinate)
        # }
        # ------------
        # return: results = {
        #     'gaussians': ...,
        #     'images_pred': ...,
        #     'alphas_pred': ...,
        #     'loss_mse': ...,
        #     'loss_lpips': ...,
        #     'loss': ...,
        #     'psnr': ...,
        # }

        results = {}
        loss = 0

        images = data['input']  # [B, 13, 9, H, W], input features (not necessarily orthogonal)

        # predicting 3DGS representation
        gaussians = self.forward_gaussians(images)  # [B, N, 14]

        results['gaussians'] = gaussians

        # always use white background
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        rendered_results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = rendered_results['image']  # [B, V, C, output_size, output_size]
        pred_alphas = rendered_results['alpha']  # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output']   # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output']     # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + (1 - gt_masks) * bg_color.view(1, 1, 3, 1, 1)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        results['loss_mse'] = loss_mse
        loss = loss + loss_mse

        if self.cfg.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # Rescale value from [0, 1] to [-1, -1] and resize to 256 to save memory cost
                F.interpolate(gt_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.cfg.lambda_lpips * loss_lpips

        results['loss'] = loss

        # metric
        with torch.no_grad():
            mse = torch.mean((pred_images.detach() - gt_images) ** 2)
            psnr = -10 * torch.log10(mse)
            results['psnr'] = psnr

        return results