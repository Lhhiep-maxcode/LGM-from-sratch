import tyro
import torch

from core.model_config import AllConfigs, Options
from core.model import LGM
from safetensors.torch import load_file

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

cfg = tyro.cli(AllConfigs)

# model
model = LGM(cfg)

# resume pretrained checkpoint
if cfg.resume is not None:
    if cfg.resume.endswith('safetensors'):
        ckpt = load_file(cfg.resume, device='cpu')
    else:
        ckpt = torch.load(cfg.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')
