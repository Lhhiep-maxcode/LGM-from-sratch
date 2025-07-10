from core.model_config import AllConfigs, Options
from core.model import LGM
from accelerate import Accelerator
from safetensors.torch import load_file

import torch
import tyro

def main():
    cfg = tyro.cli(AllConfigs)
    
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )

    model = LGM(cfg)

    # Load model checkpoint
    if cfg.resume is not None:
        if cfg.resume.endswith('safetensors'):
            ckpt = load_file(cfg.resume, device='cpu')
        else:
            ckpt = torch.load(cfg.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')

    # data
    if cfg.data_mode == 's3':
        from core.dataset import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError
    
    train_dataset = Dataset(cfg, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_dataset = Dataset(cfg, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # TODO: can consider to tuning the pct_start
    # scheduler (per-iteration)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = cfg.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    

if __name__ == "__main__":
    main()