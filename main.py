from core.model_config import AllConfigs, Options
from core.model import LGM
from accelerate import Accelerator
from safetensors.torch import load_file
from core.dataset import ObjaverseDataset as Dataset
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR


import torch
import tyro
import kiui
import wandb

def main():
    
    cfg = tyro.cli(AllConfigs)

    wandb.login(key=cfg.wandb_key)

    run = wandb.init(
        project=cfg.wandb_project_name,  # Specify your project
        name=cfg.wandb_experiment_name,
        id=cfg.wandb_experiment_id,
        resume=("must" if cfg.wandb_experiment_id else None),
        config={                        # Track hyperparameters and metadata
            "epochs": cfg.num_epochs, 
            "input_size": cfg.input_size,
            "splat_size": cfg.splat_size,
            "output_size": cfg.output_size,
            "num_views_used": cfg.num_views_used,
            "lambda_lpips_start": cfg.lambda_lpips_start, 
            "lambda_lpips_end": cfg.lambda_lpips_end,
            "lambda_mse_start": cfg.lambda_mse_start,
            "lambda_mse_end": cfg.lambda_mse_end,
            "lambda_alpha": cfg.lambda_alpha,          
        },
    )

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )

    model = LGM(cfg)

    # Load model checkpoint for FINE-TUNING
    if cfg.fine_tune and cfg.resume is not None:
        # (cfg.resume in file type)
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

    
    train_dataset = Dataset(data_path=cfg.data_path, cfg=cfg, type='train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_dataset = Dataset(data_path=cfg.data_path, cfg=cfg, type='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    # val_dataset = Dataset(data_path=cfg.data_path, cfg=cfg, type='val')
    # val_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=cfg.batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     drop_last=False,
    #     pin_memory=True
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # TODO: can consider to tuning the pct_start
    # scheduler (per-iteration)
    # consider to use ConsineAnnealingWarmRestart to escape local
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = cfg.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    # Warm up + CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    if not cfg.fine_tune and cfg.resume is not None:
        # NOTE: cfg.resume (dir type) must be saved by accelerator.save_state()
        # Continue training by loading all state of optimizer, model, scheduler
        accelerator.load_state(cfg.resume)

    best_psnr_eval = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        total_psnr = 0
        # Create tqdm only on main process
        if accelerator.is_main_process:
            print(f"----------Epoch {epoch + 1}----------")
            pbar = tqdm(total=len(train_dataloader), desc=f"[T] E{epoch+1}/{cfg.num_epochs}")
            

        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Accumulate to simulate large batch training
                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / cfg.num_epochs
                lambda_lpips = cfg.lambda_lpips_start * (cfg.lambda_lpips_end / cfg.lambda_lpips_start) ** step_ratio
                lambda_mse = cfg.lambda_mse_start * (cfg.lambda_mse_end / cfg.lambda_mse_start) ** step_ratio

                out = model(data, lambda_mse, lambda_lpips)
                loss = out['loss']
                psnr = out['psnr']

                accelerator.backward(loss)

                # synchronize to update model  
                if accelerator.sync_gradients:
                    # gradient clipping to avoid exploding gradients
                    accelerator.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

            if accelerator.is_main_process:
                pbar.update(1)
                mem_free, mem_total = torch.cuda.mem_get_info()
                pbar.set_postfix({
                    "ls": float(loss.detach()),
                    "psnr": float(psnr.detach()),
                    "vr": round((mem_total-mem_free)/1024**3),
                })

                if i % 100 == 0:
                    run.log({"Train loss (step)": loss.detach()})
                
                # save log images
                if i % 500 == 0:
                    run.log({
                        "Learning rate (step)": scheduler.get_last_lr()[0], 
                        "lambda MSE (step)": lambda_mse, 
                        "lambda LPIPS (step)": lambda_lpips,
                    })
                    with torch.no_grad():
                        gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)    # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_gt_images.jpg', gt_images)
                    
                        gt_mask = data['masks_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_mask = gt_mask.transpose(0, 3, 1, 4, 2).reshape(-1, gt_mask.shape[1] * gt_mask.shape[3], 1)    # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_gt_mask.jpg', gt_mask)

                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)  # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_pred_images.jpg', pred_images)

                        pred_mask = out['alphas_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_mask = pred_mask.transpose(0, 3, 1, 4, 2).reshape(-1, pred_mask.shape[1] * pred_mask.shape[3], 1)  # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_pred_mask.jpg', pred_mask)

        if accelerator.is_main_process:
            pbar.close()
        
        total_loss = accelerator.gather_for_metrics(total_loss).mean()  # calculate avg loss for 1 gpu: [loss_gpu1, loss_gpu2, ...] -> [loss_gpu_avg]
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()

        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[TRAIN INFO] Epoch: {epoch + 1} loss: {total_loss:.6f} psnr: {total_psnr:.4f}")
            run.log({"Train loss (Epoch)": total_loss, "Train psnr (Epoch)": total_psnr})

        # checkpoint
        if (epoch + 1) % 5 == 0 or epoch == cfg.num_epochs - 1:
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=f'{cfg.workspace}/lastest')

        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            if accelerator.is_main_process:
                pbar2 = tqdm(test_dataloader, desc=f"[E] E{epoch + 1}/{cfg.num_epochs}")

            for i, data in enumerate(test_dataloader):
                out = model(data)

                psnr = out['psnr']
                total_psnr += psnr.detach()

                if accelerator.is_main_process:
                    pbar2.update(1)
                    if i % 500 == 0:
                        gt_images = data['images_output'].detach().cpu().numpy()    # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                        kiui.utils.write_image(f'{cfg.workspace}/{epoch}_{i}_eval_gt_images.jpg', gt_images)

                        pred_images = out['images_pred'].detach().cpu().numpy()     # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.utils.write_image(f'{cfg.workspace}/{epoch}_{i}_eval_pred_images.jpg', pred_images)

            if accelerator.is_main_process:
                pbar2.close()
            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                run.log({"Test psnr (Epoch)": total_psnr})
                accelerator.print(f"[EVAL INFO] Epoch: {epoch + 1} psnr: {total_psnr:.4f}")

            if total_psnr > best_psnr_eval:
                best_psnr_eval = total_psnr
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.print("Best found => Saving model....")
                    accelerator.save_model(model, f'{cfg.workspace}/best')


if __name__ == "__main__":
    main()