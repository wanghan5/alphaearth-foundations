import argparse
from pathlib import Path
import shutil
import torch
import os
from typing import Optional, Tuple
from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.training import create_trainer
from alphaearth.data_olmoearth import create_olmoearth_dataloader


def _get_shm_total_gb() -> Optional[float]:
    try:
        usage = shutil.disk_usage("/dev/shm")
    except FileNotFoundError:
        return None
    return usage.total / (1024 ** 3)


def _create_dataloader(
    *,
    data_dir: str,
    csv_path: Optional[str],
    batch_size: int,
    num_workers: int,
    patch_size: int,
    normalize: bool,
    shuffle: bool,
    num_bands: int,
    cache_index: bool,
    cache_dir: Optional[str],
    pin_memory: bool,
    persistent_workers: Optional[bool],
    prefetch_factor: int,
):
    return create_olmoearth_dataloader(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        patch_size=patch_size,
        normalize=normalize,
        shuffle=shuffle,
        num_bands=num_bands,
        cache_index=cache_index,
        cache_dir=cache_dir,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def create_olmoearth_dataloader_with_autotune(
    *,
    data_dir: str,
    csv_path: Optional[str],
    batch_size: int,
    num_workers: int,
    patch_size: int,
    normalize: bool,
    shuffle: bool,
    num_bands: int,
    cache_index: bool,
    cache_dir: Optional[str],
    auto_tune: bool,
    shm_min_gb: float,
    pin_memory: bool,
    persistent_workers: Optional[bool],
    prefetch_factor: int,
) -> Tuple[torch.utils.data.DataLoader, int, int]:
    shm_total_gb = _get_shm_total_gb()
    if shm_total_gb is not None and shm_total_gb < shm_min_gb:
        print(
            f"/dev/shm total {shm_total_gb:.1f} GB < {shm_min_gb:.1f} GB: "
            "disabling pin_memory and reducing prefetch pressure."
        )
        pin_memory = False
        prefetch_factor = max(1, min(prefetch_factor, 2))

    if not auto_tune:
        return (
            _create_dataloader(
                data_dir=data_dir,
                csv_path=csv_path,
                batch_size=batch_size,
                num_workers=num_workers,
                patch_size=patch_size,
                normalize=normalize,
                shuffle=shuffle,
                num_bands=num_bands,
                cache_index=cache_index,
                cache_dir=cache_dir,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            ),
            batch_size,
            num_workers,
        )

    tuned_batch_size = batch_size
    tuned_num_workers = num_workers
    last_error: Optional[Exception] = None
    while tuned_batch_size >= 1:
        try:
            dataloader = _create_dataloader(
                data_dir=data_dir,
                csv_path=csv_path,
                batch_size=tuned_batch_size,
                num_workers=tuned_num_workers,
                patch_size=patch_size,
                normalize=normalize,
                shuffle=shuffle,
                num_bands=num_bands,
                cache_index=cache_index,
                cache_dir=cache_dir,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            _ = next(iter(dataloader))
            if tuned_batch_size != batch_size or tuned_num_workers != num_workers:
                print(
                    "Auto-tuned dataloader settings: "
                    f"batch_size={tuned_batch_size}, num_workers={tuned_num_workers}"
                )
            return dataloader, tuned_batch_size, tuned_num_workers
        except (RuntimeError, OSError) as exc:
            last_error = exc
            error_msg = str(exc).lower()
            print(f"Auto-tune attempt failed: {exc}")
            if tuned_num_workers > 0 and ("shm" in error_msg or "shared memory" in error_msg):
                tuned_num_workers = max(0, tuned_num_workers // 2)
                print(f"Reducing num_workers to {tuned_num_workers} due to shared memory pressure.")
                continue
            if tuned_num_workers > 0:
                tuned_num_workers = max(0, tuned_num_workers // 2)
                print(f"Reducing num_workers to {tuned_num_workers} after dataloader error.")
                continue
            if tuned_batch_size > 1:
                tuned_batch_size = max(1, tuned_batch_size // 2)
                print(f"Reducing batch_size to {tuned_batch_size} after dataloader error.")
                continue
            break

    raise RuntimeError(
        "Failed to create a working dataloader with auto-tuning."
    ) from last_error


def main():
    parser = argparse.ArgumentParser(description="Train AlphaEarth on OlmoEarth pretrain dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/olmoearth_pretrain_dataset/10_landsat_monthly",
        help="Directory containing tar files",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to CSV metadata file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Spatial patch size (H, W)",
    )
    parser.add_argument(
        "--auto_tune",
        action="store_true",
        help="Automatically tune batch size/num_workers to avoid OOM or /dev/shm exhaustion",
    )
    parser.add_argument(
        "--shm_min_gb",
        type=float,
        default=64.0,
        help="Minimum /dev/shm size to keep pinned memory and worker prefetching enabled",
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pinned memory for dataloader",
    )
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep workers persistent between epochs (default: True when num_workers > 0)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches prefetched by each worker",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides --epochs if set)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup steps for learning rate",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
        help="Log every N steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_olmoearth",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--landsat_bands",
        type=int,
        default=7,
        help="Number of Landsat bands to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not provided",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="alphaearth-foundations",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a checkpoint path",
    )
    parser.add_argument(
        "--cache_index",
        action="store_true",
        default=True,
        help="Cache the data index to avoid re-indexing tar files",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory to store cached data indices",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Loading OlmoEarth dataset from {data_dir}")
    print(f"Using {args.landsat_bands} Landsat bands")
    
    dataloader, tuned_batch_size, tuned_num_workers = create_olmoearth_dataloader_with_autotune(
        data_dir=str(data_dir),
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        normalize=True,
        shuffle=True,
        num_bands=args.landsat_bands,
        cache_index=args.cache_index,
        cache_dir=args.cache_dir,
        auto_tune=args.auto_tune,
        shm_min_gb=args.shm_min_gb,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    
    dataset_size = len(dataloader.dataset)
    steps_per_epoch = dataset_size // tuned_batch_size
    
    if args.max_steps is None:
        max_steps = args.epochs * steps_per_epoch
    else:
        max_steps = args.max_steps
    
    print(f"Dataset: {dataset_size} samples, {steps_per_epoch} steps/epoch")
    print(f"Training for {max_steps} steps ({max_steps / steps_per_epoch:.2f} epochs)")
    
    # 加载模型和可能的检查点
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        
        # 重建模型
        model = AlphaEarthFoundations(
            model_size="small",
            input_sources={"landsat": args.landsat_bands},
            decode_sources={"landsat": args.landsat_bands},
        )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 确定剩余训练步数
        start_step = checkpoint['step']
        remaining_steps = max_steps - start_step
        print(f"Resuming from step {start_step}, {remaining_steps} steps remaining")
    else:
        print("Starting training from scratch")
        model = AlphaEarthFoundations(
            model_size="small",
            input_sources={"landsat": args.landsat_bands},
            decode_sources={"landsat": args.landsat_bands},
        )
        start_step = 0
    
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Model size: small")
    print(f"Input sources: {model.input_sources}")
    print(f"Decode sources: {model.decode_sources}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    param_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Model size (float32): {param_size_mb:.2f} MB")
    
    print("\n" + "-"*80)
    print("MODEL ARCHITECTURE")
    print("-" * 80)
    print(model)
    print("="*80 + "\n")
    
    trainer = create_trainer(
        model=model,
        dataloader=dataloader,
        text_adapter=None,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    
    # 设置训练步数
    trainer.max_steps = max_steps
    trainer.warmup_steps = args.warmup_steps
    
    print(f"Starting training for {max_steps} steps...")
    print(f"Output directory: {args.output_dir}")
    if args.use_wandb:
        print(f"W&B logging enabled - Project: {args.wandb_project}, Run: {args.wandb_run_name or 'default'}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 如果是从检查点恢复，则调整训练步数
    if start_step > 0:
        trainer.train(max_steps=remaining_steps, log_every=args.log_every)
    else:
        trainer.train(max_steps=args.max_steps, log_every=args.log_every)
    
    print("Training run finished.")


if __name__ == "__main__":
    main()
