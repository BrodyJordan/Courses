import os
import warnings
from collections import defaultdict

import hydra
from omegaconf import DictConfig
import torch
import torch.distributed as dist
from torch.optim import Adam, AdamW
from transformers import get_cosine_schedule_with_warmup

from helper import (
    adjust_learning_rate,
    evaluate,
    evaluate_and_update_min_val,
    prepare_dataloader,
    set_seed,
    train,
)
from model import create_model
from utils.utils import (
    freeze_params,
    get_nb_trainable_parameters,
    load_model_checkpoint,
    logging_wandb,
    setup_wandb_logging,
)

warnings.simplefilter("ignore")


def _infer_distributed_env():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        gpu_count = max(torch.cuda.device_count(), 1)
        local_rank = int(os.environ.get("SLURM_LOCALID", rank % gpu_count))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    return rank, local_rank, world_size


def _setup_distributed():
    rank, local_rank, world_size = _infer_distributed_env()
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs.")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

    return distributed, rank, local_rank, world_size


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    distributed = False
    rank = 0
    local_rank = 0
    world_size = 1
    wandb = None
    output_dir = None

    try:
        distributed, rank, local_rank, world_size = _setup_distributed()
        is_main_process = rank == 0

        if distributed:
            cfg.device = f"cuda:{local_rank}"
        elif str(cfg.device) == "cuda" and torch.cuda.is_available():
            cfg.device = "cuda:0"

        set_seed(cfg.training.seed + rank, cfg=cfg)

        if is_main_process:
            wandb, output_dir = setup_wandb_logging(cfg)

        dataloader_train, dataloader_val = prepare_dataloader(
            cfg,
            distributed=distributed,
            include_val=True,
            distributed_val=distributed,
        )

        model = create_model(cfg)
        use_amp_default = str(cfg.device).startswith("cuda")
        scaler = torch.cuda.amp.GradScaler(
            enabled=bool(getattr(cfg.training, "use_amp", use_amp_default))
        )

        if cfg.training.optimizer == "adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=cfg["training"]["lr"],
                weight_decay=cfg["training"]["weight_decay"],
            )
        elif cfg.training.optimizer == "adam":
            optimizer = Adam(
                model.parameters(),
                lr=cfg["training"]["lr"],
                weight_decay=cfg["training"]["weight_decay"],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")

        scheduler = None
        if cfg.training.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.training.warmup_steps * len(dataloader_train),
                num_training_steps=cfg.training.epochs * len(dataloader_train),
            )

        start_epoch = 0
        if cfg.load_model.model_path:
            start_epoch, model, optimizer, scheduler = load_model_checkpoint(
                cfg, model, optimizer, scheduler
            )

        if cfg.training.freeze_encoder:
            freeze_params(model)

        trainable_params, all_param = get_nb_trainable_parameters(model)
        if is_main_process:
            print(
                f"trainable params: {trainable_params:,d} || "
                f"all params: {all_param:,d} || "
                f"trainable%: {100 * trainable_params / all_param:.4f}"
            )
            if distributed:
                print(f"Distributed training enabled with world size={world_size}")

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        model_for_saving = model.module if hasattr(model, "module") else model

        min_val = defaultdict(lambda: 1e4)
        for epoch in range(start_epoch, cfg.training.epochs):
            if distributed and hasattr(dataloader_train.sampler, "set_epoch"):
                dataloader_train.sampler.set_epoch(epoch)
            if distributed and hasattr(dataloader_val.sampler, "set_epoch"):
                dataloader_val.sampler.set_epoch(epoch)

            if cfg.training.scheduler == "adjust_lr":
                adjust_learning_rate(optimizer, epoch, cfg)

            stats = {}
            stats = train(
                cfg,
                epoch,
                dataloader_train,
                model,
                optimizer,
                scheduler,
                stats,
                scaler=scaler,
                verbose=is_main_process,
            )

            stats = evaluate(
                "val",
                cfg,
                epoch,
                model,
                dataloader_val,
                stats,
                verbose=is_main_process,
            )

            if is_main_process:
                stats, min_val = evaluate_and_update_min_val(
                    cfg,
                    epoch,
                    model_for_saving,
                    stats,
                    min_val,
                    output_dir,
                    optimizer,
                    scheduler,
                )
                stats["lr"] = optimizer.param_groups[0]["lr"]
                logging_wandb(
                    cfg,
                    model_for_saving,
                    optimizer,
                    scheduler,
                    epoch,
                    stats,
                    output_dir,
                    wandb,
                )
                print("epoch ", epoch, " finished!")

            if distributed:
                dist.barrier()

        if is_main_process and cfg.wandb and wandb is not None:
            wandb.finish()
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
