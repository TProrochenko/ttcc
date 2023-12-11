import torch

from args import TrainingArgs


def create_optimizer(params, args: TrainingArgs):
    decay_params = [p for n, p in params if (p.dim() >= 2) and p.requires_grad]
    no_decay_params = [p for n, p in params if (p.dim() < 2) and p.requires_grad]

    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups, lr=args.max_lr, betas=args.betas, fused=args.fused
    )

    return optimizer


def create_scheduler(optimizer, args: TrainingArgs):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        cycle_momentum=args.cycle_momentum,
        pct_start=args.warmup_ratio,
    )
    return scheduler
