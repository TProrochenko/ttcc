import torch
import wandb
from tqdm import tqdm

from args import ModelArgs, TrainingArgs, PreprocessingArgs
from dataloader import DataLoader
from model import Transformer
from utils import create_optimizer, create_scheduler


model_args = ModelArgs()
training_args = TrainingArgs()
preprocessing_args = PreprocessingArgs()

train_dl = DataLoader(training_args, dataset_dir="data/train")
val_dl = DataLoader(training_args, dataset_dir="data/val", shuffle=False, epochs=1)
training_args.steps = len(train_dl)


model = Transformer(model_args)
model.to(training_args.device)
model: Transformer = torch.compile(model)
model_args.total_model_params = sum(p.numel() for p in model.parameters())

optimizer = create_optimizer(model.named_parameters(), training_args)
scheduler = create_scheduler(optimizer, training_args)
autocast = torch.amp.autocast(device_type=training_args.device, dtype=torch.bfloat16)

config = {
    "model": model_args.__dict__,
    "training": training_args.__dict__,
    "preprocessing": preprocessing_args.__dict__,
}
wandb.init(training_args.wandb_project, config=config)

pbar = tqdm(train_dl)
for batch, stats in pbar:
    stats["lr"] = optimizer.param_groups[0]["lr"]
    if stats["step"] % training_args.eval_interval == 0:
        with autocast:
            model.eval()
            losses = []
            for val_batch, _ in val_dl:
                _ = model(*val_batch)
                losses.append(model.last_loss.item())
            model.train()
            stats["val_loss"] = sum(losses) / len(losses)
            tqdm.write(f", val_loss {stats['val_loss']:.4f}")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "config": config,
        }
        torch.save(checkpoint, f="models/ckpt.pt")
        wandb.log(stats, step=stats["step"])

    with autocast:
        _ = model(*batch)
        loss = model.last_loss

    pbar.set_description(f"lr {stats['lr']:.4f}, loss {loss.item():.4f}")
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.grad_clip)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
