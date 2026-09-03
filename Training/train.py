import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler as DS

from Dataset.dataset import PickPlaceDataset
from Model.policy import DiffusionPolicy

EMA_DECAY = 0.995

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")


def main():

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank, rank = 0, 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)

    dataset_path = os.path.join(os.path.dirname(SCRIPT_DIR), "Dataset", "pick_place_demos.hdf5")
    dataset = PickPlaceDataset(dataset_path, obs_horizon=2, action_horizon=8)
    sampler = DS(dataset, shuffle=True) if ddp else None
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=(device != "cpu"),
        persistent_workers=True,
    )

    model = DiffusionPolicy(action_dim=8, obs_horizon=2, action_horizon=8).to(device)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    if ddp:
        model = DDP(model, device_ids=[local_rank])
    core = model.module if ddp else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    num_epochs = 500

    for epoch in range(num_epochs):
        if ddp:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), core.parameters()):
                    ema_p.mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)

            epoch_loss += loss.item()

        if rank == 0:
            print(f"epoch {epoch}: loss={epoch_loss / len(loader):.5f}")

            if epoch % 50 == 0 or epoch == num_epochs - 1:
                torch.save({
                    "model": ema_model.state_dict(),
                    "action_min": dataset.action_min,
                    "action_max": dataset.action_max,
                }, os.path.join(CKPT_DIR, f"checkpoint_{epoch}.pt"))

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    
    main()