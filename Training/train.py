import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from Dataset.dataset import PickPlaceDataset
from Model.policy import DiffusionPolicy

EMA_DECAY = 0.995


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset", "pick_place_demos.hdf5")
    dataset = PickPlaceDataset(dataset_path, obs_horizon=2, action_horizon=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = DiffusionPolicy(action_dim=8, obs_horizon=2, action_horizon=8).to(device)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    num_epochs = 500

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)

            epoch_loss += loss.item()

        print(f"epoch {epoch}: loss={epoch_loss / len(loader):.5f}")

        if epoch % 50 == 0:
            torch.save({
                "model": ema_model.state_dict(),
                "action_min": dataset.action_min,
                "action_max": dataset.action_max,
            }, f"Training/checkpoints/checkpoint_{epoch}.pt")


if __name__ == "__main__":
    
    main()