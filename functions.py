from typing import Iterable
from copy import deepcopy
import torch


def train_loop(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_data: Iterable, val_data: Iterable, n_steps: int, patience: int):
    best_state = deepcopy(model.state_dict())
    best_step = 0
    best_loss = float("inf")
    try:
        for step in range(n_steps):
            optimizer.zero_grad()
            # training
            model.train()
            losses = []
            for batch in train_data:
                x, y = batch
                loss = model.loss(y, *x)
                loss.backward()
                losses.append(loss.item())
            train_loss = sum(losses) / max(1, len(losses))
            # validation
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_data:
                    x, y = batch
                    loss = model.loss(y, *x).item()
                    losses.append(loss)
            val_loss = sum(losses) / max(1, len(losses))
            # checkpointing
            if val_loss < best_loss:
                best_state = deepcopy(model.state_dict())
                best_step = step
                best_loss = val_loss
            elif step - best_step >= patience:
                print("early stopping")
                break
            # stepping
            optimizer.step()
            # displaying
            print(f"Step {step}: train loss {train_loss:.3g}, val loss = {val_loss:.3g}")
    except KeyboardInterrupt:
        print("interupted by user")
    model.load_state_dict(best_state)
