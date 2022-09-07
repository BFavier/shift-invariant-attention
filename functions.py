from typing import Iterable, Iterator
from copy import deepcopy
from typing import Optional
import numpy as np
import torch

from dataset import x_to_tensors, y_to_tensors


class Batchifyer:

    def __repr__(self) -> str:
        return "Batchifyer(x="+repr(self.x)+", y="+repr(self.y)+", batch_size={self.batch_size}, n_batches={self.n_batches})" 

    def __init__(self, x, y, n_batches: Optional[int] = None, batch_size: Optional[int] = None, shuffle: bool = True, device: torch.device = "cpu"):
        self.x = x_to_tensors(x, device)
        self.y = y_to_tensors(y, device)
        self.device = device
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator:
        if self.batch_size is None and self.n_batches is None:
            return ((x, y) for x, y in ((self.x, self.y),))
        else:
            if self.shuffle:
                indexes = torch.randperm(len(self.y), device=self.device)
            else:
                indexes = torch.arange(len(self.y), device=self.device)
            if self.batch_size is not None:
                n_batches = range(len(self.y) // self.batch_size)
                if self.n_batches is not None:
                    n_batches = min(self.n_batches, n_batches)
                batches = (indexes[i*self.batch_size:(i+1)*self.batch_size] for i in n_batches)
                return ((tuple(_x[batch] for _x in self.x), self.y[batch]) for batch in batches)
            elif self.n_batches is not None:
                batches = np.array_split(indexes, self.n_batches)
                return ((tuple(_x[batch] for _x in self.x), self.y[batch]) for batch in batches)


def train_loop(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_data: Batchifyer, val_data: Batchifyer, n_steps: int, patience: int):
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
