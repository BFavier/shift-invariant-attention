import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional
import torch


class Shape:
    """
    A Shape is a collection of Shape
    """

    def __init__(self, P1: np.ndarray, P2: np.ndarray):
        """
        (P1, P2) are the first and second point of each segment
        arrays of shape (N, 2)
        """
        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
    
    def distance(self, P: np.ndarray) -> np.ndarray:
        """
        distance between each point P and the closest segment of the shape

        Parameters
        ----------
        P : np.ndarray
            array of shape (N, 2)
        
        Returns
        -------
        np.ndarray :
            array of min distances of shape (N,)
        """
        P = np.array(P)[:, None, :]
        P1, P2 = self.P1[None, ...], self.P2[None, ...]
        d1 = np.linalg.norm(P1 - P, axis=-1)
        d2 = np.linalg.norm(P2 - P, axis=-1)
        v = P - P1
        v12 = P2 - P1
        scale = np.sum(v * v12, axis=-1)/np.linalg.norm(v12, axis=-1)**2
        Pd = np.clip(scale[..., None], 0, 1) * v12 + P1
        d = np.linalg.norm(P - Pd, axis=-1)
        return d.min(axis=1)

    def draw(self, ax: Optional[Axes] = None):
        if ax is None:
            ax = plt.gca()
        for p1, p2 in zip(self.P1, self.P2):
            ax.plot(*zip(p1, p2), color="k")
        ax.set_aspect("equal")


def _generate_crosses(n: int) -> list[Shape]:
    """
    generate n crosses shapes
    """
    P1 = np.array([(-1, 0), (0, 1)])[None, ...]
    P2 = np.array([(1, 0), (0, -1)])[None, ...]
    theta = np.random.uniform(0., 2*np.pi, n)
    rot = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], axis=-1).reshape(n, 1, 2, 2)
    P1 = (rot @ P1[..., None] + 1) / 2
    P2 = (rot @ P2[..., None] + 1) / 2
    P1, P2 = P1[..., 0], P2[..., 0]
    fact = np.random.uniform(0.2, 2., (n, 1, 1))
    offset = np.random.uniform(0., 2-fact, (n, 1, 2))
    P1 = fact*P1 + offset
    P2 = fact*P2 + offset
    return [Shape(p1, p2) for p1, p2 in zip(P1, P2)]

    

def _generate_squares(n: int) -> list[Shape]:
    """
    generate n square shapes
    """
    Ps = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    P1 = np.array(Ps)[None, ...]
    P2 = np.array(Ps[1:]+[Ps[0]])[None, ...]
    theta = np.random.uniform(0., 2*np.pi, n)
    rot = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], axis=-1).reshape(n, 1, 2, 2)
    P1 = (rot @ P1[..., None] + 1) / 2
    P2 = (rot @ P2[..., None] + 1) / 2
    P1, P2 = P1[..., 0], P2[..., 0]
    fact = np.random.uniform(0.2, 1., (n, 1, 1))
    offset = np.random.uniform(0., 1-fact, (n, 1, 2))
    P1 = fact*P1 + offset
    P2 = fact*P2 + offset
    return [Shape(p1, p2) for p1, p2 in zip(P1, P2)]


def _generate_cloud(shape: Shape) -> pd.DataFrame:
    """
    generate a cloud of points
    """
    P = np.random.uniform(0, 2, (np.random.randint(1200, 2000), 2))
    d = shape.distance(P)
    w = P[..., 0]
    t = P[..., 1]
    i = np.exp(-d*2)
    return pd.DataFrame.from_dict({"i": i, "w": w, "t": t})


def generate_dataset(n: int) -> tuple[list[pd.DataFrame], list[str]]:
    """
    generate a dataset of cloud observations
    """
    n_squares = int(round(0.8*n))
    n_crosses = n - n_squares
    x = [_generate_cloud(shape) for shapes in [_generate_squares(n_squares), _generate_crosses(n_crosses)] for shape in shapes]
    y = ["square"]*n_squares + ["cross"]*n_crosses
    indexes = np.random.permutation(len(x))
    return [x[i] for i in indexes], [y[i] for i in indexes]


def draw_dataset_sample(x: list[pd.DataFrame], y: list[str]):
    """
    draw a sample of the generated dataset
    """
    x, y = iter(x), iter(y)
    f, axes = plt.subplots(figsize=[4*3, 4*2], ncols=3, nrows=2)
    for row in axes:
        for ax in row:
            _x, _y = next(x), next(y)
            ax.scatter(_x["w"], _x["t"], c=_x["i"], marker=".", cmap="inferno")
            ax.set_title(_y)
            ax.axis("off")
    plt.show()


def x_to_tensors(x: list[pd.DataFrame], device: torch.device = "cpu") -> tuple[torch.Tensor]:
    """
    converts x into tensors (i, w, t, padding_mask)
    """
    lengths = [len(df) for df in x]
    L = max(len(df) for df in x)
    padding_mask = torch.tensor([[i < length for i in range(L)] for length in lengths], dtype=torch.bool, device=device)
    i = torch.tensor([df["i"].tolist() + [0]*(L-length) for df, length in zip(x, lengths)], dtype=torch.float32, device=device)
    w = torch.tensor([df["w"].tolist() + [0]*(L-length) for df, length in zip(x, lengths)], dtype=torch.float32, device=device)
    t = torch.tensor([df["t"].tolist() + [0]*(L-length) for df, length in zip(x, lengths)], dtype=torch.float32, device=device)
    return i, w, t, padding_mask


def y_to_tensors(y: list[str], classes: list[str], device: torch.device = "cpu") -> tuple[torch.Tensor]:
    """
    converts y to tensor of shape (N, n_classes) of 1 and 0
    """
    return torch.tensor([[index == target for index in classes] for target in y], dtype=torch.float32, device=device)


if __name__ == "__main__":
    import IPython
    shape = Shape([(0., 0.)], [(1., 1.)])
    d = shape.distance([(-1., -1.)])
    x, y = generate_dataset(20)
    draw_dataset_sample(x, y)
    IPython.embed()
