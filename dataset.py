import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


class Shape:
    """
    A Shape is a collection of Shape
    """

    def __init__(self, P1: np.ndarray, P2: np.ndarray):
        """
        (P1, P2) are the first and second point of each segment
        """
        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
    
    def distance(self, P: np.ndarray) -> np.ndarray:
        """
        distance between each point P and the closest segment of the shape
        """
        P = np.array(P)[:, None, :]
        P1, P2 = self.P1[None, ...], self.P2[None, ...]
        d1 = np.linalg.norm(P1 - P, axis=-1)
        d2 = np.linalg.norm(P2 - P, axis=-1)
        v = P - P1
        v12 = P2 - P1
        vl = np.sum(v * v12, axis=-1)/np.linalg.norm(v12, axis=-1)**2 * v12
        vt = v - vl
        d = np.linalg.norm(vt)
        return np.minimum(np.minimum(d1, d2), d).min(axis=1)


def generate_crosses(n: int) -> list[Shape]:
    """
    generate n crosses shapes
    """
    P1 = np.array([[(-1, -1), (-1, 1)]])
    P2 = np.array([[(1, 1), (1, -1)]])
    theta = np.random.uniform(0., 2*np.pi, (n, 1))
    rot = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], axis=-1).reshape(n, 2, 2)
    P1 = (rot @ P1 + 1) / 2
    P2 = (rot @ P2 + 1) / 2
    fact = np.random.uniform(0.2, 1., (n, 1, 1))
    offset = np.random.uniform(0., 1-fact, (n, 1, 2))
    P1 = fact*P1 + offset
    P2 = fact*P2 + offset
    return [Shape(p1, p2) for p1, p2 in zip(P1, P2)]

    

def generate_squares() -> list[Shape]:
    """
    generate n square shapes
    """
    P1 = np.array([[(-1, -1), (-1, 1)]])
    P2 = np.array([[(1, 1), (1, -1)]])
    theta = np.random.uniform(0., 2*np.pi, (n, 1))
    rot = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], axis=-1).reshape(n, 2, 2)
    P1 = (rot @ P1 + 1) / 2
    P2 = (rot @ P2 + 1) / 2
    fact = np.random.uniform(0.2, 1., (n, 1, 1))
    offset = np.random.uniform(0., 1-fact, (n, 1, 2))
    P1 = fact*P1 + offset
    P2 = fact*P2 + offset
    return [Shape(p1, p2) for p1, p2 in zip(P1, P2)]

def generate_shapes() -> list[tuple[pd.DataFrame, str]]:
    pass

if __name__ == "__main__":
    import IPython
    s = Shape([[0, 0]], [[2, 2]])
    d = s.distance([[0, 2]])
    IPython.embed()