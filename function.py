import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


class Segments:
    """
    Segments is a collection of segments
    """

    def __init__(self, P1: np.ndarray, P2: np.ndarray):
        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
    
    def distance(self, P: np.ndarray) -> np.ndarray:
        """
        distance between each point point and each segment
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
        plt.plot(*zip([0, 0], v12[0, 0, :]), label="M1M2")
        plt.plot(*zip([0, 0], v[0, 0, :]), label="M1P")
        plt.plot(*zip([0, 0], vl[0, 0, :]), label="Vl")
        plt.plot(*zip([0, 0], vt[0, 0, :]), label="Vt")
        plt.legend()
        plt.gca().set_aspect("equal")
        plt.show()
        return np.minimum(np.minimum(d1, d2), d)


def generate()

def generate_shapes() -> pd.DataFrame:
    pass

if __name__ == "__main__":
    import IPython
    s = Segments([[0, 0]], [[2, 2]])
    d = s.distance([[0, 2]])
    IPython.embed()
