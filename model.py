from typing import Optional, Callable
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataset import x_to_tensors


class ShiftInvariantAttention(torch.nn.Module):
    """
    Shift invariant attention perform an attention transformation that transforms
    a series of query, as a function of a series of keys and their relative positions with each query.
    """

    def __init__(self, projection_dim: int, n_heads: int, position_dim : int, activation: Callable = torch.relu):
        """
        Shift invariant attention with positional encoding

        Parameters
        ----------
        projection_dim : int
            dimension of projection for query/key vectors
        n_heads : int
            number of projection heads
        position_dim : int
            dimension of the position vector (1 for X, 2 for XY, 3 for XYZ ...)
        activation : Callable
            activation function
        """
        super().__init__()
        self.n_heads = n_heads
        self.projection_dim = projection_dim
        D = projection_dim * n_heads
        self.q_projection = torch.nn.Linear(D, n_heads * projection_dim)
        self.k_projection = torch.nn.Linear(D, n_heads * projection_dim)
        self.v_projection = torch.nn.Linear(D, n_heads * projection_dim)
        self.p_projection = torch.nn.Linear(position_dim, n_heads * projection_dim)
        self.expand = torch.nn.Linear(D, 2*D)
        self.activation = activation
        self.contract = torch.nn.Linear(2*D, D)
        self.batch_norm = torch.nn.BatchNorm1d(D)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, Pq: torch.Tensor, Pk: torch.Tensor,
                K_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        positional invariant attention of the form

        S_{ij} = sum_k Q_{ik} \times K_{jk} \times sin(a_k \times (Pq_{i} - Pk_{j}) + b_k)

        T_{id} = softmax(sum_j S_{ij}, dim=1) \times V_{jk}


        Because of the identity sin(a-b) = sin(a) \times cos(b) - cos(a) \times sin(b) we get:
        S_{ij} = sum_k [Q_{ik} \times sin(a_k \times Pq_{i} + b_k)] \times [K_{jk} \times cos(a_k \times Pk_{j})]
                       - [Q_{ik} \times sin(a_k \times Pq_{i} + b_k)] \times [K_{jk} \times cos(a_k \times Pk_{j})]

        Parameters
        ----------
        Q : torch.Tensor
            tensor of queries of shape (N, S, Lq, D)
        K : torch.Tensor
            tensor of keys of shape (N, S, Lk, D)
        Pq : torch.Tensor
            tensor of query positions of shape (N, S, Lq, P)
        Pk : torch.Tensor
            tensor of key position of shape (N, S, Lk, P)
        K_padding_mask : torch.tensor
            tensor of booleans of shape (N, S, Lk) marking keys that are padding and should be ignored

        Returns
        -------
        torch.Tensor :
            tensor T of transformed queries of shape (N, S, Lq, D)
        """
        N, S, Lq, D = Q.shape
        N, S, Lk, D = K.shape
        N, S, Lq, P = Pq.shape
        q = self.q_projection(Q).reshape(N, Q.shape[1], Lq, self.n_heads, self.projection_dim).permute(0, 1, 3, 2, 4)  # shape (N, 1|S, n_heads, Lq, projection_dim)
        k = self.k_projection(K).reshape(N, Q.shape[1], Lq, self.n_heads, self.projection_dim).permute(0, 1, 3, 2, 4)  # shape (N, 1|S, n_heads, Lk, projection_dim)
        v = self.v_projection(K).reshape(N, Q.shape[1], Lq, self.n_heads, self.projection_dim).permute(0, 1, 3, 2, 4)  # shape (N, 1|S, n_heads, Lk, projection_dim)
        pq = self.p_projection(Pq).reshape(N, S, Lq, self.n_heads, self.projection_dim).permute(0, 1, 3, 2, 4)  # shape (N, S, n_heads, Lk, projection_dim)
        pk = self.p_projection(Pk).reshape(N, S, Lk, self.n_heads, self.projection_dim).permute(0, 1, 3, 2, 4)  # shape (N, S, n_heads, Lk, projection_dim)
        b = self.p_projection.bias.reshape(1, 1, self.n_heads, 1, self.projection_dim)  # shape (1, 1, n_heads, 1, projection_dim)
        s = self.score_matrix(q, k, torch.sin(pq), torch.cos(pk - b)) - self.score_matrix(q, k, torch.cos(pq), torch.sin(pk - b))  # shape (N, S, n_heads, Lq, Lk)
        if K_padding_mask is not None:  # masking padding
            s = torch.masked_fill(s, K_padding_mask.reshape(N, 1, 1, 1, Lk), -float("inf"))
        s = torch.softmax(s, dim=-1)
        T = torch.matmul(s, v) # shape (N, S, n_heads, Lq, projection_dim)
        T = T.permute(0, 1, 3, 2, 4).reshape(N, S, Lq, D)  # shape (N, S, Lq, D)
        T = self.contract(self.activation(self.expand(T)))
        T = Q + self.batch_norm(T.reshape(-1, D)).reshape(N, S, Lq, D)
        return T

    def score_matrix(self, q: torch.Tensor, k: torch.Tensor, pq: torch.Tensor, pk: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        q : torch.Tensor
            tensor of shape (N, S, n_heads, Lq, projection_dim)
        k : torch.Tensor
            tensor of shape (N, S, n_heads, Lk, projection_dim)
        pq : torch.Tensor
            tensor of shape (N, S, n_heads, Lq, projection_dim)
        pk : torch.Tensor
            tensor of shape (N, S, n_heads, Lq, projection_dim)

        Returns
        -------
        torch.Tensor :
            tensor of shape (N, S, n_heads, Lq, Lk)
        """
        return torch.einsum("hijql, hijkl -> hijqk", q*pq, k*pk)



class ShiftInvariantTransformer(torch.nn.Module):
    """
    A shift invariant transformer with position invariance
    """

    def __init__(self, classes: list[str], n_stages: int, projection_dim: int,
                 n_heads: int, t_scaling_factors : list[float],
                 activation: Callable = torch.relu, low_memory: bool = True):
        """
        Parameters
        ----------
        classes : list of str
            name of the classes to predict
        n_stages : int
            number of attention stages
        projection_dim : int
            dimension of projection for query/key vectors
        n_heads : int
            number of projection heads
        t_scaling_factor : list of float
            the scaling factors to apply to t (for structural scale invariance)
        activation : Callable
            activation function
        low_memory : bool
            if True, use gradient checkpointing
        """
        super().__init__()
        self.projection_dim = projection_dim
        self.n_heads = n_heads
        D = projection_dim*n_heads
        self.classes = classes
        self.low_memory = low_memory
        self.t_scaling_factors = torch.tensor(t_scaling_factors, dtype=torch.float).reshape(1, -1, 1)  # shape (1, S, 1)
        self.expand = torch.nn.Linear(1, D)
        self.stages = torch.nn.ModuleList()
        for _ in range(n_stages):
            self.stages.append(ShiftInvariantAttention(projection_dim, n_heads, 2, activation))
        self.contract = torch.nn.Linear(D, len(classes))

    def forward(self, i: torch.Tensor, w: torch.Tensor, t: torch.Tensor,
                padding_mask : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        i : torch.Tensor
            tensor of signal intensity of shape (N, L)
        w : torch.Tensor
            tensor of molar weight of shape (N, L)
        t : torch.Tensor
            tensor of time of shape (N, L)
        padding_mask : torch.Tensor
            tensor of booleans of shape (N, L)

        Returns
        -------
        torch.Tensor
            tensor of class probabilities of shape (N, n_classes)
        """
        N, L = i.shape
        D = self.projection_dim * self.n_heads
        padding_mask = padding_mask.unsqueeze(1)
        X = self.expand(i.unsqueeze(-1)).unsqueeze(1)  # shape (N, 1, Lq, D)
        _, S, _ = self.t_scaling_factors.shape
        P = torch.stack([w.unsqueeze(1).expand(-1, S, -1), t.unsqueeze(1)*self.t_scaling_factors.to(t.device)], dim=-1)  # shape (N, S, Lq, 2)
        for stage in self.stages:
            if self.low_memory and self.training:
                X = checkpoint(stage, X, X, P, P, padding_mask)  # shape (N, S, Lq, D)
            else:
                X = stage(X, X, P, P, padding_mask)  # shape (N, S, Lq, D)
        # masking padding observations
        X = torch.masked_fill(X, padding_mask.unsqueeze(-1), -float("inf"))
        X = X.reshape(N, -1, D).max(dim=1).values
        return torch.sigmoid(self.contract(X))

    def loss(self, c: torch.Tensor, i: torch.Tensor, w: torch.Tensor, t: torch.Tensor, padding_mask : torch.tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        c : torch.tensor
            target classes as a tensor of booleans of shape (N, C)
        i : torch.Tensor
            signal intensity as a tensor of shape (N, L)
        w : torch.Tensor
            molar weights as a tensor of shape (N, L)
        t : torch.Tensor
            times as a tensor of shape (N, L)
        padding_mask : torch.Tensor
            tensor of booleans of shape (N, L)
        
        Returns
        -------
        torch.Tensor :
            scalar 
        """
        c = c.float()
        counts = c.sum(dim=0)  # shape (C,)
        weights = counts.sum() / (counts + 1.0E-3)  # shape (C,)
        return F.binary_cross_entropy(self(i, w, t, padding_mask), c, weight=weights.unsqueeze(0))
    
    def predict(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """
        predict the class of each of the given dataframes
        """
        self.eval()
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
        X = x_to_tensors(dfs, device=self.device)
        with torch.no_grad():
            Y = self(X)
        return pd.DataFrame(data=Y.cpu().numpy(), columns=self.classes)

    @property
    def device(self) -> torch.device:
        return self.contract.weight.device