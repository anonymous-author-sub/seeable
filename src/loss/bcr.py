import random
import math

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def pedcc_generation(
    n: int, k: int = None, seed: Optional[int] = None
) -> np.ndarray:
    def pedcc_frame(n: int, k: int = None) -> np.ndarray:
        assert 0 < k <= n + 1
        zero = [0] * (n - k + 1)
        u0 = [-1][:0] + zero + [-1][0:]
        u1 = [1][:0] + zero + [1][0:]
        u = np.stack((u0, u1)).tolist()
        for i in range(k - 2):
            c = np.insert(u[len(u) - 1], 0, 0)
            for j in range(len(u)):
                p = np.append(u[j], 0).tolist()
                s = len(u) + 1
                u[j] = math.sqrt(s * (s - 2)) / (s - 1) * np.array(p) - 1 / (
                    s - 1
                ) * np.array(c)
            u.append(c)
        return np.array(u)

    U = pedcc_frame(n=n, k=k)
    r = np.random.RandomState(seed)
    while True:
        try:
            noise = r.rand(n, n) # [0, 1)   
            V, _ = np.linalg.qr(noise)
            break
        except np.linalg.LinAlgError:
            continue

    points = np.dot(U, V)
    return points

def generate(
    n: int,
    k: int,
    filename: Optional[str] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate k evenly distributed R^n points in a unit (n-1)-hypersphere 
    Args:
        n (int): dimension of the Euclidean space
        k (int): number of points to generate
        method (str): method to generate the points. Defaults to "simplex".
        filename (str, optional): filename to save the points. Defaults to None.
        seed (int, optional): seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: k evenly distributed points in a unit (n-1)-hypersphere

    >>> generate(2, 3, method="simplex")
    array([[ 0.        ,  0.        ],
           [ 0.70710678,  0.70710678],
    """
    if seed is None or not (isinstance(seed, int) and 0 <= seed < 2 ** 32):
        print("[pedcc.generate] seed must be an integer between 0 and 2**32-1")
        seed = random.randrange(2 ** 32)
        print("[pedcc.generate] seed set to", seed)

    points = pedcc_generation(n, k, seed=seed)

    if filename:
        path = f"{filename}_n{n}_k{k}_seed{seed}.npy"
        np.save(path, points)

    return points




class BCR(nn.Module):
    def __init__(
        self,
        num_times: int = 1,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = "all",
        # pedcc
        n: int = None,
        k: int = None,
        seed: Optional[int] = None,
    ):
        super(BCR, self).__init__()
        self.num_times = num_times
        # assert num_times == 2
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

        # generate pedcc points
        self.n = n
        self.k = k
        points = generate(
            n, k, filename="BCR", seed=seed
        )
        self.points = torch.from_numpy(points)
        print(self)

    def __str__(self):
        lines = []
        lines.append(f"pedcc({self.k}, {self.n})")
        lines.append(f"num_times: {self.num_times}")
        lines.append(f"temperature: {self.temperature}")
        lines.append(f"base_temperature: {self.base_temperature}")
        lines.append(f"contrast_mode: {self.contrast_mode}")
        return "[BCR]" + "\n" + "\n".join(["\t" + l for l in lines])

    def unstack(self, features, labels):
        """
        in:
            features torch.Size([bsz, d])
            labels torch.Size([bsz])

        out:
            features torch.Size([bsz//num_times, num_times, d])
            labels torch.Size([bsz//num_times])
        """
        batch_size = features.shape[0] // self.num_times
        features = torch.cat(
            [
                fi.unsqueeze(1)
                for fi in torch.split(features, [batch_size] * self.num_times, dim=0)
            ],
            dim=1,
        )
        if labels is not None:
            labels = labels[:batch_size]
        return features, labels

    def get_mask(self, labels, mask, batch_size: int):
        # labels|  mask | mask_output
        #   0   |   0   | torch.eye(batch_size)
        #   0   |   1   | mask
        #   1   |   0   | torch.eq(labels, labels.T)
        #   1   |   1   | "Cannot define both `labels` and `mask`"
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            return torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            return torch.eq(labels, labels.T).float()
        else:
            return mask.float()

    def forward(self, features, labels=None, mask=None, simclr=False):
        features, labels = self.unstack(features, labels)

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size, contrast_count = features.shape[:2]

        if not simclr:
            mask = self.get_mask(labels, mask, batch_size)
        else:
            mask = self.get_mask(None, None, batch_size)
        mask = mask.to(features)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_count = 1
            anchor_feature = features[:, 0]
        elif self.contrast_mode == "all":
            anchor_count = contrast_count
            anchor_feature = contrast_feature
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # ----------------------------------------------------------------------
        # replace self-contrast cases with pedcc sim
        # ----------------------------------------------------------------------

        self.points = self.points.to(features)
        centroids_feature = self.points[labels.repeat(anchor_count)]
        points_dot_contrast = torch.div(
            torch.mul(anchor_feature, centroids_feature).sum(1), self.temperature
        )
        I = torch.arange(centroids_feature.size(0))
        anchor_dot_contrast[I, I] = points_dot_contrast

        # ----------------------------------------------------------------------

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # fix : https://github.com/HobbitLong/SupContrast/pull/86
        pos_per_sample = mask.sum(1)  # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample  # mask.sum(1)

        # loss
        # the gradient scales inversely with choice of temperature τ;
        # therefore we rescale the loss by τ during training for stability
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

