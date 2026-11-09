"""Funções de perda comparadas nos experimentos."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AngularMarginHead(nn.Module):
    """ArcFace: soma a margem `margin` ao ângulo entre o embedding e o centro da classe.

    Com margem zero, equivale à entropia cruzada sobre similaridade de cosseno
    (softmax normalizada), o que isola o efeito da margem na comparação.
    """

    def __init__(self, dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.centers = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.centers)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.normalize(embeddings, dim=1) @ F.normalize(self.centers, dim=1).T
        if self.margin > 0:
            theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
            target = F.one_hot(labels, cosine.shape[1]).bool()
            cosine = torch.where(target, torch.cos(theta + self.margin), cosine)
        return F.cross_entropy(self.scale * cosine, labels)


class BatchHardTriplet(nn.Module):
    """Tripleto: para cada âncora, usa o positivo mais distante e o negativo mais próximo."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embeddings, dim=1)
        distance = (2 - 2 * normalized @ normalized.T).clamp_min(1e-8).sqrt()
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        hardest_positive = (distance * (same & ~eye)).max(dim=1).values
        hardest_negative = distance.masked_fill(same, float("inf")).min(dim=1).values
        return F.relu(hardest_positive - hardest_negative + self.margin).mean()
