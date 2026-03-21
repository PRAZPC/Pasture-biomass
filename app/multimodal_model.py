from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

TARGET_COLUMNS = [
    "Dry_Total_g",
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
]


class BiomassMultimodalNet(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        unfreeze_layers: Sequence[str] = ("layer4",),
    ) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        for param in self.backbone.parameters():
            param.requires_grad = False

        for layer_name in unfreeze_layers:
            layer = getattr(self.backbone, layer_name, None)
            if layer is None:
                continue
            for param in layer.parameters():
                param.requires_grad = True

        self.image_head = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.context_head = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict(
            {
                target: nn.Linear(hidden_dim // 2, 1)
                for target in TARGET_COLUMNS
            }
        )

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        image_features = self.backbone(image)
        image_features = self.image_head(image_features)
        context_features = self.context_head(context)
        shared = self.shared(torch.cat([image_features, context_features], dim=1))
        return {
            target: head(shared).squeeze(1)
            for target, head in self.heads.items()
        }
