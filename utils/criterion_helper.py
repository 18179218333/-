import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec, feature_align)


class ImageMSELoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["image"]
        image_rec = input["image_rec"]
        return self.criterion_mse(image, image_rec)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict

class MultiScaleFeatureMSELoss(nn.Module):
    """Mean within each scale, followed by normalized scale weighting."""
    def __init__(self, weight=1.0, scale_weights=(1, 1, 1)):
        super().__init__()
        weights = torch.tensor(scale_weights, dtype=torch.float32)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Invalid reconstruction scale weights")
        self.register_buffer("scale_weights", weights / weights.sum())
        self.weight = weight

    def forward(self, output):
        rec = output["reconstructed_features"]
        target = output["target_features"]
        if len(rec) != len(target) or len(rec) != len(self.scale_weights):
            raise ValueError("Mismatch between targets, reconstructions and weights")
        losses = torch.stack([F.mse_loss(x.float(), y.detach().float())
                              for x, y in zip(rec, target)])
        return (losses * self.scale_weights.to(losses)).sum()

