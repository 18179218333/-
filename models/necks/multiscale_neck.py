"""R2/R4/R5: fixed targets; optional trainable input-only feature pyramid."""
import torch
from torch import nn
from torch.nn import functional as F


class SimAM(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        delta = (x - x.mean((2, 3), keepdim=True)).square()
        variance = delta.sum((2, 3), keepdim=True) / max(x.shape[2] * x.shape[3] - 1, 1)
        inverse_energy = delta / (4 * (variance + self.eps)) + 0.5
        return x * inverse_energy.sigmoid()


class MultiScaleNeck(nn.Module):
    def __init__(self, inplanes, instrides, hidden_dim=128,
                 use_fpn=False, use_simam=False, jitter_scale=20.0):
        super().__init__()
        if len(inplanes) != 3 or list(instrides) != [4, 8, 16]:
            raise ValueError('Select EfficientNet-B4 outlayers=[2,3,4], strides=[4,8,16]')
        self.inplanes, self.instrides = list(inplanes), list(instrides)
        self.jitter_scale = float(jitter_scale)
        self.projections = nn.ModuleList([nn.Conv2d(c, hidden_dim, 1) for c in inplanes])
        self.enhance = SimAM() if use_simam else nn.Identity()
        self.use_fpn = use_fpn
        if use_fpn:
            self.refine = nn.ModuleList([
                nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1,
                                        groups=hidden_dim),
                              nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 1))
                for _ in inplanes
            ])

    def get_outplanes(self):
        # These are TARGET channels, not the projected memory channels.
        return self.inplanes

    def get_outstrides(self):
        return self.instrides

    def forward(self, data):
        targets = [x.detach() for x in data['features']]
        memory = []
        for x, projection in zip(targets, self.projections):
            y = self.enhance(x)
            if self.training and self.jitter_scale > 0:
                scale = y.norm(dim=1, keepdim=True) / y.shape[1]
                y = y + torch.randn_like(y) * scale * self.jitter_scale
            memory.append(projection(y))
        if self.use_fpn:
            # Input-only top-down pyramid. Never replace the fixed targets.
            for s in (1, 0):
                memory[s] = memory[s] + F.interpolate(
                    memory[s + 1], size=memory[s].shape[-2:],
                    mode='bilinear', align_corners=False)
            memory = [layer(x) for layer, x in zip(self.refine, memory)]
        return {'target_features': targets, 'memory_features': memory}
