"""Multi-scale reconstruction with fixed backbone targets and shared SMoE.

R1..R6 refer to the six findings in the accompanying review/README.
Requires the original repository's FastMoE installation; not executed here.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .moeblock import FMoETransformerMLP
from .moeblock.gates.base_gate import BaseGate
from .multiscale_attention import LocalCrossAttention, Neighborhood


class ProbabilityBalanceGate(BaseGate):
    """Top-k routing; probability importance CV^2, not CV^2 of signed logits."""
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        if not 1 <= top_k <= self.tot_expert:
            raise ValueError('Invalid top_k')
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k

    def forward(self, x, return_all_scores=False):
        logits = self.gate(x)
        importance = logits.float().softmax(-1).mean(0)
        self.set_loss(importance.var(unbiased=False) / (importance.mean().square() + 1e-8))
        selected, indices = logits.topk(self.top_k, dim=-1, sorted=False)
        weights = selected.float().softmax(-1).to(x.dtype)
        if return_all_scores:
            return indices, weights, logits
        return indices, weights


class ScaleScoreFusion(nn.Module):
    def __init__(self, weights):
        super().__init__()
        weights = torch.tensor(weights, dtype=torch.float32)
        if weights.ndim != 1 or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Invalid scale weights')
        self.register_buffer('weights', weights / weights.sum())
        self.register_buffer('normal_mean', torch.zeros(len(weights)))
        self.register_buffer('normal_std', torch.ones(len(weights)))
        self.register_buffer('calibrated', torch.tensor(False))

    @torch.no_grad()
    def set_statistics(self, mean, std):
        mean = torch.as_tensor(mean, device=self.normal_mean.device, dtype=torch.float32)
        std = torch.as_tensor(std, device=self.normal_std.device, dtype=torch.float32)
        if mean.shape != self.normal_mean.shape or std.shape != self.normal_std.shape:
            raise ValueError('Wrong number of calibration scales')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError('Invalid calibration statistics')
        self.normal_mean.copy_(mean)
        self.normal_std.copy_(std.clamp_min(1e-6))
        self.calibrated.fill_(True)

    def forward(self, errors, image_size):
        result = []
        for s, error in enumerate(errors):
            score = error
            if bool(self.calibrated.item()):
                score = ((score - self.normal_mean[s]) / self.normal_std[s]).clamp_min(0)
            result.append(self.weights[s] * F.interpolate(
                score, size=image_size, mode='bilinear', align_corners=False))
        return torch.stack(result).sum(0)


class MultiScaleMoEAD(nn.Module):
    expects_multiscale = True

    def __init__(self, inplanes, instrides, feature_sizes, hidden_dim=128,
                 nhead=4, depths=(2, 3, 4), moe_nume=4, moe_topk=2,
                 dim_feedforward=512, dropout=0.1, local_window=7,
                 local_exclusion_px=(4, 8, 16), cross_window=5,
                 cross_exclusion_px=0.0, use_cross=True,
                 score_weights=(1, 1, 1), query_chunk=128):
        super().__init__()
        if len(inplanes) != 3 or len(feature_sizes) != 3 or len(depths) != 3:
            raise ValueError('Exactly three scales required')
        if any(int(d) != d or d < 1 for d in depths):
            raise ValueError('All depths must be positive integers')
        if len(local_exclusion_px) != 3 or len(score_weights) != 3:
            raise ValueError('Three scale settings required')
        self.sizes = [tuple(hw) for hw in feature_sizes]
        self.channels = list(inplanes)
        self.depths = tuple(int(d) for d in depths)
        self.use_cross = bool(use_cross)
        self.cross_sources = (1, 2, 1)  # fine<-mid, mid<-coarse, coarse<-mid
        self.query = nn.ParameterList([
            nn.Parameter(torch.empty(1, h * w, hidden_dim)) for h, w in self.sizes])
        self.position = nn.ParameterList([
            nn.Parameter(torch.empty(1, h * w, hidden_dim)) for h, w in self.sizes])
        self.scale_embedding = nn.Parameter(torch.zeros(3, 1, 1, hidden_dim))
        for p in list(self.query) + list(self.position):
            nn.init.normal_(p, std=0.02)
        self.local = LocalCrossAttention(hidden_dim, nhead, dropout, query_chunk)
        self.local_geometry = nn.ModuleList([
            Neighborhood(hw, hw, stride, stride, local_window, radius)
            for hw, stride, radius in zip(self.sizes, instrides, local_exclusion_px)])
        if self.use_cross:
            self.cross = LocalCrossAttention(hidden_dim, nhead, dropout, query_chunk)
            self.cross_geometry = nn.ModuleList([
                Neighborhood(self.sizes[s], self.sizes[j], instrides[s], instrides[j],
                             cross_window, cross_exclusion_px)
                for s, j in enumerate(self.cross_sources)])
        self.moe = nn.ModuleList([
            FMoETransformerMLP(num_expert=moe_nume, d_model=hidden_dim,
                               d_hidden=dim_feedforward, top_k=moe_topk,
                               gate=ProbabilityBalanceGate,
                               activation=nn.Sequential(nn.ReLU(), nn.Dropout(dropout)))
            for _ in range(2)])
        # Only norms differ across scale/round. Attention/expert weights are shared.
        norm_count = 4 if self.use_cross else 3
        self.norms = nn.ModuleList([
            nn.ModuleList([nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(norm_count)])
                           for _ in range(depth)]) for depth in self.depths])
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, c) for c in inplanes])
        self.dropout = nn.Dropout(dropout)
        self.score_fusion = ScaleScoreFusion(score_weights)

    def forward(self, data):
        targets, memory = data['target_features'], data['memory_features']
        if len(targets) != 3 or len(memory) != 3:
            raise ValueError('Expected three target and memory maps')
        batch = targets[0].shape[0]
        tokens = []
        for s, (target, value, size, channels) in enumerate(zip(targets, memory, self.sizes, self.channels)):
            if target.shape != (batch, channels, *size) or value.shape[-2:] != size:
                raise ValueError('Feature shape mismatch at scale {}'.format(s))
            tokens.append(value.flatten(2).transpose(1, 2) + self.position[s] + self.scale_embedding[s])
        states = [q.expand(batch, -1, -1) + self.scale_embedding[s]
                  for s, q in enumerate(self.query)]
        aux = [[] for _ in range(3)]
        for iteration in range(max(self.depths)):
            for s in range(3):
                if iteration >= self.depths[s]:
                    continue
                norms = self.norms[s][iteration]
                # No direct target/raw-memory residual into the output state.
                state = states[s]
                state = norms[0](state + self.dropout(self.local(
                    state + self.position[s], tokens[s], self.local_geometry[s])))
                offset = 1
                if self.use_cross:
                    j = self.cross_sources[s]
                    state = norms[1](state + self.dropout(self.cross(
                        state + self.position[s], tokens[j], self.cross_geometry[s])))
                    offset = 2
                for stage, moe in enumerate(self.moe):
                    update = moe(state)
                    # Read NOW: the shared gate overwrites its loss on the next call.
                    gate_loss = moe.gate.get_loss(clear=True)
                    if self.training:
                        aux[s].append(gate_loss)
                    state = norms[offset + stage](state + self.dropout(update))
                states[s] = state
        reconstructed, errors = [], []
        for s, state in enumerate(states):
            h, w = self.sizes[s]
            rec = self.heads[s](state).transpose(1, 2).reshape(batch, self.channels[s], h, w)
            reconstructed.append(rec)
            errors.append((rec.float() - targets[s].float()).square().mean(1, keepdim=True).clamp_min(1e-12).sqrt())
        # Average over calls WITHIN scale first, then across scales.
        auxloss = torch.stack([torch.stack(losses).mean() for losses in aux]).mean() \
            if self.training else states[0].new_zeros(())
        pred = self.score_fusion(errors, data['image'].shape[-2:])
        return {'reconstructed_features': reconstructed, 'target_features': targets,
                'scale_errors': errors, 'pred': pred, 'auxloss': auxloss}
