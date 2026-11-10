"""R1/R3/R6: bounded sliding-neighborhood attention, no full L x L matrix.

This is deliberately NOT a cyclic shifted-window/Swin implementation.
Each query owns a local candidate neighborhood; adjacent neighborhoods overlap.
Masks are defined in input-image coordinates, including cross-scale cell overlap.
They reduce direct copying; CNN receptive-field/multi-hop leakage is not ruled out.
"""
import math
import torch
from torch import nn


def candidate_geometry(query_hw, source_hw, query_stride, source_stride,
                       window=7, exclusion_px=4.0):
    if window < 3 or window % 2 != 1:
        raise ValueError('window must be odd and >= 3')
    hq, wq = query_hw
    hs, ws = source_hw
    if (hq * query_stride, wq * query_stride) != (hs * source_stride, ws * source_stride):
        raise ValueError('Query and source grids must cover the same image')
    # Token centers, measured in input-image pixels.
    yy, xx = torch.meshgrid(torch.arange(hq), torch.arange(wq))
    qy = (yy.flatten().float() + .5) * query_stride
    qx = (xx.flatten().float() + .5) * query_stride
    anchor_y = torch.floor(qy / source_stride).long()
    anchor_x = torch.floor(qx / source_stride).long()
    offsets = torch.arange(-(window // 2), window // 2 + 1)
    oy, ox = torch.meshgrid(offsets, offsets)
    sy = anchor_y[:, None] + oy.flatten()[None]
    sx = anchor_x[:, None] + ox.flatten()[None]
    valid = (sy >= 0) & (sy < hs) & (sx >= 0) & (sx < ws)
    dy = (sy.float() + .5) * source_stride - qy[:, None]
    dx = (sx.float() + .5) * source_stride - qx[:, None]
    if query_stride == source_stride:
        # At stride 4 and exclusion_px=4 this excludes exactly a 3 x 3 square.
        excluded = (dy.abs() <= exclusion_px) & (dx.abs() <= exclusion_px)
    else:
        # Exclude overlapping token cells plus a configurable image-space margin.
        half_extent = (query_stride + source_stride) / 2 + exclusion_px
        excluded = (dy.abs() < half_extent) & (dx.abs() < half_extent)
    allowed = valid & ~excluded
    if not allowed.any(dim=1).all():
        raise ValueError('Some queries have no valid context: enlarge window or reduce exclusion')
    indices = sy.clamp(0, hs - 1) * ws + sx.clamp(0, ws - 1)
    relative_xy = torch.stack((dx, dy), dim=-1) / (window * source_stride)
    return indices, allowed, relative_xy


class Neighborhood(nn.Module):
    def __init__(self, query_hw, source_hw, query_stride, source_stride,
                 window, exclusion_px):
        super().__init__()
        idx, allowed, xy = candidate_geometry(query_hw, source_hw, query_stride,
                                               source_stride, window, exclusion_px)
        self.register_buffer('indices', idx, persistent=False)
        self.register_buffer('allowed', allowed, persistent=False)
        self.register_buffer('relative_xy', xy, persistent=False)


class LocalCrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1, query_chunk=128):
        super().__init__()
        if dim % heads or query_chunk < 1:
            raise ValueError('dim must be divisible by heads; query_chunk must be positive')
        self.heads, self.head_dim, self.chunk = heads, dim // heads, query_chunk
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.relative_bias = nn.Linear(2, heads, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, geometry):
        # query: [B,Lq,D], memory: [B,Ls,D]; output ALWAYS has Lq tokens.
        b, length, dim = query.shape
        q = self.q(query).reshape(b, length, self.heads, self.head_dim)
        k = self.k(memory).reshape(b, memory.shape[1], self.heads, self.head_dim)
        v = self.v(memory).reshape(b, memory.shape[1], self.heads, self.head_dim)
        outputs = []
        for start in range(0, length, self.chunk):
            end = min(start + self.chunk, length)
            idx = geometry.indices[start:end]
            keys, values = k[:, idx], v[:, idx]  # [B,chunk,K,H,d]
            scores = torch.einsum('bqhd,bqkhd->bqhk', q[:, start:end], keys)
            scores = scores / math.sqrt(self.head_dim)
            bias = self.relative_bias(geometry.relative_xy[start:end]).permute(0, 2, 1)
            scores = scores + bias.unsqueeze(0)
            scores = scores.masked_fill(~geometry.allowed[start:end][None, :, None, :],
                                        float('-inf'))
            weights = self.dropout(torch.softmax(scores.float(), dim=-1).to(values.dtype))
            out = torch.einsum('bqhk,bqkhd->bqhd', weights, values)
            outputs.append(out.reshape(b, end - start, dim))
        return self.out(torch.cat(outputs, dim=1))
