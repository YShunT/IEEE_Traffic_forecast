"""DCRNN 型の符号化・復号モデル（道路グラフ上の拡散 GRU とパネル内の注意）と、そのグラフの行列。"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch import nn

BUCKETS = 10
BLOCKED = BUCKETS


def supports(n, index, edges, feeds, diffusion):
    up, down = np.zeros((n, n), np.float64), np.zeros((n, n), np.float64)
    for (a, b), steps in edges.items():
        i, j = index[a], index[b]
        up[j, i] = down[i, j] = 1.0 / steps
    for b, a in feeds.items():
        up[index[b], index[a]] = down[index[b], index[a]] = 1.0
    mats = []
    for m in (up, down):
        s = m.sum(1, keepdims=True)
        m = np.divide(m, s, out=np.zeros_like(m), where=s > 0)
        power = np.eye(n)
        for _ in range(diffusion):
            power = power @ m
            mats.append(power)
    return np.concatenate(mats).astype(np.float32)


def relation_buckets(n, index, edges, feeds):
    rows, cols = [index[a] for a, _ in edges], [index[b] for _, b in edges]
    d = shortest_path(csr_matrix((np.array(list(edges.values()), float), (rows, cols)), shape=(n, n)), directed=True, method="D")
    cut = lambda s: np.digitize(s, [1.5, 4.5, 10.5])
    out = np.full((n, n), 9, np.int64)
    down, up = np.isfinite(d), np.isfinite(d.T)
    use_down = down & (~up | (d <= d.T))
    out[use_down] = 1 + cut(d[use_down])
    use_up = up & ~use_down
    out[use_up] = 5 + cut(d.T[use_up])
    np.fill_diagonal(out, 0)
    return bypass_buckets(out, [(index[b], index[a]) for b, a in feeds.items()])


def graph_tensors(n, index, edges, feeds, diffusion, device="cuda"):
    return (
        torch.as_tensor(supports(n, index, edges, feeds, diffusion), device=device),
        torch.as_tensor(relation_buckets(n, index, edges, feeds), device=device),
    )


def bypass_buckets(out, feeds):
    out = out.copy()
    for b, a in feeds:
        out[b] = out[a]
        out[b, b] = 0
    for b, _ in feeds:
        out[:, b] = BLOCKED
        out[b, b] = 0
    return out


class DiffConv(nn.Module):
    def __init__(self, c_in, c_out, n_supports):
        super().__init__()
        self.k = n_supports
        self.lin = nn.Linear(c_in * (1 + n_supports), c_out)

    def forward(self, x, s):
        b, n, c = x.shape
        y = torch.matmul(s, x).view(b, self.k, n, c).permute(0, 2, 1, 3).reshape(b, n, self.k * c)
        return self.lin(torch.cat([x, y], -1))


class DCGRUCell(nn.Module):
    def __init__(self, c_in, hidden, n_supports):
        super().__init__()
        self.hidden = hidden
        self.gates = DiffConv(c_in + hidden, 2 * hidden, n_supports)
        self.cand = DiffConv(c_in + hidden, hidden, n_supports)

    def forward(self, x, h, s):
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], -1), s)).chunk(2, -1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1), s))
        return u * h + (1 - u) * c


class PanelAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.bias = nn.Parameter(torch.zeros(heads, BUCKETS))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, buckets):
        b, n, d = h.shape
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        score = q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5 + self.bias[:, buckets.clamp(max=BUCKETS - 1)]
        score = score.masked_fill(buckets == BLOCKED, float("-inf"))
        out = (score.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d)
        return self.norm(h + self.proj(out))


class DCRNN(nn.Module):
    def __init__(self, c_hist, c_ctx, c_dec, hidden, layers, diffusion, ctx_hidden, heads, attention, dropout, z0):
        super().__init__()
        k = 2 * diffusion
        self.z0, self.layers = z0, layers
        self.ctx = nn.Sequential(nn.Linear(c_ctx, ctx_hidden), nn.ReLU())
        self.enc = nn.ModuleList([DCGRUCell(c_hist + ctx_hidden if i == 0 else hidden, hidden, k) for i in range(layers)])
        self.dec = nn.ModuleList([DCGRUCell(c_dec if i == 0 else hidden, hidden, k) for i in range(layers)])
        self.attn = PanelAttention(hidden, heads) if attention else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def encode(self, hist, ctx, s, buckets):
        b, t, n, _ = hist.shape
        e = self.ctx(ctx)
        h = [hist.new_zeros(b, n, cell.hidden) for cell in self.enc]
        for i in range(t):
            x = torch.cat([hist[:, i], e], -1)
            for layer, cell in enumerate(self.enc):
                h[layer] = cell(x, h[layer], s)
                x = self.drop(h[layer])
        if self.attn is not None:
            h[-1] = self.attn(h[-1], buckets)
        return h

    def forward(self, hist, ctx, dec, start, s, buckets=None, teacher=None, teach=None, delta_scale=1.0):
        h = self.encode(hist, ctx, s, buckets)
        z = start * self.z0
        logits = []
        for step in range(dec.shape[1]):
            prev = torch.sigmoid(z).detach()
            if teacher is not None and step >= 2:
                prev = torch.where(teach[..., step - 2], teacher[..., step - 2], prev)
            x = torch.cat([dec[:, step], prev[..., None]], -1)
            for layer, cell in enumerate(self.dec):
                h[layer] = cell(x, h[layer], s)
                x = self.drop(h[layer])
            z = z + delta_scale * self.head(x)[..., 0]
            if step >= 1:
                logits.append(z)
        return torch.stack(logits, -1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
