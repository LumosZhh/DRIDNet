# CrossModal_Routing.py
# -*- coding: utf-8 -*-
"""
Routing + Cells + Classifier (merged)


"""

from __future__ import annotations
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from data import args


# ===========================
# ===========================
class classify(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=args.uni_dimension, out_channels=512, kernel_size=1, stride=1
        )
        self.linear1 = nn.Linear(512, args.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # [B,512,H,W]
        w = int(args.hsi_windowSize)
        k = int(round(0.4 * w))                    # α=0.4: 13→5, 15→7, 17→7...
        k = 3 if k < 3 else (w if k > w else k)    # clamp  [3, w]
        if k % 2 == 0:                             # ,
            k = k + 1 if k < w else k - 1
        x = F.avg_pool2d(x, kernel_size=k, stride=1)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)  # [B,num_classes]
        return x.squeeze(-1).squeeze(-1)

def activateFunc(x: torch.Tensor) -> torch.Tensor:
    x = torch.tanh(x)
    return F.relu(x)


# ===========================
# Sinkhorn-OT Router
# ===========================
class Router(nn.Module):
    """
    : x ∈ R^{B×C×H×W}
      4) Sinkhorn  (a,b) , → soft_g
    : soft_g ∈ R^{B×E}
    """
    def __init__(self, num_out_path: int, embed_size: int, hid: int,
                 d_model: int = 64, sinkhorn_iters: int = 7, epsilon: float | None = None):
        """
         args.router_eps; epsilon ().
        """
        super().__init__()
        self.E = num_out_path
        self.C = embed_size
        self.d = d_model
        self.eps = float(getattr(args, "router_eps", 0.03))
        self.eps = 1e-6 if self.eps <= 0 else self.eps
        self.iters = sinkhorn_iters

        self.src_embed  = nn.Parameter(torch.randn(self.C, self.d) * (1.0 / math.sqrt(self.d)))
        self.path_embed = nn.Parameter(torch.randn(self.E, self.d) * (1.0 / math.sqrt(self.d)))

        self.target_mlp = nn.Sequential(
            nn.Linear(self.C, hid),
            nn.ReLU(True),
            nn.Linear(hid, self.E),
        )
        with torch.no_grad():
            self.target_mlp[2].bias.fill_(1.5)

    @staticmethod
    def _softplus_norm(v: torch.Tensor, total: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        v = F.softplus(v) + eps
        s = v.sum(dim=-1, keepdim=True).clamp_min(eps)
        return v * (total / s)

    def _precompute_kernel(self, device: torch.device) -> torch.Tensor:
        src = self.src_embed
        dst = self.path_embed
        cost = (src.pow(2).sum(-1, keepdim=True)
                + dst.pow(2).sum(-1).unsqueeze(0)
                - 2.0 * (src @ dst.t())) / float(self.d)
        return torch.exp(-cost.to(device) / self.eps)  # [C,E]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,C]
        B = x.shape[0]
        x = x.mean(dim=(-1, -2))

        u = F.softplus(x) + 1e-6                    # [B,C]
        total_mass = u.sum(dim=-1, keepdim=True)    # [B,1]
        v0 = self.target_mlp(x)                     # [B,E]
        v = self._softplus_norm(v0, total_mass)     # [B,E]

        # Sinkhorn
        K = self._precompute_kernel(x.device)       # [C,E]
        a = torch.ones(B, self.C, device=x.device)
        b = torch.ones(B, self.E, device=x.device)
        eps = 1e-6
        for _ in range(self.iters):
            Kb  = (K @ b.T).T                       # [B,C]
            a   = u / Kb.clamp_min(eps)
            KTa = (K.T @ a.T).T                     # [B,E]
            b   = v / KTa.clamp_min(eps)

        s   = u @ self.src_embed                    # [B,d]
        sim = (s @ self.path_embed.T) / math.sqrt(self.d)  # [B,E]
        colsum = (K.T @ a.T).T * b                  # [B,E]
        fused  = 0.5 * colsum + 0.5 * F.softplus(sim + v0)

        return activateFunc(fused)


# ===========================
# ===========================
class CrossModalLinearAttention(nn.Module):
    """
     φ(t)=ELU(t)+1:
        out = [φ(q) (φ(k)^T v)] / [φ(q) (φ(k)^T 1)]
    :
    """
    def __init__(self, input_size: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.key   = nn.Linear(input_size, hidden_size, bias=True)
        self.query = nn.Linear(input_size, hidden_size, bias=True)
        self.value = nn.Linear(input_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.eps = eps
        self._proj_x: nn.Linear | None = None
        self._proj_y: nn.Linear | None = None

    @staticmethod
    def _phi(z: torch.Tensor) -> torch.Tensor:
        return F.elu(z, alpha=1.0) + 1.0

    def _maybe_project(self, t: torch.Tensor, target_dim: int, attr_name: str):
        Din = t.shape[-1]
        if Din == target_dim:
            return t
        proj = getattr(self, attr_name, None)
        if (proj is None) or (proj.in_features != Din) or (proj.out_features != target_dim):
            proj = nn.Linear(Din, target_dim, bias=False).to(t.device)
            setattr(self, attr_name, proj)
        return proj(t)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        in_feats = self.query.in_features
        x = self._maybe_project(x, in_feats, "_proj_x")
        y = self._maybe_project(y, in_feats, "_proj_y")

        q = self._phi(self.query(x))  # (..., Lq)
        k = self._phi(self.key(y))    # (..., Lk)
        v = self.value(y)

        kv   = torch.einsum('...ld,...le->...de', k, v)
        z    = torch.einsum('...ld,...de->...le', q, kv)
        ksum = k.sum(dim=-2, keepdim=True)  # (...,1,D)
        denom= torch.einsum('...ld,...le->...l', q, ksum).unsqueeze(-1).clamp_min(self.eps)
        out  = z / denom
        return self.dropout(out)


# ===========================
# ===========================
def _align_pair_last2(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """DRIDNet module."""
    if a.dim() == 4 and b.dim() == 4:
        Ha, Wa = a.shape[-2], a.shape[-1]
        Hb, Wb = b.shape[-2], b.shape[-1]
        if (Ha, Wa) != (Hb, Wb):
            Ht, Wt = max(Ha, Hb), max(Wa, Wb)
            a = F.interpolate(a, size=(Ht, Wt), mode='bilinear', align_corners=False)
            b = F.interpolate(b, size=(Ht, Wt), mode='bilinear', align_corners=False)
    return a, b


def _align_triple_last2(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DRIDNet module."""
    if a.dim() == 4 and b.dim() == 4 and c.dim() == 4:
        Ha, Wa = a.shape[-2], a.shape[-1]
        Hb, Wb = b.shape[-2], b.shape[-1]
        Hc, Wc = c.shape[-2], c.shape[-1]
        Ht, Wt = max(Ha, Hb, Hc), max(Wa, Wb, Wc)
        if (Ha, Wa) != (Ht, Wt):
            a = F.interpolate(a, size=(Ht, Wt), mode='bilinear', align_corners=False)
        if (Hb, Wb) != (Ht, Wt):
            b = F.interpolate(b, size=(Ht, Wt), mode='bilinear', align_corners=False)
        if (Hc, Wc) != (Ht, Wt):
            c = F.interpolate(c, size=(Ht, Wt), mode='bilinear', align_corners=False)
    return a, b, c


def _factor_pair_from_L(L: int, prefer: int | None = None) -> Tuple[int, int]:
    """
     L  (n, d), prefer(),
     sqrt(L) ; (L,1).
    """
    if prefer and prefer > 0 and L % prefer == 0:
        return L // prefer, prefer
    import math
    s = int(round(math.sqrt(L)))
    if s * s == L:
        return s, s
    for a in range(s, 0, -1):
        if L % a == 0:
            return a, L // a
    return L, 1


# ===========================
# ===========================
class Cell_1_0(nn.Module):
    def __init__(self, num_out_path: int):
        super().__init__()
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.sa = CrossModalLinearAttention(args.size_SA, args.size_SA)
        self.conv = nn.Conv2d(args.embed_size * 2, args.embed_size, kernel_size=1, stride=1)
        self.gate = nn.Sequential(
            nn.Linear(num_out_path, args.embed_size),
            nn.Sigmoid()
        )

    def forward(self, lidar: torch.Tensor, hsi: torch.Tensor):
        lidar, hsi = _align_pair_last2(lidar, hsi)

        path_prob = self.router(torch.concat((lidar, hsi), 1))
        l = rearrange(lidar, 'b h n d -> b h (n d)')
        h = rearrange(hsi,  'b h n d -> b h (n d)')
        h_emb = self.sa(l, h)  # (b, h, L)
        l_emb = self.sa(h, l)  # (b, h, L)

        L = h_emb.shape[-1]
        n, d = _factor_pair_from_L(L, prefer=args.hsi_windowSize)
        h_emb = rearrange(h_emb, 'b h (n d) -> b h n d', n=n, d=d)
        l_emb = rearrange(l_emb, 'b h (n d) -> b h n d', n=n, d=d)

        emb = self.conv(torch.concat((l_emb, h_emb), 1))
        g = self.gate(path_prob).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        emb = emb * g
        return emb, path_prob


class Cell_1(nn.Module):
    def __init__(self, num_out_path: int):
        super().__init__()
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.sa = CrossModalLinearAttention(args.size_SA, args.size_SA)
        self.conv1 = nn.Conv2d(args.embed_size * 2, args.embed_size, kernel_size=1, stride=1)
        self.gate = nn.Sequential(
            nn.Linear(num_out_path, args.embed_size),
            nn.Sigmoid()
        )

    def forward(self, lidar: torch.Tensor, hsi: torch.Tensor, x: torch.Tensor):
        lidar, hsi = _align_pair_last2(lidar, hsi)

        path_prob = self.router(torch.concat((lidar, hsi), 1))
        l = rearrange(lidar, 'b h n d -> b h (n d)')
        h = rearrange(hsi,  'b h n d -> b h (n d)')
        h_emb = self.sa(l, h)
        l_emb = self.sa(h, l)

        L = h_emb.shape[-1]
        n, d = _factor_pair_from_L(L, prefer=args.hsi_windowSize)
        h_emb = rearrange(h_emb, 'b h (n d) -> b h n d', n=n, d=d)
        l_emb = rearrange(l_emb, 'b h (n d) -> b h n d', n=n, d=d)

        out = self.conv1(torch.concat((l_emb, h_emb), 1))
        g = self.gate(path_prob).unsqueeze(-1).unsqueeze(-1)
        out = out * g

        if x.dim() == 4 and out.dim() == 4 and x.shape[-2:] != out.shape[-2:]:
            x = F.interpolate(x, size=out.shape[-2:], mode='bilinear', align_corners=False)
        emb = out + x
        return emb, path_prob


class Cell_2_0(nn.Module):
    def __init__(self, num_out_path: int):
        super().__init__()
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.sa = CrossModalLinearAttention(args.embed_size, args.embed_size)
        self.conv = nn.Conv2d(args.embed_size * 2, args.embed_size, kernel_size=1, stride=1)
        self.gate = nn.Sequential(
            nn.Linear(num_out_path, args.embed_size),
            nn.Sigmoid()
        )

    def forward(self, lidar: torch.Tensor, hsi: torch.Tensor):
        lidar, hsi = _align_pair_last2(lidar, hsi)

        path_prob = self.router(torch.concat((lidar, hsi), 1))
        l = rearrange(lidar, 'b h n d -> b (n d) h')
        h = rearrange(hsi,  'b h n d -> b (n d) h')
        h_emb = self.sa(l, h)  # (b, L, h)
        l_emb = self.sa(h, l)  # (b, L, h)

        L = h_emb.shape[1]
        n, d = _factor_pair_from_L(L, prefer=args.hsi_windowSize)
        h_emb = rearrange(h_emb, 'b (n d) h -> b h n d', n=n, d=d)
        l_emb = rearrange(l_emb, 'b (n d) h -> b h n d', n=n, d=d)

        emb = self.conv(torch.concat((l_emb, h_emb), 1))
        g = self.gate(path_prob).unsqueeze(-1).unsqueeze(-1)
        emb = emb * g
        return emb, path_prob


class Cell_2(nn.Module):
    def __init__(self, num_out_path: int):
        super().__init__()
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.sa = CrossModalLinearAttention(args.embed_size, args.embed_size)
        self.conv1 = nn.Conv2d(args.embed_size * 2, args.embed_size, kernel_size=1, stride=1)
        self.gate = nn.Sequential(
            nn.Linear(num_out_path, args.embed_size),
            nn.Sigmoid()
        )

    def forward(self, lidar: torch.Tensor, hsi: torch.Tensor, x: torch.Tensor):
        lidar, hsi = _align_pair_last2(lidar, hsi)

        path_prob = self.router(torch.concat((lidar, hsi), 1))
        l = rearrange(lidar, 'b c h w -> b (h w) c')
        h = rearrange(hsi,  'b c h w -> b (h w) c')
        h_emb = self.sa(l, h)
        l_emb = self.sa(h, l)

        L = h_emb.shape[1]
        h_side, w_side = _factor_pair_from_L(L, prefer=args.hsi_windowSize)
        h_emb = rearrange(h_emb, 'b (h w) c -> b c h w', h=h_side, w=w_side)
        l_emb = rearrange(l_emb, 'b (h w) c -> b c h w', h=h_side, w=w_side)

        out = self.conv1(torch.concat((l_emb, h_emb), 1))
        g = self.gate(path_prob).unsqueeze(-1).unsqueeze(-1)
        out = out * g

        if x.dim() == 4 and out.dim() == 4 and x.shape[-2:] != out.shape[-2:]:
            x = F.interpolate(x, size=out.shape[-2:], mode='bilinear', align_corners=False)
        emb = out + x
        return emb, path_prob


class Cell_3(nn.Module):
    def __init__(self, num_out_path: int):
        super().__init__()
        self.router = Router(num_out_path, args.embed_size * 3, args.hid_router)
        self.conv = nn.Sequential(
            nn.Conv2d(args.embed_size * 2, args.embed_size, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.embed_size)
        )
        self.gate = nn.Sequential(
            nn.Linear(num_out_path, args.embed_size),
            nn.Sigmoid()
        )

    def forward(self, lidar: torch.Tensor, hsi: torch.Tensor, pre: torch.Tensor):
        hsi, lidar, pre = _align_triple_last2(hsi, lidar, pre)
        path_prob = self.router(torch.concat((hsi, lidar, pre), 1))
        conv_out = self.conv(torch.concat((hsi, lidar), 1))
        g = self.gate(path_prob).unsqueeze(-1).unsqueeze(-1)
        sa_emb = conv_out * g + pre
        return sa_emb, path_prob


class Cell_3_0(nn.Module):
    def __init__(self, num_out_path: int):
        super().__init__()
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.conv = nn.Sequential(
            nn.Conv2d(args.embed_size * 2, args.embed_size, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.embed_size)
        )
        self.gate = nn.Sequential(
            nn.Linear(num_out_path, args.embed_size),
            nn.Sigmoid()
        )

    def forward(self, lidar: torch.Tensor, hsi: torch.Tensor):
        lidar, hsi = _align_pair_last2(lidar, hsi)
        path_prob = self.router(torch.concat((hsi, lidar), 1))
        conv_out = self.conv(torch.concat((hsi, lidar), 1))
        g = self.gate(path_prob).unsqueeze(-1).unsqueeze(-1)
        sa_emb = conv_out * g
        return sa_emb, path_prob


# ===========================
# Routing Module
# ===========================
class RoutingModule(nn.Module):
    """
    , Feature_Interaction  Feature_Interaction_*,.
    """
    def __init__(self, args, num_layer_routing: int = 5, path_hid: int = 128):
        super().__init__()
        self.args = args
        self.num_cells = num_cells = 3

        from dsfem import Feature_Interaction_Layer0, Feature_Interaction_Layern

        self.dynamic_itr_l0 = Feature_Interaction_Layer0(args, num_cells, num_cells)
        self.dynamic_itr_l1 = Feature_Interaction_Layern(args, num_cells, num_cells, stage=1)
        self.dynamic_itr_l2 = Feature_Interaction_Layern(args, num_cells, 1, stage=2)

        total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(args.embed_size)

    def forward(self, hsi_feats, lidar_feats):
        pairs_emb_lst, _ = self.dynamic_itr_l0(hsi_feats[0],  lidar_feats[0])
        pairs_emb_lst, _ = self.dynamic_itr_l1(pairs_emb_lst, hsi_feats[1], lidar_feats[1])
        pairs_emb_lst, p2 = self.dynamic_itr_l2(pairs_emb_lst, hsi_feats[2], lidar_feats[2])
        return pairs_emb_lst, p2
