# DRIDNet_prior.py
# -*- coding: utf-8 -*-
"""
All-in-one:  + DDPM +  + .

"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from data import args as _ext_args
from data import train_loader as _ext_train_loader
from data import test_loader as _ext_test_loader
from data import train_dataset as _ext_train_dataset

from ciidm import RoutingModule
from ciidm import classify as Classifier
from dsfem import HSI_Encoder as _HSI
from dsfem import Lidar_Encoder as _Lidar

# ============================================================
# 1) Fourier features / SIREN / MLP / IntrinsicCoordNet
# ============================================================
class FourierFeatureEncoding(nn.Module):
    def __init__(self, dim: int, num_freqs: int = 6, include_input: bool = True, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        freq = (2.0 ** torch.arange(num_freqs)) * (2.0 * math.pi * scale)
        self.register_buffer("freq_bands", freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs == 0:
            return x if self.include_input else x.new_zeros((x.shape[0], 0))
        freq = self.freq_bands.to(dtype=x.dtype, device=x.device)
        xb = x.unsqueeze(1) * freq.view(1, -1, 1)   # (N,F,D)
        sin = torch.sin(xb); cos = torch.cos(xb)
        feats = torch.cat([sin, cos], dim=1).flatten(1)
        if self.include_input: feats = torch.cat([x, feats], dim=1)
        return feats

def siren_uniform_(tensor, w0=1.0, in_features=None, is_first=False):
    if is_first: bound = 1.0 / (in_features if in_features else tensor.shape[1])
    else:        bound = math.sqrt(6.0 / (in_features if in_features else tensor.shape[1])) / w0
    with torch.no_grad(): return tensor.uniform_(-bound, bound)

class Sine(nn.Module):
    def __init__(self, w0=30.0): super().__init__(); self.w0 = w0
    def forward(self, x): return torch.sin(self.w0 * x)

class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int,
                 activation: str = "silu", use_siren: bool = False, w0: float = 30.0):
        super().__init__()
        if use_siren: activation = "sine"
        layers, last_dim = [], in_dim
        for _ in range(num_layers):
            lin = nn.Linear(last_dim, hidden_dim); layers.append(lin)
            if activation == "sine":   layers.append(Sine(w0=w0))
            elif activation == "relu": layers.append(nn.ReLU(inplace=True))
            elif activation == "silu": layers.append(nn.SiLU(inplace=True))
            elif activation == "tanh": layers.append(nn.Tanh())
            else: raise ValueError(f"Unknown activation {activation}")
            last_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.use_siren = use_siren; self.w0 = w0; self.activation = activation
        self._init_weights(in_dim, hidden_dim)

    def _init_weights(self, in_dim, hidden_dim):
        if self.use_siren:
            lin0 = self.net[0]
            siren_uniform_(lin0.weight, w0=self.w0, in_features=in_dim, is_first=True)
            nn.init.zeros_(lin0.bias)
            for m in self.net[1:]:
                if isinstance(m, nn.Linear):
                    siren_uniform_(m.weight, w0=self.w0, in_features=hidden_dim, is_first=False)
                    nn.init.zeros_(m.bias)
        else:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    if self.activation in ("relu", "silu"): nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    else:                                   nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class IntrinsicCoordNet(nn.Module):
    """DRIDNet module."""
    def __init__(self, use_intensity=True, pe_num_freqs=6, pe_include_input=True,
                 pe_scale=1.0, hidden_dim=256, num_layers=6, activation="silu",
                 use_siren=False, w0=30.0):
        super().__init__()
        self.use_intensity = use_intensity
        self.pe = FourierFeatureEncoding(3, pe_num_freqs, pe_include_input, pe_scale)
        pe_out_dim = 3 * (2 * pe_num_freqs) + (3 if pe_include_input else 0)
        in_dim = pe_out_dim + (1 if use_intensity else 0)
        self.backbone = MLPBlock(in_dim, hidden_dim, num_layers, activation, use_siren, w0)
        self.head_R = nn.Linear(hidden_dim, 1)
        self.head_S = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head_R.bias); nn.init.zeros_(self.head_S.bias)
        nn.init.xavier_uniform_(self.head_R.weight, gain=0.1)
        nn.init.xavier_uniform_(self.head_S.weight, gain=0.1)

    def forward(self, coords: torch.Tensor, logI: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.pe(coords)
        if self.use_intensity:
            assert logI is not None, "use_intensity=True but logI is None"
            feats = torch.cat([feats, logI], dim=1)
        h = self.backbone(feats)
        return self.head_R(h), self.head_S(h)

# ============================================================
# ============================================================
class SinPE(nn.Module):
    def __init__(self, d): super().__init__(); self.d=d
    def forward(self, t):
        half=self.d//2
        freqs=torch.exp(torch.arange(half, device=t.device)*(-math.log(10000)/(half-1)))
        a=t[:,None].float()*freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

class _Block(nn.Module):
    def __init__(self, ch, tdim):
        super().__init__()
        self.c=nn.Conv1d(ch,ch,3,padding=1)
        self.n=nn.GroupNorm(8,ch)
        self.a=nn.SiLU()
        self.f=nn.Linear(tdim,ch)
    def forward(self,x,t):
        h=self.c(x); h=self.n(h); h=h+self.f(t)[:,:,None]
        return self.a(h)

class UNet1D(nn.Module):
    def __init__(self, B=144, base=128, tdim=128):
        super().__init__()
        self.temb=nn.Sequential(SinPE(tdim), nn.Linear(tdim, tdim), nn.SiLU())
        self.inp=nn.Conv1d(1,base,1)
        self.b1=_Block(base,tdim); self.b2=_Block(base,tdim); self.b3=_Block(base,tdim)
        self.out=nn.Conv1d(base,1,1)
    def forward(self,x,t):
        t=self.temb(t); x=x[:,None,:]
        h=self.inp(x); h=self.b1(h,t); h=self.b2(h,t); h=self.b3(h,t)
        return self.out(h)[:,0,:]

def cosine_betas(T=1000):
    s=0.008
    u=torch.arange(T+1)
    a=torch.cos(((u/T)+s)/(1+s)*math.pi/2)**2
    a=a/a[0]
    b=1-(a[1:]/a[:-1])
    return torch.clamp(b,1e-5,0.999)

def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for k in ["model", "state_dict", "net", "ema", "ema_model"]:
            if k in obj and isinstance(obj[k], dict):
                maybe = obj[k]
                if "model" in maybe and isinstance(maybe["model"], dict):
                    return maybe["model"]
                return maybe
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format: cannot find a state_dict-like dict.")

def _load_weights_strict(module: nn.Module, sd: Dict[str, torch.Tensor], strict: bool=True):
    own = module.state_dict()
    matched = {k:v for k,v in sd.items() if (k in own and own[k].shape==v.shape)}
    miss = [k for k in own.keys() if k not in matched]
    extra = [k for k in sd.keys() if k not in own]
    module.load_state_dict(matched, strict=False if (len(miss)>0 or len(extra)>0) else strict)
    return len(matched), len(miss), len(extra)

class SpectralDDPMPrior(nn.Module):
    """DRIDNet module."""
    def __init__(self, ckpt_path: str, T: int = 1000, device: Optional[torch.device] = None, bands:int=144, verbose: bool=False):
        super().__init__()
        assert ckpt_path and isinstance(ckpt_path, str), "ckpt_path must be a valid file path string"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet1D(B=bands).to(self.device).eval()

        betas = cosine_betas(T).to(self.device)
        self.register_buffer('betas', betas, persistent=False)
        self.register_buffer('alphas', 1.0 - betas, persistent=False)
        self.register_buffer('abar', torch.cumprod(self.alphas, dim=0), persistent=False)
        self.register_buffer('sqrt_abar', torch.sqrt(self.abar), persistent=False)
        self.register_buffer('sqrt_1m_abar', torch.sqrt(1.0 - self.abar), persistent=False)

        state = torch.load(ckpt_path, map_location="cpu")
        sd = _extract_state_dict(state)
        n_ok, n_miss, n_extra = _load_weights_strict(self.net, sd, strict=True)
        for p in self.net.parameters(): p.requires_grad_(False)

    def _to_vecs(self, logR: torch.Tensor) -> torch.Tensor:
        if logR.dim() == 4:
            R = torch.exp(logR).permute(0,2,3,1).reshape(-1, logR.size(1))
        elif logR.dim() == 2:
            R = torch.exp(logR)
        else:
            raise ValueError(f"unexpected shape for logR: {tuple(logR.shape)}")
        return R.clamp_(0.0, 1.0)

    def prior_loss_on_R(self, logR: torch.Tensor, num_pix: int = 2048) -> torch.Tensor:
        self.net.eval()
        Rg = self._to_vecs(logR).requires_grad_(True)
        if Rg.size(0) > num_pix:
            idx = torch.randint(0, Rg.size(0), (num_pix,), device=Rg.device)
            Rg = Rg[idx]
        T = self.betas.numel()
        t = torch.randint(0, T, (Rg.size(0),), device=Rg.device, dtype=torch.long)
        eps = torch.randn_like(Rg)
        x_t = self.sqrt_abar[t][:, None] * Rg + self.sqrt_1m_abar[t][:, None] * eps
        eps_hat = self.net(x_t, t)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def denoise_R(self, logR: torch.Tensor, steps: int = 50) -> torch.Tensor:
        self.net.eval()
        N, C = self._to_vecs(logR).shape
        x = torch.randn(N, C, device=self.device)
        T = self.betas.numel()
        for k in reversed(range(T - steps, T)):
            t = torch.full((N,), k, device=self.device, dtype=torch.long)
            eps = self.net(x, t)
            beta_t = self.betas[k]; a_t = 1.0 - beta_t; abar_t = self.abar[k]
            mean = (1/torch.sqrt(a_t))*(x - (beta_t/torch.sqrt(1.0 - abar_t))*eps)
            x = mean + (torch.sqrt(beta_t)*torch.randn_like(x) if k > 0 else 0.0)
        R_ref = x.clamp(0.0, 1.0)
        return torch.log(R_ref.clamp_min(1e-6))

# ============================================================
# ============================================================
class DecompBlock(nn.Module):
    def __init__(self, bands: int, patch: int, amp: bool = True, pe_freqs: int = 4, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.net = IntrinsicCoordNet(
            use_intensity=True, pe_num_freqs=pe_freqs, pe_include_input=True, pe_scale=1.0,
            hidden_dim=hidden, num_layers=layers, activation='silu', use_siren=False, w0=30.0
        )
        self.amp = amp
        self.patch = patch
        self.bands = bands
        self.eps = 1e-6
        self.dsm_embed = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True)
        )
        self._coords_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def _mk_coords_hw(self, P: int, device):
        y = torch.linspace(0, P - 1, P, device=device)
        x = torch.linspace(0, P - 1, P, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return (xx / (P - 1)).view(1, 1, P, P), (yy / (P - 1)).view(1, 1, P, P)

    @torch.no_grad()
    def _get_cached_coords(self, B: int, C: int, P: int, device: torch.device, dtype: torch.dtype):
        """
         (B,C,P,P)  xx, yy, λ ;(xg,yg,λ_base)  (C,P,device.type) .
        """
        key = (C, P, device.type)
        if key not in self._coords_cache:
            y = torch.linspace(0, P - 1, P, device=device, dtype=dtype)
            x = torch.linspace(0, P - 1, P, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            xg = (xx / (P - 1)).view(1, 1, P, P)  # (1,1,P,P)
            yg = (yy / (P - 1)).view(1, 1, P, P)  # (1,1,P,P)
            lam = torch.linspace(0, 1, C, device=device, dtype=dtype).view(1, C, 1, 1)  # (1,C,1,1)
            self._coords_cache[key] = (xg, yg, lam)
        else:
            xg, yg, lam = self._coords_cache[key]
            if xg.dtype is not dtype:
                xg = xg.to(dtype)
                yg = yg.to(dtype)
                lam = lam.to(dtype)
                self._coords_cache[key] = (xg, yg, lam)

        xx = xg.expand(B, C, P, P)
        yy = yg.expand(B, C, P, P)
        lam_full = lam.expand(B, C, P, P)
        return xx, yy, lam_full

    def forward(self, hsi_bchw: torch.Tensor, bands_chunk: int = 32, dsm_patch: torch.Tensor | None = None):
        B, C, P, _ = hsi_bchw.shape
        assert P == self.patch and C == self.bands, f"Expected (C={self.bands}, P={self.patch}), got (C={C}, P={P})"
        device = hsi_bchw.device
        logI = torch.log(hsi_bchw.clamp_min(self.eps))

        xx, yy, lam_full = self._get_cached_coords(B, C, P, device, dtype=logI.dtype)
        coords_full = torch.stack([xx, yy, lam_full], dim=-1).reshape(-1, 3)
        logI_vec = logI.reshape(-1, 1)

        logR_out = torch.empty_like(logI)
        logS_out = torch.empty_like(logI)
        use_amp = (device.type == 'cuda') and self.amp
        amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            for b0 in range(0, C, bands_chunk):
                bc = min(bands_chunk, C - b0)
                mask = torch.zeros(C, device=device, dtype=torch.bool)
                mask[b0:b0 + bc] = True
                m = mask.view(1, C, 1, 1).expand(B, C, P, P).reshape(-1)
                coords = coords_full[m]
                logIv = logI_vec[m]
                logRv, logSv = self.net(coords, logIv)
                logRv = logRv.to(logR_out.dtype); logSv = logSv.to(logS_out.dtype)
                logR_out.reshape(-1, 1)[m] = logRv
                logS_out.reshape(-1, 1)[m] = logSv

        if dsm_patch is not None:
            try:
                if dsm_patch.dim() == 3:
                    dsm_patch = dsm_patch.unsqueeze(1)  # (B,1,P,P)
                bias = self.dsm_embed(dsm_patch)        # (B,1,P,P)
                bias = bias.expand_as(logS_out)         # (B,C,P,P)
                logS_out = logS_out + bias
            except Exception:
                pass

        return logR_out, logS_out

    def compute_losses(self, hsi, logR, logS, dsm_patch=None, w_spec2=True):
        eps = 1e-6
        logI = torch.log(hsi.clamp_min(eps))
        recon = F.l1_loss(logR + logS, logI)
        d1 = (logR[:, 1:] - logR[:, :-1]).abs().mean()
        d2 = (logR[:, 2:] - 2 * logR[:, 1:-1] + logR[:, :-2]).abs().mean() if w_spec2 else logR.new_zeros(())
        tv = (logR[..., 1:, :] - logR[..., :-1, :]).abs().mean() + (logR[..., :, 1:] - logR[..., :, :-1]).abs().mean()
        if dsm_patch is not None:
            if dsm_patch.dim() == 3:
                dsm_patch = dsm_patch.unsqueeze(1)
            gx = (logS[..., :, 1:] - logS[..., :, :-1])
            gy = (logS[..., 1:, :] - logS[..., :-1, :])
            dsm_w = torch.sigmoid(F.avg_pool2d(dsm_patch, 3, 1, 1))
            dsm_term = (dsm_w[..., :, 1:] * gx.abs()).mean() + (dsm_w[..., 1:, :] * gy.abs()).mean()
        else:
            dsm_term = logR.new_zeros(())
        return dict(recon=recon, spec=d1, spec2=d2, local=tv, dsm=dsm_term)


# ============================================================
# ============================================================
class ReflectanceMixer(nn.Module):
    def __init__(self, mode: str = "blend", gate_logit_init: float = -4.0):
        super().__init__()
        self.mode = mode
        self.alpha = nn.Parameter(torch.tensor(float(gate_logit_init)))
    def forward(self, hsi: torch.Tensor, logR_detached: torch.Tensor) -> torch.Tensor:
        if self.mode != "blend": return hsi
        gate = torch.sigmoid(self.alpha)
        R = torch.exp(logR_detached)
        return hsi + gate * (R - hsi)


class PostCIIDMPCA(nn.Module):
    """Fixed whitened PCA applied after CIIDM and reflectance residual reinjection."""
    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-12):
        super().__init__()
        if out_channels > in_channels:
            raise ValueError(
                f"PCA output channels ({out_channels}) cannot exceed input channels ({in_channels})."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.eps = float(eps)

        self.register_buffer("mean", torch.zeros(self.in_channels))
        self.register_buffer("components", torch.zeros(self.out_channels, self.in_channels))
        self.register_buffer("explained_variance", torch.ones(self.out_channels))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit_from_statistics(
        self,
        num_samples: int,
        sum_x: torch.Tensor,
        sum_xx: torch.Tensor,
    ):
        if int(num_samples) <= 1:
            raise ValueError("At least two post-CIIDM pixels are required to fit PCA.")

        n = int(num_samples)
        sum_x = sum_x.to(dtype=torch.float64, device="cpu")
        sum_xx = sum_xx.to(dtype=torch.float64, device="cpu")

        mean = sum_x / float(n)
        covariance = (sum_xx - float(n) * torch.outer(mean, mean)) / float(n - 1)
        covariance = 0.5 * (covariance + covariance.T)

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order][:self.out_channels].clamp_min(self.eps)
        components = eigenvectors[:, order][:, :self.out_channels].T.contiguous()

        # Fix each component sign deterministically. PCA component signs are otherwise arbitrary.
        max_abs_index = torch.argmax(torch.abs(components), dim=1)
        signs = components[
            torch.arange(self.out_channels, dtype=torch.long),
            max_abs_index,
        ].sign()
        signs[signs == 0] = 1
        components = components * signs.unsqueeze(1)

        self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
        self.components.copy_(components.to(device=self.components.device, dtype=self.components.dtype))
        self.explained_variance.copy_(
            eigenvalues.to(device=self.explained_variance.device, dtype=self.explained_variance.dtype)
        )
        self.fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError("Post-CIIDM PCA has not been fitted or loaded from a checkpoint.")

        # x: (..., C, H, W) -> (..., H, W, C)
        x_last = x.movedim(-3, -1)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        components = self.components.to(device=x.device, dtype=x.dtype)
        variance = self.explained_variance.to(device=x.device, dtype=x.dtype)

        y = torch.matmul(x_last - mean, components.T)
        y = y / torch.sqrt(variance.clamp_min(self.eps))
        return y.movedim(-1, -3)


# ============================================================
# ============================================================
@dataclass
class DatasetCfg:
    ddpm_ckpt: Optional[str] = None
    prior_refine_steps: int = 0
    prior_loss_pixels: int = 2048
    lambda_gen: float = 0.0

def get_dataset_cfg(name: str) -> DatasetCfg:
    return DatasetCfg()

# ============================================================
# ============================================================
def _build_encoders():
    return _HSI(), _Lidar()

class RGRFNet(nn.Module):
    def __init__(self, args=None, train_loader=None, test_loader=None, dataset_selected: Optional[str]=None):
        super().__init__()
        self.args = args if args is not None else _ext_args
        if self.args is None:
            raise RuntimeError(" Data_Prepare  args,/ Data_Prepare.py.")

        self.hsi_enc, self.lidar_enc = _build_encoders()
        self.routing_module = RoutingModule(self.args)
        self.classifier = Classifier()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        self.train_loader = train_loader if train_loader is not None else _ext_train_loader
        self.test_loader  = test_loader  if test_loader  is not None else _ext_test_loader
        if self.train_loader is None or self.test_loader is None:
            raise RuntimeError(" Data_Prepare  train_loader/test_loader,/ Data_Prepare.py.")

        self.dataset_selected = dataset_selected if dataset_selected is not None else _ext_train_dataset

        self.last_train_loss_avg = None
        self.last_train_acc = None
        self.last_batch_loss = None
        self.last_val_loss_avg = None
        self.last_val_acc = None
        self.last_metrics = None
        self.last_cls_report = None

    def _device(self):
        return next(self.parameters()).device

    def forward(self, hsi, lidar):
        logits, _ = self._forward_logits(hsi, lidar)
        return logits

    def _forward_logits(self, hsi, lidar):
        e_hsi = self.hsi_enc(hsi)
        e_lidar = self.lidar_enc(lidar)
        fused_list, paths = self.routing_module(e_hsi, e_lidar)
        logits = self.classifier(fused_list[0])
        return logits, paths

    def train(self, mode=True):
        if isinstance(mode, bool): return super().train(mode)
        epoch = int(mode); return self._train_one_epoch(epoch)

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            hsi   = batch.get("hsi") or batch.get("data_hsi") or batch.get("HSI") or batch.get("hsi_patches")
            lidar = batch.get("lidar") or batch.get("data_lidar") or batch.get("DSM") or batch.get("lidar_patches")
            labels= batch.get("y") or batch.get("label") or batch.get("gt") or batch.get("labels")
            return hsi, lidar, labels
        if isinstance(batch, (list, tuple)):
            hsi, lidar = batch[0], batch[1]
            labels = batch[2] if len(batch) >= 3 else None
            return hsi, lidar, labels
        raise TypeError(f"Unsupported batch format: {type(batch)}")

    def _train_one_epoch(self, epoch: int):
        super().train(True)
        dev = self._device()
        running_loss = running_correct = running_samples = 0

        for batch in self.train_loader:
            hsi, lidar, labels = self._unpack_batch(batch)
            hsi = hsi.to(dev); lidar = lidar.to(dev); labels = labels.to(dev)

            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self._forward_logits(hsi, lidar)
            loss = self.criterion(logits, labels)

            if hasattr(self, "_last_decomp") and (self._last_decomp is not None) and hasattr(self, "decomp_block") and (self.decomp_block is not None):
                hsi_f, logR_f, logS_f, lidar_f = self._last_decomp
                decomp = self.decomp_block.compute_losses(hsi_f, logR_f, logS_f, dsm_patch=lidar_f)

                lr   = float(getattr(self.args, "lambda_recon",  0.0))
                ls1  = float(getattr(self.args, "lambda_spec",   0.0))
                ls2  = float(getattr(self.args, "lambda_spec2",  0.0))
                lloc = float(getattr(self.args, "lambda_local",  0.0))
                ldsm = float(getattr(self.args, "lambda_dsm",    0.00))

                loss = loss + (lr*decomp['recon'] + ls1*decomp['spec'] + ls2*decomp['spec2']
                               + lloc*decomp['local'] + ldsm*decomp['dsm'])

                lam_gen = float(getattr(self.args, "lambda_gen", 0.0))
                if lam_gen == 0.0 and hasattr(self, "ds_cfg"):
                    lam_gen = float(getattr(self.ds_cfg, "lambda_gen", 0.0))
                if lam_gen > 0.0 and hasattr(self, "prior") and (self.prior is not None):
                    num_pix = int(getattr(self.args, "prior_loss_pixels",
                                getattr(self.ds_cfg, "prior_loss_pixels", 2048)))
                    prior_loss = self.prior.prior_loss_on_R(logR_f, num_pix=num_pix)
                    loss = loss + lam_gen * prior_loss
                    

                lam_anchor = float(getattr(self.args, "lambda_anchor", 0.0))
                if lam_anchor > 0.0 and hasattr(self, "decomp_anchor") and (self.decomp_anchor is not None):
                    l2 = 0.0
                    for n, p in self.decomp_block.net.named_parameters():
                        if p.requires_grad and (n in self.decomp_anchor):
                            l2 = l2 + (p - self.decomp_anchor[n]).pow(2).mean()
                    loss = loss + lam_anchor * l2

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                bs = labels.numel()
                running_loss   += float(loss.item()) * bs
                running_correct+= correct
                running_samples+= bs
                self.last_batch_loss = float(loss.item())

        self.last_train_loss_avg = running_loss / max(1, running_samples)
        self.last_train_acc = running_correct / max(1, running_samples)
        return self

    def train_start(self): super().train(True)
    @torch.no_grad()
    def val_start(self): self.eval()

    @torch.no_grad()
    def cal_acc(self, epoch: int):
        self.eval()
        dev = self._device()
        total_loss = total_samples = 0
        y_true, y_pred = [], []

        for batch in self.test_loader:
            hsi, lidar, labels = self._unpack_batch(batch)
            hsi = hsi.to(dev); lidar = lidar.to(dev); labels = labels.to(dev)
            logits, _ = self._forward_logits(hsi, lidar)
            loss = self.criterion(logits, labels)
            total_loss += float(loss.item()) * labels.numel()
            total_samples += labels.numel()
            y_true.append(labels.detach().cpu().numpy().reshape(-1))
            y_pred.append(logits.argmax(dim=1).detach().cpu().numpy().reshape(-1))

        y_true = np.concatenate(y_true, 0) if y_true else np.empty((0,), dtype=np.int64)
        y_pred = np.concatenate(y_pred, 0) if y_pred else np.empty((0,), dtype=np.int64)
        OA = (y_true == y_pred).mean() if y_true.size else 0.0

        num_classes = int(self.args.num_classes)
        AA_list = []
        for c in range(num_classes):
            idx = (y_true == c)
            AA_list.append((y_pred[idx] == c).mean() if idx.sum() > 0 else 0.0)
        AA_mean = float(np.mean(AA_list))

        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred): cm[t, p] += 1
        N = cm.sum(); po = np.trace(cm) / max(1, N)
        pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / max(1, N*N)
        Kappa = (po - pe) / max(1e-12, 1 - pe)

        target_names = [f"Class_{i}" for i in range(num_classes)]
        report = classification_report(
            y_true, y_pred,
            labels=list(range(num_classes)),
            target_names=target_names, digits=4, zero_division=0
        )

        self.last_val_loss_avg = total_loss / max(1, total_samples)
        self.last_val_acc = float(OA)
        self.last_metrics = {"OA": float(OA), "AA_mean": float(AA_mean), "Kappa": float(Kappa)}
        self.last_cls_report = report
        self.lr_scheduler.step(self.last_val_acc)
        return float(OA)

# ============================================================
# ============================================================
class DRIDNet_HSI(RGRFNet):
    def __init__(self, intrinsic_ckpt: Optional[str] = None, ddpm_ckpt: Optional[str] = None,
                 args=None, train_loader=None, test_loader=None, dataset_selected: Optional[str]=None):
        super().__init__(args=args, train_loader=train_loader, test_loader=test_loader,
                         dataset_selected=dataset_selected)
        self.decomp_block: Optional[DecompBlock] = None
        self._intrinsic_ckpt_path = intrinsic_ckpt
        self._intrinsic_loaded = False

        ds_name = str(dataset_selected or getattr(self.args, "dataset", "Houston2013"))
        self.ds_cfg = get_dataset_cfg(ds_name)
        self._ddpm_ckpt_path = ddpm_ckpt or getattr(self.args, "ddpm_ckpt", self.ds_cfg.ddpm_ckpt)

        self._prior_loaded = False
        self.prior: Optional[SpectralDDPMPrior] = None

        self._last_decomp = None
        self.decomp_anchor = None
        self.post_ciidm_pca: Optional[PostCIIDMPCA] = None

    def _sync_scheduler_groups(self):
        sch = getattr(self, "lr_scheduler", None)
        if sch is None or getattr(sch, "optimizer", None) is not self.optimizer: return
        if hasattr(sch, "min_lrs"):
            tgt = len(self.optimizer.param_groups)
            cur = len(sch.min_lrs)
            if tgt > cur:
                last_min = sch.min_lrs[-1] if cur > 0 else 0.0
                sch.min_lrs.extend([last_min] * (tgt - cur))
        if hasattr(sch, "_last_lr"):
            sch._last_lr = [pg["lr"] for pg in self.optimizer.param_groups]

    def _add_params_to_optim(self, module: nn.Module, lr_fallback=1e-3, lr_scale=0.1):
        if hasattr(self, "optimizer") and len(self.optimizer.param_groups) > 0:
            base = self.optimizer.param_groups[0]
            self.optimizer.add_param_group({
                "params": module.parameters(),
                "lr": base.get("lr", lr_fallback) * lr_scale,
                "weight_decay": base.get("weight_decay", 0.0),
            })
            self._sync_scheduler_groups()

    def _lazy_build_and_load(self, C: int, P: int, device: torch.device):
        if self.decomp_block is None:
            self.decomp_block = DecompBlock(bands=C, patch=P, amp=True).to(device)
            self._add_params_to_optim(self.decomp_block, lr_scale=0.1)

        if self.post_ciidm_pca is None:
            self.post_ciidm_pca = PostCIIDMPCA(
                in_channels=C,
                out_channels=int(self.args.channels),
            ).to(device)

        if (not self._intrinsic_loaded) and self._intrinsic_ckpt_path:
            try:
                state = torch.load(self._intrinsic_ckpt_path, map_location="cpu")
                sd = None
                if isinstance(state, dict):
                    for k in ["state_dict", "model", "net", "decomp", "intrinsic"]:
                        if k in state and isinstance(state[k], dict):
                            sd = state[k]; break
                if sd is None and isinstance(state, dict):
                    sd = {k: v for k, v in state.items() if torch.is_tensor(v)}
                if sd is None: sd = state

                my_sd = self.decomp_block.net.state_dict()
                filt = {k: v for k, v in sd.items() if k in my_sd and my_sd[k].shape == v.shape}
                self.decomp_block.net.load_state_dict(filt, strict=False)

                self.decomp_anchor = {k: w.detach().clone() for k, w in self.decomp_block.net.state_dict().items()}
            except Exception:
                pass
            finally:
                self._intrinsic_loaded = True

        if (not self._prior_loaded):
            if self._ddpm_ckpt_path:
                try:
                    bands = int(C)
                    self.prior = SpectralDDPMPrior(
                        self._ddpm_ckpt_path, T=1000, device=device, bands=bands, verbose=False
                    ).eval()
                except Exception:
                    pass
                finally:
                    self._prior_loaded = True
            else:
                self._prior_loaded = True

    def _post_ciidm_hsi(self, hsi, lidar):
        orig = hsi.shape
        C, P = hsi.size(-3), hsi.size(-1)
        B = 1
        for d in orig[:-3]: B *= int(d)
        hsi_f   = hsi.reshape(B, C, P, P)
        lidar_f = lidar.reshape(B, lidar.size(-3), P, P)

        self._lazy_build_and_load(C, P, hsi.device)

        logR_f, logS_f = self.decomp_block(hsi_f, dsm_patch=lidar_f)

        steps = int(getattr(self.args, "prior_refine_steps", 0))
        if steps > 0 and self.prior is not None and (not self.training):
            with torch.no_grad():
                logR_f = self.prior.denoise_R(logR_f, steps=steps)

        self._last_decomp = (hsi_f, logR_f, logS_f, lidar_f)

        # if not hasattr(self, "mixer") or self.mixer is None:
        #     self.mixer = ReflectanceMixer(mode=getattr(self.args, "intrinsic_mode", "blend")).to(hsi.device)
        # logR = logR_f.reshape(orig)
        # hsi_eff = self.mixer(hsi, logR.detach())
        if not hasattr(self, "mixer") or self.mixer is None:
            gate_logit = float(getattr(self.args, "intrinsic_gate_logit", -4.0))
            self.mixer = ReflectanceMixer(
                mode=getattr(self.args, "intrinsic_mode", "blend"),
                gate_logit_init=gate_logit
            ).to(hsi.device)
        else:
            new_logit = float(getattr(self.args, "intrinsic_gate_logit", -4.0))
            if abs(new_logit - float(self.mixer.alpha.item())) > 1e-6:
                with torch.no_grad():
                    self.mixer.alpha.fill_(new_logit)

        logR = logR_f.reshape(orig)
        return self.mixer(hsi, logR.detach())

    @torch.no_grad()
    def extract_post_ciidm_hsi(self, hsi, lidar):
        return self._post_ciidm_hsi(hsi, lidar)

    def _forward_logits(self, hsi, lidar):
        hsi_eff = self._post_ciidm_hsi(hsi, lidar)
        hsi_pca = self.post_ciidm_pca(hsi_eff)

        e_hsi   = self.hsi_enc(hsi_pca)
        e_lidar = self.lidar_enc(lidar)
        fused_list, paths = self.routing_module(e_hsi, e_lidar)
        logits = self.classifier(fused_list[0])
        return logits, paths

# ============================================================
# ============================================================
def set_runtime_context(args_obj, train_loader_obj, test_loader_obj, train_dataset_name: Optional[str] = None):
    global _ext_args, _ext_train_loader, _ext_test_loader, _ext_train_dataset
    _ext_args = args_obj
    _ext_train_loader = train_loader_obj
    _ext_test_loader = test_loader_obj
    _ext_train_dataset = train_dataset_name
