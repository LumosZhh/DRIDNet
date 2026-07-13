
"""
Dual-Stream Feature Encoders + Interaction Module (DSFEM)


10.14Houston2013 Trentochannels
"""

from __future__ import annotations
from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from data import args, train_dataset
from ciidm import Cell_1_0, Cell_2_0, Cell_3_0, Cell_1, Cell_2, Cell_3


# =============================================================================
# =============================================================================
def _ensure_hsi_5d(x: torch.Tensor) -> torch.Tensor:
    """DRIDNet module."""
    if x.dim() == 5:
        return x
    if x.dim() == 4:
        return x.unsqueeze(1)
    raise ValueError(f"HSI input must be 4D/5D, got shape={tuple(x.shape)}")


def _ensure_lidar_4d(x: torch.Tensor) -> torch.Tensor:
    """DRIDNet module."""
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        return x.unsqueeze(1)
    raise ValueError(f"LiDAR input must be 3D/4D, got shape={tuple(x.shape)}")


def _unsqueeze2d(x: torch.Tensor) -> torch.Tensor:
    """DRIDNet module."""
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    t = x
    while t.dim() < 4:
        t = t.unsqueeze(-1)
    if t.shape[1] != 1:
        t = t.mean(dim=1, keepdim=True)
    return t


def _ensure_hw_to_target(x: torch.Tensor, target: int) -> torch.Tensor:
    """DRIDNet module."""
    x = _unsqueeze2d(x)
    if x.shape[-2:] != (target, target):
        x = F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)
    return x


def _align_path_to_target(cur_path: torch.Tensor, out_channels: int, target: int) -> torch.Tensor:
    """
     (B,out_channels,target,target):
    """
    cur_path = _unsqueeze2d(cur_path)  # (B,Cp,Hp,Wp)

    if cur_path.shape[-2:] != (target, target):
        cur_path = F.interpolate(cur_path, size=(target, target), mode='bilinear', align_corners=False)

    cp = cur_path.shape[1]
    ce = out_channels
    if cp == 1 and ce > 1:
        cur_path = cur_path.expand(-1, ce, -1, -1)
    elif cp != ce:
        cur_path = cur_path.mean(dim=1, keepdim=True).expand(-1, ce, -1, -1)

    return cur_path


def _pool_to(x: torch.Tensor, size: int) -> torch.Tensor:
    """DRIDNet module."""
    if x.shape[-2:] != (size, size):
        x = F.adaptive_avg_pool2d(x, (size, size))
    return x




# =============================================================================
# =============================================================================
class HSI_Encoder_Houston2013(nn.Module):
    """
    3D  (9,7,5) +  3×3 valid;head  (7,5,3).
    "":8*S1 / 16*S2 / 32*S3.
    """
    def __init__(self):
        super().__init__()
        self.hsi_step1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(9, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),
        )
        self.hsi_step2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(7, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(16),
        )
        self.hsi_step3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
        )

        S0 = int(args.channels)
        S1 = S0 - 9 + 1
        S2 = S1 - 7 + 1
        S3 = S2 - 5 + 1
        if min(S1, S2, S3) <= 0:
            raise ValueError(f"[Houston2013] args.channels={S0} ; S3>=1 ⇒ channels>=19")

        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(8 * S1, args.uni_dimension, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(16 * S2, args.uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv2d(32 * S3, args.uni_dimension, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        X: List[torch.Tensor] = []
        x = _ensure_hsi_5d(x)

        # step1
        x1 = self.hsi_step1(x)                                   # (B,8,S1,H1,W1)
        B, G1, C1, H1, W1 = x1.shape
        x1f = x1.reshape(B, G1 * C1, H1, W1)                     # , 1×1  8*22
        x1f = _pool_to(x1f, 7)
        X.append(self.hsi_conv1(x1f))

        # step2
        x2 = self.hsi_step2(x1)                                  # (B,16,S2,H2,W2)
        B, G2, C2, H2, W2 = x2.shape
        x2f = x2.reshape(B, G2 * C2, H2, W2)
        x2f = _pool_to(x2f, 5)
        X.append(self.hsi_conv2(x2f))

        # step3
        x3 = self.hsi_step3(x2)                                  # (B,32,S3,H3,W3)
        B, G3, C3, H3, W3 = x3.shape
        x3f = x3.reshape(B, G3 * C3, H3, W3)
        x3f = _pool_to(x3f, 3)
        X.append(self.hsi_conv3(x3f))

        return X

class Lidar_Encoder_Houston2013(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, args.lidar_channel, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.lidar_channel),
        )
        self.out_conv1 = nn.Sequential(
            nn.Conv2d(64, args.uni_dimension, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(128, args.uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        X: List[torch.Tensor] = []
        x = _ensure_lidar_4d(x)
        x1 = self.conv1(x)
        x1 = _pool_to(x1, 7)
        X.append(self.out_conv1(x1))
        x2 = self.conv2(x1)
        x2 = _pool_to(x2, 5)
        X.append(self.out_conv2(x2))
        x3 = self.conv3(x2)
        X.append(x3)
        return X


# =============================================================================
# =============================================================================
class HSI_Encoder_Houston2018(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsi_step1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(5, 5, 5), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=8),
        )
        self.hsi_step2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=16),
        )
        self.hsi_step3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=32),
        )

        S0 = int(args.channels)         # (PCA )
        S1 = S0 - 5 + 1                 #  5 
        S2 = S1 - 3 + 1                 #  3 
        S3 = S2 - 3 + 1                 #  3 
        if not (S1 > 0 and S2 > 0 and S3 > 0):
            raise ValueError(f"[Houston2018] args.channels={S0} , S3>=1 ⇒ channels>=9")

        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8 * S1, out_channels=args.uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16 * S2, out_channels=args.uni_dimension, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32 * S3, out_channels=args.uni_dimension, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
        )

    def forward(self, x):
        X = []
        x = _ensure_hsi_5d(x)

        x1 = self.hsi_step1(x)  # [B,8,S1,H1,W1]
        X1_in = x1.reshape(-1, x1.shape[1] * x1.shape[2], x1.shape[3], x1.shape[4])
        X.append(self.hsi_conv1(X1_in))

        x2 = self.hsi_step2(x1) # [B,16,S2,H2,W2]
        X2_in = x2.reshape(-1, x2.shape[1] * x2.shape[2], x2.shape[3], x2.shape[4])
        X.append(self.hsi_conv2(X2_in))

        x3 = self.hsi_step3(x2) # [B,32,S3,H3,W3]
        X3_in = x3.reshape(-1, x3.shape[1] * x3.shape[2], x3.shape[3], x3.shape[4])
        X.append(self.hsi_conv3(X3_in))
        return X


class Lidar_Encoder_Houston2018(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=1, out_channels=64, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(kernel_size=5, in_channels=64, out_channels=args.lidar_channel, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.lidar_channel),
        )
        self.out_conv1 = nn.Sequential(
            nn.Conv2d(kernel_size=7, in_channels=64, out_channels=args.uni_dimension, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
        )
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(kernel_size=5, in_channels=64, out_channels=args.uni_dimension, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
        )

    def forward(self, x):
        X = []
        x = _ensure_lidar_4d(x)
        x1 = self.conv1(x)
        X.append(self.out_conv1(x1))
        x2 = self.conv2(x1)
        X.append(self.out_conv2(x2))
        x3 = self.conv3(x2)
        X.append(x3)
        return X



# =============================================================================
# MUUFL
# =============================================================================
class HSI_Encoder_Muufl(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsi_step1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7, 3, 3), padding=(3, 0, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(16),
        )
        self.hsi_step2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=(2, 0, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
        )
        self.hsi_step3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
        )
        self.pool_spec = nn.AdaptiveAvgPool3d((1, None, None))

        self.head1 = nn.Sequential(
            nn.Conv2d(16, args.uni_dimension, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.uni_dimension),
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(32, args.uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.uni_dimension),
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(64, args.uni_dimension, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.uni_dimension),
        )

    def forward(self, x):
        X = []
        x = _ensure_hsi_5d(x)              # [B,1,C,H,W],C  56/64/10 

        x1 = self.hsi_step1(x)              # [B,16,C,H',W'],
        x1p = self.pool_spec(x1).squeeze(2) # [B,16,H',W']
        X.append(self.head1(x1p))

        x2 = self.hsi_step2(x1)             # [B,32,C,H'',W'']
        x2p = self.pool_spec(x2).squeeze(2)
        X.append(self.head2(x2p))

        x3 = self.hsi_step3(x2)             # [B,64,C,H''',W''']
        x3p = self.pool_spec(x3).squeeze(2)
        X.append(self.head3(x3p))
        return X

class Lidar_Encoder_Muufl(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, args.lidar_channel, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.lidar_channel),
        )
        self.out_conv1 = nn.Sequential(
            nn.Conv2d(64, args.uni_dimension, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.uni_dimension),
        )
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(128, args.uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.uni_dimension),
        )

    def forward(self, x):
        X = []
        x = _ensure_lidar_4d(x)  # [B,1,H,W]
        x1 = self.conv1(x)
        X.append(self.out_conv1(x1))
        x2 = self.conv2(x1)
        X.append(self.out_conv2(x2))
        x3 = self.conv3(x2)
        X.append(x3)
        return X


# =============================================================================
# Trento 
# =============================================================================
class HSI_Encoder_Trento(nn.Module):
    """
    (1,3,3),head (5,3,1); 5/3/1.
    :Sg = (args.trento_spec_groups > 0 ?  : args.channels).
     C != Sg, 1×1  (G*C) → (G*Sg).
    """
    def __init__(self):
        super().__init__()

        self.hsi_step1 = nn.Sequential(
            nn.Conv3d(1, 8,  kernel_size=(1, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),
        )
        self.hsi_step2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(16),
        )
        self.hsi_step3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
        )

        C0 = int(args.channels)
        Sg_cfg = int(getattr(args, "trento_spec_groups", -1))
        Sg = Sg_cfg if Sg_cfg > 0 else C0
        if Sg < 1:
            raise ValueError(f"[Trento]  Sg={Sg}")
        self.Sg = Sg

        self.hsi_trento_proj1 = None  # (8*C → 8*Sg)
        self.hsi_trento_proj2 = None  # (16*C → 16*Sg)
        self.hsi_trento_proj3 = None  # (32*C → 32*Sg)

        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(8 * Sg,  args.uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(16 * Sg, args.uni_dimension, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv2d(32 * Sg, args.uni_dimension, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        X: List[torch.Tensor] = []
        x = _ensure_hsi_5d(x)  # [B,1,C,H,W]

        # ---- step1 ----
        x1 = self.hsi_step1(x)                     # [B,8,C,H',W']
        B, G1, C1, H1, W1 = x1.shape
        x1f = x1.reshape(B, G1 * C1, H1, W1)       # -> [B, 8*C1, H', W']
        if C1 != self.Sg:
            if (self.hsi_trento_proj1 is None) or (self.hsi_trento_proj1.in_channels != x1f.shape[1]):
                self.hsi_trento_proj1 = nn.Conv2d(x1f.shape[1], 8 * self.Sg, kernel_size=1, bias=False).to(x1.device)
            x1f = self.hsi_trento_proj1(x1f)       #  8*Sg
        x1f = _pool_to(x1f, 5)                     #  5×5
        X.append(self.hsi_conv1(x1f))

        # ---- step2 ----
        x2 = self.hsi_step2(x1)                    # [B,16,C,H'',W'']
        B, G2, C2, H2, W2 = x2.shape
        x2f = x2.reshape(B, G2 * C2, H2, W2)       # -> [B, 16*C2, H'', W'']
        if C2 != self.Sg:
            if (self.hsi_trento_proj2 is None) or (self.hsi_trento_proj2.in_channels != x2f.shape[1]):
                self.hsi_trento_proj2 = nn.Conv2d(x2f.shape[1], 16 * self.Sg, kernel_size=1, bias=False).to(x2.device)
            x2f = self.hsi_trento_proj2(x2f)       #  16*Sg
        x2f = _pool_to(x2f, 3)                     # → 3×3
        X.append(self.hsi_conv2(x2f))

        # ---- step3 ----
        x3 = self.hsi_step3(x2)                    # [B,32,C,H''',W''']
        B, G3, C3, H3, W3 = x3.shape
        x3f = x3.reshape(B, G3 * C3, H3, W3)       # -> [B, 32*C3, H''', W''']
        if C3 != self.Sg:
            if (self.hsi_trento_proj3 is None) or (self.hsi_trento_proj3.in_channels != x3f.shape[1]):
                self.hsi_trento_proj3 = nn.Conv2d(x3f.shape[1], 32 * self.Sg, kernel_size=1, bias=False).to(x3.device)
            x3f = self.hsi_trento_proj3(x3f)       #  32*Sg
        x3f = _pool_to(x3f, 1)                     # → 1×1
        X.append(self.hsi_conv3(x3f))

        return X

class Lidar_Encoder_Trento(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, args.lidar_channel, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.lidar_channel),
        )
        self.out_conv1 = nn.Sequential(
            nn.Conv2d(64, args.uni_dimension, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(64, args.uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        X: List[torch.Tensor] = []
        x = _ensure_lidar_4d(x)
        x1 = self.conv1(x)
        x1 = _pool_to(x1, 7)
        X.append(self.out_conv1(x1))
        x2 = self.conv2(x1)
        x2 = _pool_to(x2, 5)
        X.append(self.out_conv2(x2))
        x3 = self.conv3(x2)
        X.append(x3)
        return X



# =============================================================================
# =============================================================================
def make_hsi_encoder() -> nn.Module:
    if train_dataset == "Houston2013":
        return HSI_Encoder_Houston2013()
    if train_dataset == "Houston2018":
        return HSI_Encoder_Houston2018()
    if train_dataset == "Trento":
        return HSI_Encoder_Trento()
    if train_dataset == "Muufl":
        return HSI_Encoder_Muufl()
    raise ValueError(f"Unknown dataset: {train_dataset}")


def make_lidar_encoder() -> nn.Module:
    if train_dataset == "Houston2013":
        return Lidar_Encoder_Houston2013()
    if train_dataset == "Houston2018":
        return Lidar_Encoder_Houston2018()
    if train_dataset == "Trento":
        return Lidar_Encoder_Trento()
    if train_dataset == "Muufl":
        return Lidar_Encoder_Muufl()
    raise ValueError(f"Unknown dataset: {train_dataset}")


if train_dataset == "Houston2013":
    HSI_Encoder, Lidar_Encoder = HSI_Encoder_Houston2013, Lidar_Encoder_Houston2013
elif train_dataset == "Houston2018":
    HSI_Encoder, Lidar_Encoder = HSI_Encoder_Houston2018, Lidar_Encoder_Houston2018
elif train_dataset == "Trento":
    HSI_Encoder, Lidar_Encoder = HSI_Encoder_Trento, Lidar_Encoder_Trento
elif train_dataset == "Muufl":
    HSI_Encoder, Lidar_Encoder = HSI_Encoder_Muufl, Lidar_Encoder_Muufl
else:
    raise ValueError(f"Unsupported train_dataset={train_dataset}")


# =============================================================================
# =============================================================================
class Feature_Interaction_Layer0(nn.Module):
    """
    ( c1/c2/c3 ),.
    :
      aggr_res_lst: List[Tensor], (B, C, P, P)
      all_path_prob: (B, num_out_path, num_cell, P, P) 
    """
    def __init__(self, args, num_cell: int, num_out_path: int):
        super().__init__()
        self.args = args
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.c1 = Cell_1_0(num_out_path)
        self.c2 = Cell_2_0(num_out_path)
        self.c3 = Cell_3_0(num_out_path)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor):
        target = self.args.hsi_windowSize

        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell

        emb_lst[0], path_prob[0] = self.c1(lidar, hsi)
        emb_lst[1], path_prob[1] = self.c2(lidar, hsi)
        emb_lst[2], path_prob[2] = self.c3(lidar, hsi)

        for j in range(self.num_cell):
            emb_lst[j] = _ensure_hw_to_target(emb_lst[j], target)

        all_path_prob = torch.stack(path_prob, dim=2)  # (B, num_out_path, num_cell, H, W) or (B,*,*,H,W)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            res = 0
            for j in range(self.num_cell):
                cur_emb = emb_lst[j]                          # (B,C,P,P)
                cur_path = path_prob[j][:, i]                 # (B,?,P,P) or (B,P,P)
                cur_path = _align_path_to_target(cur_path, cur_emb.shape[1], target)
                res = res + cur_path * cur_emb
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob


class Feature_Interaction_Layern(nn.Module):
    """
    (stage=1/2), ref_rgn().
    """
    def __init__(self, args, num_cell: int, num_out_path: int, stage: int = 1):
        super().__init__()
        self.args = args
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.stage = stage

        self.c11 = Cell_1(num_out_path)
        self.c12 = Cell_2(num_out_path)
        self.c13 = Cell_3(num_out_path)

        self.c21 = Cell_1(num_out_path)
        self.c22 = Cell_2(num_out_path)
        self.c23 = Cell_3(num_out_path)

        self.conv2 = nn.Sequential(
            nn.Conv2d(args.uni_dimension * 2, args.uni_dimension, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(args.uni_dimension)
        )

    def forward(self, ref_rgn: List[torch.Tensor], hsi: torch.Tensor, lidar: torch.Tensor):
        target = self.args.hsi_windowSize

        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell

        if self.stage == 1:
            emb_lst[0], path_prob[0] = self.c11(lidar, hsi, ref_rgn[0])
            emb_lst[1], path_prob[1] = self.c12(lidar, hsi, ref_rgn[1])
            emb_lst[2], path_prob[2] = self.c13(lidar, hsi, ref_rgn[2])
        elif self.stage == 2:
            emb_lst[0], path_prob[0] = self.c21(lidar, hsi, ref_rgn[0])
            emb_lst[1], path_prob[1] = self.c22(lidar, hsi, ref_rgn[1])
            emb_lst[2], path_prob[2] = self.c23(lidar, hsi, ref_rgn[2])
        else:
            raise ValueError(f"Unsupported stage={self.stage}")

        for j in range(self.num_cell):
            emb_lst[j] = _ensure_hw_to_target(emb_lst[j], target)

        all_path_prob = torch.stack(path_prob, dim=2)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            res = 0
            for j in range(self.num_cell):
                cur_emb = emb_lst[j]                          # (B,C,P,P)
                cur_path = path_prob[j][:, i]                 # (B,?,P,P) or (B,P,P)
                cur_path = _align_path_to_target(cur_path, cur_emb.shape[1], target)
                res = res + cur_path * cur_emb
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob
