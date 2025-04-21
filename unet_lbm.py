import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Timestep Embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# --- Basic Building Blocks ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- U-Net with Timestep (Modified for 32x32) ---
class UNetLBM(nn.Module):
    # Removed one level of down/up sampling
    def __init__(self, n_channels, n_out_channels, bilinear=True, time_emb_dim=32):
        super(UNetLBM, self).__init__()
        if n_channels != 27:
             print(f"Warning: Expected n_channels=27 (9 distributions * 3 RGB), got {n_channels}")
        if n_out_channels != 27:
             print(f"Warning: Expected n_out_channels=27, got {n_out_channels}")

        self.n_channels = n_channels
        self.n_out_channels = n_out_channels
        self.bilinear = bilinear
        self.time_emb_dim = time_emb_dim

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # U-Net structure (3 down, 3 up)
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // factor) # Bottleneck is now 512 channels at 4x4

        # Linear layers to project time embedding
        self.time_proj1 = nn.Linear(time_emb_dim, 128)
        self.time_proj2 = nn.Linear(time_emb_dim, 256)
        self.time_proj3 = nn.Linear(time_emb_dim, 512 // factor) # Project to bottleneck channels

        # Removed up1
        self.up2 = Up(512, 256 // factor, bilinear) # Takes bottleneck (down3 output) and x3
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_out_channels)

    def _add_time_embedding(self, x, t_emb, proj_layer):
        """ Adds time embedding to feature map """
        t_projected = proj_layer(t_emb)
        t_projected = t_projected[:, :, None, None]
        return x + t_projected

    def forward(self, x, t):
        # x shape: [B, C=27, H=32, W=32]
        # t shape: [B]
        t_emb = self.time_mlp(t) # [B, time_emb_dim]

        x1 = self.inc(x) # 32x32
        x2 = self.down1(x1) # 16x16
        x2 = self._add_time_embedding(x2, t_emb, self.time_proj1)
        x3 = self.down2(x2) # 8x8
        x3 = self._add_time_embedding(x3, t_emb, self.time_proj2)
        x4 = self.down3(x3) # 4x4 (Bottleneck)
        x4 = self._add_time_embedding(x4, t_emb, self.time_proj3)

        # Removed up1(x5, x4)
        x = self.up2(x4, x3) # Upsample from 4x4 to 8x8
        x = self.up3(x, x2) # Upsample from 8x8 to 16x16
        x = self.up4(x, x1) # Upsample from 16x16 to 32x32
        logits = self.outc(x) # Output shape: [B, C=27, H=32, W=32]
        return logits