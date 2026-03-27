import torch
import torch.nn as nn

class ConvBlock2D(nn.Module):
    """2D Conv block: two conv layers + InstanceNorm + LeakyReLU + Dropout, optional residual."""
    def __init__(self, in_chans, out_chans, drop_prob, use_res=True, leaky_negative_slope=0.0):
        super().__init__()
        self.use_res = use_res
        self.leaky_negative_slope = leaky_negative_slope

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(leaky_negative_slope, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        )

        self.conv1x1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)

        self.out_layers = nn.Sequential(
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(leaky_negative_slope, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, x):
        if self.use_res:
            return self.out_layers(self.layers(x) + self.conv1x1(x))
        else:
            return self.out_layers(self.layers(x))


class TransposeConvBlock2D(nn.Module):
    """2D transpose conv block for upsampling: ConvTranspose2d + InstanceNorm + LeakyReLU."""
    def __init__(self, in_chans, out_chans, leaky_negative_slope=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(leaky_negative_slope, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class AttentionBlock2D(nn.Module):
    """2D attention block: channel + spatial attention."""
    def __init__(self, num_chans, reduction=2):
        super().__init__()
        self.C = num_chans
        self.r = reduction
        self.sig = nn.Sigmoid()

        # Channel attention
        self.fc_ch = nn.Sequential(
            nn.Linear(self.C, self.C // self.r),
            nn.ReLU(inplace=True),
            nn.Linear(self.C // self.r, self.C),
        )
        # Spatial attention
        self.conv = nn.Conv2d(self.C, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        # Spatial attention
        sa = self.sig(self.conv(x))
        x_sa = sa * x

        # Channel attention
        ca = torch.mean(torch.abs(x).reshape(b, c, -1), dim=2)  # (b, c)
        ca = self.sig(self.fc_ch(ca)).reshape(b, c, 1, 1)       # (b, c, 1, 1)
        x_ca = ca * x

        return torch.max(x_sa, x_ca)