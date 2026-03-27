import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *


class Unet(nn.Module):
    """
    2D UNet with optional attention, residual connections, PET adaptation.

    This model follows the encoder-decoder design with skip connections. Optionally,
    it integrates PET signals via adapters and supports multi-head outputs for
    tumor and organ segmentation.

    Args:
        in_chans (int): Number of input channels. Default 1.
        out_chans (int): Number of output channels for tumor head. Default 1.
        chans (int): Base number of feature channels. Default 32.
        num_pool_layers (int): Number of down/up sampling layers. Default 4.
        drop_prob (float): Dropout probability. Default 0.2.
        use_att (bool): Whether to include attention blocks. Default False.
        use_res (bool): Whether to use residual connections. Default False.
        leaky_negative_slope (float): Negative slope for LeakyReLU. Default 0.0.

    Example:
        >>> model = Unet(in_chans=2, out_chans=1, multihead=True)
        >>> x = torch.randn(1, 2, 64, 128, 128)
        >>> tumor, organ = model(x)
        >>> tumor.shape, organ.shape
        (torch.Size([1, 1, 64, 128, 128]), torch.Size([1, 8, 64, 128, 128]))
    """
    def __init__(self,
                in_chans:int = 1,
                out_chans:int = 1,
                chans:int = 32,
                num_pool_layers:int = 4,
                drop_prob:float = 0.2,
                use_att:bool = False,
                use_res = False,
                leaky_negative_slope:float = 0.0,
                ):

        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.use_res = use_res
        self.use_att = use_att
        self.leaky_negative_slope = leaky_negative_slope

        self.down_sample_layers = nn.ModuleList(
            [
                ConvBlock2D(
                    in_chans = in_chans,
                    out_chans = chans,
                    drop_prob = drop_prob, 
                    use_res = self.use_res, 
                    leaky_negative_slope = leaky_negative_slope
                    )
            ]
        )


        if use_att:
            self.down_att_layers = nn.ModuleList([AttentionBlock2D(chans)])

        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock2D(ch, ch * 2, drop_prob, self.use_res, leaky_negative_slope))
            if use_att:
                self.down_att_layers.append(AttentionBlock2D(ch * 2))
            ch *= 2
        
        self.conv = ConvBlock2D(ch, ch * 2, drop_prob, self.use_res, leaky_negative_slope)
        if use_att:
            self.conv_att = AttentionBlock2D(ch * 2)

        self.cross_atts=nn.ModuleList()

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()

        if use_att:
            self.up_att = nn.ModuleList()

        for _ in range(num_pool_layers):
            self.up_transpose_conv.append(TransposeConvBlock2D(ch * 2, ch, leaky_negative_slope))
            self.up_conv.append(ConvBlock2D(ch * 2, ch, drop_prob, self.use_res, leaky_negative_slope))
            if use_att:
                self.up_att.append(AttentionBlock2D(ch))
            ch //= 2

        self.out_conv = nn.Conv2d(ch * 2, self.out_chans, kernel_size=1, stride=1)


    def forward(self, image: torch.Tensor):
        """
        Forward pass of the UNet.

        Args:
            image (torch.Tensor): Input tensor (N, in_chans, D, H, W).

        Returns:
            torch.Tensor or tuple(torch.Tensor, torch.Tensor):
                - Tumor segmentation if multihead=False.
                - Tuple (tumor, organ) if multihead=True.
        """
        stack = []
        output = image

        # Downsampling path
        for idx, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            if self.use_att:
                if idx > 0:
                    output = self.down_att_layers[idx](output)
                else:
                    output = self.down_att_layers[idx](output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2)

        # Bottleneck
        output = self.conv(output)
        if self.use_att:
            output = self.conv_att(output)

        # Upsampling path
        for idx in range(self.num_pool_layers):
            skip_connection = stack.pop()
            output = self.up_transpose_conv[idx](output)

            # Handle shape mismatch due to odd input dims
            diff_d = skip_connection.shape[-3] - output.shape[-3]
            diff_h = skip_connection.shape[-2] - output.shape[-2]
            diff_w = skip_connection.shape[-1] - output.shape[-1]
            if diff_d != 0 or diff_h != 0 or diff_w != 0:
                output = F.pad(output, [0, diff_w, 0, diff_h, 0, diff_d], mode='reflect')

            # Concatenate with skip connection
            output = torch.cat([output, skip_connection], dim=1)
            output = self.up_conv[idx](output)

            if self.use_att:
                output = self.up_att[idx](output)

        tumor_out = self.out_conv(output)
        return tumor_out


# ======================
# MODEL SUMMARY
# ======================
from torchinfo import summary
import torch

# Instantiate the model
model = Unet(
    in_chans=1,     
    out_chans=1,    # tumor segmentation
    chans=32,
    num_pool_layers=4,
    drop_prob=0.2,
    use_att=True,
    use_res=True
)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create a dummy input tensor
# Shape: (batch, channels, depth, height, width)
x = torch.randn(1, 1, 128, 128).to(device)

# Print the model summary
summary(
    model,
    input_data=x,
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
)
