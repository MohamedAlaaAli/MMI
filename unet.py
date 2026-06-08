import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *

class BaseFusionUNet(nn.Module):
    """
    Base class for fusion-based UNets.
    
    Provides common functionality for encoder/decoder building,
    fusion operations, and multi-stream encoding.
    """

    def __init__(self):
        super().__init__()
        self.fusion_mode = None
        self.use_att = None
        self.use_res = None
        self.num_pool_layers = None

    def _build_encoder(self, in_chans, enc_chans, drop_prob, leaky_slope):
        """Return (ModuleList of ConvBlocks, ModuleList of AttBlocks or None)."""
        layers = nn.ModuleList()
        att_layers = nn.ModuleList() if self.use_att else None

        in_c = in_chans
        for out_c in enc_chans:
            layers.append(
                ConvBlock2D(in_c, out_c, drop_prob, self.use_res, leaky_slope)
            )
            if self.use_att:
                att_layers.append(AttentionBlock2D(out_c))
            in_c = out_c

        return layers, att_layers

    def _build_fuser(self, single_stream_chans: int, mode: str) -> nn.Module:
        """
        Returns a fusion module that takes two tensors of shape (B, C, H, W)
        and produces one tensor of shape (B, C, H, W).
        """
        C = single_stream_chans

        if mode == "concat":
            return nn.Sequential(
                nn.Conv2d(C * 2, C, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, kernel_size=1),
            )

        elif mode in ("add", "avg"):
            return nn.Identity()

        elif mode == "attention":
            # Squeeze-and-Excitation gating
            return _SEFuser(C)

        else:
            raise ValueError(
                f"Unknown fusion_mode '{mode}'. "
                "Choose from ['concat', 'add', 'avg', 'attention']."
            )

    def _fuse(self, fuser: nn.Module, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Apply fusion operator to two same-shape tensors."""
        if self.fusion_mode == "concat":
            return fuser(torch.cat([a, b], dim=1))
        elif self.fusion_mode == "add":
            return a + b
        elif self.fusion_mode == "avg":
            return (a + b) * 0.5
        elif self.fusion_mode == "attention":
            return fuser(a, b)
        raise ValueError(self.fusion_mode)

    def _encode_path(self, x, down_layers, att_layers):
        """Run one encoder path; returns (final_x, skip_stack)."""
        skips = []
        for idx, layer in enumerate(down_layers):
            x = layer(x)
            if self.use_att and att_layers is not None:
                x = att_layers[idx](x)
            skips.append(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x, skips

    def _build_decoder(self, bottleneck_chans, enc_chans, drop_prob, leaky_negative_slope):
        """Build decoder layers (transpose conv + conv)."""
        up_transpose_conv = nn.ModuleList()
        up_conv = nn.ModuleList()
        up_att = nn.ModuleList() if self.use_att else None

        ch = bottleneck_chans
        for c in reversed(enc_chans):
            up_transpose_conv.append(
                TransposeConvBlock2D(ch, c, leaky_negative_slope)
            )
            up_conv.append(
                ConvBlock2D(c * 2, c, drop_prob, self.use_res, leaky_negative_slope)
            )
            if self.use_att:
                up_att.append(AttentionBlock2D(c))
            ch = c

        return up_transpose_conv, up_conv, up_att


class IntermediateFusionUNet(BaseFusionUNet):
    """
    Intermediate fusion UNet for CT + PET inputs.

    Architecture:
        - Two independent encoders (CT path, PET path)
        - Fusion at the bottleneck (after the last encoder stage)
        - Single shared decoder with fused skip connections (optional)
        - Single output head

    Fusion strategies:
        - "concat"   : cat(ct, pet) along channel dim, reduce with conv
        - "add"      : element-wise sum
        - "avg"      : element-wise average
        - "attention": learned channel-wise gating (SE-style)

    Args:
        in_chans_ct  (int)  : Input channels for CT.  Default 1.
        in_chans_pet (int)  : Input channels for PET. Default 1.
        out_chans    (int)  : Output channels.        Default 1.
        chans        (int)  : Base feature channels.  Default 32.
        num_pool_layers (int): Encoder/decoder depth. Default 4.
        drop_prob    (float): Dropout probability.    Default 0.2.
        use_att      (bool) : Self-attention in conv blocks. Default False.
        use_res      (bool) : Residual connections.   Default False.
        leaky_negative_slope (float): LeakyReLU slope. Default 0.0.
        fusion_mode  (str)  : One of concat/add/avg/attention. Default "concat".
        fuse_skips   (bool) : Also fuse encoder skips before decoder. Default True.
    """

    def __init__(
        self,
        in_chans_ct: int = 1,
        in_chans_pet: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.2,
        use_att: bool = False,
        use_res: bool = False,
        leaky_negative_slope: float = 0.0,
        fusion_mode: str = "concat",
        fuse_skips: bool = True,
    ):
        super().__init__()

        self.num_pool_layers = num_pool_layers
        self.fusion_mode = fusion_mode.lower()
        self.fuse_skips = fuse_skips
        self.use_att = use_att
        self.use_res = use_res

        enc_chans = [chans * (2 ** i) for i in range(num_pool_layers)]
        bottleneck_chans = enc_chans[-1] * 2 

        # CT encoder
        self.ct_down, self.ct_att = self._build_encoder(
            in_chans_ct, enc_chans, drop_prob, leaky_negative_slope
        )
        self.ct_bottleneck = ConvBlock2D(
            enc_chans[-1], bottleneck_chans, drop_prob, use_res, leaky_negative_slope
        )

        # PET encoder
        self.pet_down, self.pet_att = self._build_encoder(
            in_chans_pet, enc_chans, drop_prob, leaky_negative_slope
        )
        self.pet_bottleneck = ConvBlock2D(
            enc_chans[-1], bottleneck_chans, drop_prob, use_res, leaky_negative_slope
        )

        # bottleneck fusion
        self.bottleneck_fuser = self._build_fuser(bottleneck_chans, fusion_mode)

        # per-skip fusion (one per encoder level, from deepest to shallowest)
        if fuse_skips:
            self.skip_fusers = nn.ModuleList([
                self._build_fuser(c, fusion_mode) for c in reversed(enc_chans)
            ])

        # Decoder
        self.up_transpose_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        if use_att:
            self.up_att = nn.ModuleList()

        ch = bottleneck_chans
        for c in reversed(enc_chans):
            self.up_transpose_conv.append(
                TransposeConvBlock2D(ch, c, leaky_negative_slope)
            )
            self.up_conv.append(
                ConvBlock2D(c * 2, c, drop_prob, use_res, leaky_negative_slope)
            )
            if use_att:
                self.up_att.append(AttentionBlock2D(c))
            ch = c

        self.out_conv = nn.Conv2d(ch, out_chans, kernel_size=1)


    def forward(self, ct: torch.Tensor, pet: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            ct  : (B, in_chans_ct,  H, W)
            pet : (B, in_chans_pet, H, W)  — if None, ct is reused (ablation).

        Returns:
            (B, out_chans, H, W)
        """
        if pet is None:
            pet = ct

        ct_pre_bn,  ct_skips  = self._encode_path(ct,  self.ct_down,  self.ct_att)
        pet_pre_bn, pet_skips = self._encode_path(pet, self.pet_down, self.pet_att)

        ct_bn  = self.ct_bottleneck(ct_pre_bn)
        pet_bn = self.pet_bottleneck(pet_pre_bn)

        x = self._fuse(self.bottleneck_fuser, ct_bn, pet_bn)

        if self.fuse_skips:
            fused_skips = [
                self._fuse(fuser, ct_s, pet_s)
                for fuser, ct_s, pet_s
                in zip(self.skip_fusers, reversed(ct_skips), reversed(pet_skips))
            ]
        else:
            fused_skips = list(reversed(ct_skips))

        # Decode
        for idx in range(self.num_pool_layers):
            skip = fused_skips[idx]
            x = self.up_transpose_conv[idx](x)

            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            if dh or dw:
                x = F.pad(x, [0, dw, 0, dh], mode="reflect")

            x = torch.cat([x, skip], dim=1)
            x = self.up_conv[idx](x)
            if self.use_att:
                x = self.up_att[idx](x)

        return self.out_conv(x)



class _SEFuser(nn.Module):
    """
    Squeeze-and-Excitation fusion gate.

    Concatenates two streams, globally pools, projects to per-channel
    weights, then applies them back to a weighted sum of the two streams.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           
            nn.Flatten(),                      
            nn.Linear(channels * 2, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels * 2),
            nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([a, b], dim=1)
        weights  = self.gate(combined).unsqueeze(-1).unsqueeze(-1)
        wa, wb   = weights.chunk(2, dim=1)
        return wa * a + wb * b

class EarlyIntermediateFusionUNet(BaseFusionUNet):
    """
    Early Intermediate Fusion UNet for CT + PET inputs.

    Architecture:
        - Two independent encoders (CT path, PET path) for first N-K layers
        - Fusion at layer N-K (K layers before bottleneck)
        - Single shared encoder for remaining K layers
        - Shared decoder with fused skip connections
        - Single output head

    Fusion strategies:
        - "concat"   : cat(ct, pet) along channel dim, reduce with conv
        - "add"      : element-wise sum
        - "avg"      : element-wise average
        - "attention": learned channel-wise gating (SE-style)

    Args:
        in_chans_ct  (int)  : Input channels for CT.  Default 1.
        in_chans_pet (int)  : Input channels for PET. Default 1.
        out_chans    (int)  : Output channels.        Default 1.
        chans        (int)  : Base feature channels.  Default 32.
        num_pool_layers (int): Encoder/decoder depth. Default 4.
        fusion_depth (int)  : Number of layers before bottleneck to fuse. Default 2.
        drop_prob    (float): Dropout probability.    Default 0.2.
        use_att      (bool) : Self-attention in conv blocks. Default False.
        use_res      (bool) : Residual connections.   Default False.
        leaky_negative_slope (float): LeakyReLU slope. Default 0.0.
        fusion_mode  (str)  : One of concat/add/avg/attention. Default "concat".
        fuse_skips   (bool) : Also fuse encoder skips before decoder. Default True.
    """

    def __init__(
        self,
        in_chans_ct: int = 1,
        in_chans_pet: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_pool_layers: int = 4,
        fusion_depth: int = 2,
        drop_prob: float = 0.2,
        use_att: bool = False,
        use_res: bool = False,
        leaky_negative_slope: float = 0.0,
        fusion_mode: str = "concat",
        fuse_skips: bool = True,
    ):
        super().__init__()

        self.num_pool_layers = num_pool_layers
        self.fusion_depth = min(fusion_depth, num_pool_layers - 1)  # Clamp to valid range
        self.independent_depth = num_pool_layers - self.fusion_depth
        self.fusion_mode = fusion_mode.lower()
        self.fuse_skips = fuse_skips
        self.use_att = use_att
        self.use_res = use_res

        enc_chans = [chans * (2 ** i) for i in range(num_pool_layers)]
        fusion_chans = enc_chans[self.independent_depth - 1] if self.independent_depth > 0 else chans
        bottleneck_chans = enc_chans[-1] * 2

        # CT/PET encoders
        if self.independent_depth > 0:
            self.ct_down_early, self.ct_att_early = self._build_encoder(
                in_chans_ct, enc_chans[:self.independent_depth], drop_prob, leaky_negative_slope
            )
            self.pet_down_early, self.pet_att_early = self._build_encoder(
                in_chans_pet, enc_chans[:self.independent_depth], drop_prob, leaky_negative_slope
            )
        else:
            self.ct_down_early = None
            self.ct_att_early = None
            self.pet_down_early = None
            self.pet_att_early = None

        self.early_fuser = self._build_fuser(fusion_chans, fusion_mode)

        if self.fusion_depth > 0:
            shared_enc_chans = enc_chans[self.independent_depth:num_pool_layers]
            self.shared_down, self.shared_att = self._build_encoder(
                fusion_chans, shared_enc_chans, drop_prob, leaky_negative_slope
            )
        else:
            self.shared_down = None
            self.shared_att = None

        self.bottleneck = ConvBlock2D(
            enc_chans[-1], bottleneck_chans, drop_prob, use_res, leaky_negative_slope
        )

        if fuse_skips:
            self.skip_fusers = nn.ModuleList([
                self._build_fuser(c, fusion_mode) for c in reversed(enc_chans)
            ])

        # Decoder
        self.up_transpose_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        if use_att:
            self.up_att = nn.ModuleList()

        ch = bottleneck_chans
        for c in reversed(enc_chans):
            self.up_transpose_conv.append(
                TransposeConvBlock2D(ch, c, leaky_negative_slope)
            )
            self.up_conv.append(
                ConvBlock2D(c * 2, c, drop_prob, use_res, leaky_negative_slope)
            )
            if use_att:
                self.up_att.append(AttentionBlock2D(c))
            ch = c

        self.out_conv = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, ct: torch.Tensor, pet: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            ct  : (B, in_chans_ct,  H, W)
            pet : (B, in_chans_pet, H, W) — if None, ct is reused.

        Returns:
            (B, out_chans, H, W)
        """
        if pet is None:
            pet = ct

        all_skips_ct = []
        all_skips_pet = []

        if self.independent_depth > 0:
            ct_early, ct_early_skips = self._encode_path(
                ct, self.ct_down_early, self.ct_att_early
            )
            pet_early, pet_early_skips = self._encode_path(
                pet, self.pet_down_early, self.pet_att_early
            )
            all_skips_ct.extend(ct_early_skips)
            all_skips_pet.extend(pet_early_skips)
        else:
            ct_early, pet_early = ct, pet

        fused = self._fuse(self.early_fuser, ct_early, pet_early)

        if self.fusion_depth > 0:
            shared, shared_skips = self._encode_path(
                fused, self.shared_down, self.shared_att
            )
            all_skips_ct.extend(shared_skips)
            all_skips_pet.extend(shared_skips)
        else:
            shared = fused

        x = self.bottleneck(shared)

        if self.fuse_skips and len(all_skips_ct) > 0:
            fused_skips = [
                self._fuse(fuser, ct_s, pet_s)
                for fuser, ct_s, pet_s
                in zip(self.skip_fusers, reversed(all_skips_ct), reversed(all_skips_pet))
            ]
        else:
            fused_skips = list(reversed(all_skips_ct))

        # Decoder
        for idx in range(self.num_pool_layers):
            skip = fused_skips[idx]
            x = self.up_transpose_conv[idx](x)

            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            if dh or dw:
                x = F.pad(x, [0, dw, 0, dh], mode="reflect")

            x = torch.cat([x, skip], dim=1)
            x = self.up_conv[idx](x)
            if self.use_att:
                x = self.up_att[idx](x)

        return self.out_conv(x)


class LateFusionUNet(nn.Module):
    """
    Late fusion wrapper for two UNet-style models.

    Assumptions:
    - Both models return feature maps/logits of shape:
        [B, C, H, W]
    - Outputs have same spatial dimensions.
    - You want to fuse AFTER both networks produce outputs.

    Fusion modes:
    - concat  -> concatenate channels then fuse with conv
    - add     -> element
    -wise addition
    - avg     -> average outputs
    """

    def __init__(
        self,
        unet1: nn.Module,
        unet2: nn.Module,
        out_channels: int,
        fusion_mode: str = "concat",
    ):
        super().__init__()

        self.unet1 = unet1
        self.unet2 = unet2
        self.fusion_mode = fusion_mode.lower()

        if self.fusion_mode == "concat":
            self.fusion = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
            )

        elif self.fusion_mode in ["add", "avg"]:
            self.fusion = nn.Identity()

        else:
            raise ValueError(
                f"Unsupported fusion mode: {fusion_mode}. "
                f"Choose from ['concat', 'add', 'avg']"
            )

    def forward(self, x1, x2=None):
        """
        x1 : input for first UNet
        x2 : input for second UNet
             if None -> uses x1
        """

        if x2 is None:
            x2 = x1

        out1 = self.unet1(x1)
        out2 = self.unet2(x2)

        if self.fusion_mode == "concat":
            fused = torch.cat([out1, out2], dim=1)
            fused = self.fusion(fused)

        elif self.fusion_mode == "add":
            fused = out1 + out2

        elif self.fusion_mode == "avg":
            fused = (out1 + out2) / 2

        return fused

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