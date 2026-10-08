import torch
import torch.nn as nn

SE_SCALE_MODES = {"sigmoid", "tanh_residual"}


def _check_se_scale_mode(scale_mode):
    if scale_mode not in SE_SCALE_MODES:
        valid_modes = ", ".join(sorted(SE_SCALE_MODES))
        raise ValueError(f"Unsupported SE scale mode: {scale_mode}. Valid: {valid_modes}")


def _make_se_activation(scale_mode):
    _check_se_scale_mode(scale_mode)
    if scale_mode == "sigmoid":
        return nn.Sigmoid()
    return nn.Tanh()


def _init_se_projection(projection, scale_mode):
    _check_se_scale_mode(scale_mode)
    nn.init.zeros_(projection.weight)
    if scale_mode == "sigmoid":
        nn.init.constant_(projection.bias, 5.0)
    else:
        nn.init.zeros_(projection.bias)


def _apply_se_scale(x, scale, scale_mode):
    _check_se_scale_mode(scale_mode)
    if scale_mode == "sigmoid":
        return x * scale
    return x * (1 + scale)


class ZeroConvBlock(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        mid_channels=1024,
        out_channels=1024,
        enable_zero_conv=True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.enable_zero_conv = enable_zero_conv

        if enable_zero_conv:
            nn.init.constant_(self.conv2.weight, 0.0)
            nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, x):
        if self.enable_zero_conv:
            return self.conv2(self.act(self.conv1(x)))
        return self.conv2(self.act(self.conv1(x))) + x


class DownBlock(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=1024, out_channels=1024):
        super().__init__()
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, mid_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class ScaleProjector(nn.Module):
    def __init__(self, depth_feat_channels=128, token_dim=1024, cnn_output_dim=512):
        super().__init__()

        self.depth_cnn_downsampler = nn.Sequential(
            nn.Conv2d(depth_feat_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        self.token_aggregator = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.LayerNorm(token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim // 2),
        )

        self.combined_projector = nn.Sequential(
            nn.Linear(cnn_output_dim + token_dim // 2, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, depth_feat, enc_tokens):
        """
        Forward pass for the fusion and scale prediction.

        Args:
            depth_feat (torch.Tensor): The depth feature map.
                                       Assumed Shape: (Bs, nimgs, Channels, H, W)
            enc_tokens (torch.Tensor): The tokens from a transformer.
                                       Shape: (Bs * nimgs, num_tokens, token_dim)

        Returns:
            torch.Tensor: The predicted scale factor. Shape: (Bs, nimgs, 1)
        """
        Bs, nimgs, C, H, W = depth_feat.shape
        N = enc_tokens.shape[0]
        depth_feat_reshaped = depth_feat.view(N, C, H, W)

        # Downsample depth feature using the CNN.
        # Shape: (N, C, H, W) -> (N, 512, 1, 1)
        downsampled_depth = self.depth_cnn_downsampler(depth_feat_reshaped)

        # Flatten the CNN output to get a feature vector.
        # Shape: (N, 512, 1, 1) -> (N, 512)
        depth_vec = downsampled_depth.flatten(start_dim=1)
        token_vec = self.token_aggregator(enc_tokens).mean(dim=1)

        # Shape: (N, 512 + token_dim) -> (N, 1)
        scale = self.combined_projector(torch.cat([depth_vec, token_vec], dim=1))
        final_scale = scale.view(Bs, nimgs, 1)

        return final_scale


class ChannelSE(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention recalibration.

    Applies global average pooling across the token dimension (Squeeze),
    learns channel interdependencies via two FC layers (Excitation),
    and rescales the original features (Scale).

    The default sigmoid mode uses output = x * sigmoid(excitation(x)).
    tanh_residual remains available for explicitly requested experiments.

    Args:
        channels: Number of input channels (default: 1024 for VGGT)
        reduction: Reduction ratio for the bottleneck (default: 16)
        scale_mode: SE scaling formulation, either "tanh_residual" or "sigmoid"
    """

    def __init__(self, channels=1024, reduction=16, scale_mode="sigmoid"):
        super().__init__()
        _check_se_scale_mode(scale_mode)
        self.scale_mode = scale_mode
        mid_channels = channels // reduction
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, channels),
            _make_se_activation(scale_mode),
        )

        # Keep initialization compatible with the selected scale formulation.
        _init_se_projection(self.excitation[2], scale_mode)

    def forward(self, x):
        # x: (B*T, P, C)
        sq = self.squeeze(x.transpose(1, 2)).squeeze(-1)  # (B*T, C)
        scale = self.excitation(sq).unsqueeze(1)  # (B*T, 1, C)
        return _apply_se_scale(x, scale, self.scale_mode)


class EncoderConditionedSE(nn.Module):
    """SE block conditioned on encoder intermediate features.

    Excitation receives concat of decoder squeeze and encoder squeeze,
    enabling multi-scale encoder information to guide channel calibration.
    Uses the configured SE scale formulation.

    Args:
        channels: Number of input channels (default: 1024 for VGGT)
        reduction: Reduction ratio for the bottleneck (default: 16)
        scale_mode: SE scaling formulation, either "tanh_residual" or "sigmoid"
    """

    def __init__(self, channels=1024, reduction=16, scale_mode="sigmoid"):
        super().__init__()
        _check_se_scale_mode(scale_mode)
        self.scale_mode = scale_mode
        mid_channels = channels // reduction
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels * 2, mid_channels),  # [dec_sq, enc_sq] -> bottleneck
            nn.ReLU(),
            nn.Linear(mid_channels, channels),  # bottleneck -> adjustment
            _make_se_activation(scale_mode),
        )

        # Keep initialization compatible with the selected scale formulation.
        _init_se_projection(self.excitation[2], scale_mode)

    def forward(self, x, enc_feat):
        # x: (B*T, P, C), enc_feat: (B*T, P, C)
        dec_sq = self.squeeze(x.transpose(1, 2)).squeeze(-1)  # (B*T, C)
        enc_sq = self.squeeze(enc_feat.transpose(1, 2)).squeeze(-1)  # (B*T, C)
        combined = torch.cat([dec_sq, enc_sq], dim=-1)  # (B*T, 2C)
        scale = self.excitation(combined).unsqueeze(1)  # (B*T, 1, C)
        return _apply_se_scale(x, scale, self.scale_mode)


class GlobalEncoderConditionedSE(nn.Module):
    """Global SE block conditioned on all encoder intermediate layers.

    Projects 4 encoder layers' squeeze signals into a single vector,
    then concatenates with decoder squeeze for excitation.
    Uses the configured SE scale formulation.

    Args:
        channels: Number of input channels (default: 1024 for VGGT)
        num_enc_layers: Number of encoder layers to fuse (default: 4)
        reduction: Reduction ratio for the bottleneck (default: 16)
        scale_mode: SE scaling formulation, either "tanh_residual" or "sigmoid"
    """

    def __init__(
        self,
        channels=1024,
        num_enc_layers=4,
        reduction=16,
        scale_mode="sigmoid",
    ):
        super().__init__()
        _check_se_scale_mode(scale_mode)
        self.scale_mode = scale_mode
        mid_channels = channels // reduction
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.enc_proj = nn.Linear(channels * num_enc_layers, channels)
        self.excitation = nn.Sequential(
            nn.Linear(channels * 2, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, channels),
            _make_se_activation(scale_mode),
        )

        # Keep initialization compatible with the selected scale formulation.
        _init_se_projection(self.excitation[2], scale_mode)

    def forward(self, x, enc_feats_list):
        # x: (B*T, P, C), enc_feats_list: list of 4 x (B*T, P, C)
        dec_sq = self.squeeze(x.transpose(1, 2)).squeeze(-1)  # (B*T, C)
        enc_sqs = [
            self.squeeze(f.transpose(1, 2)).squeeze(-1) for f in enc_feats_list
        ]  # list of (B*T, C)
        enc_combined = self.enc_proj(torch.cat(enc_sqs, dim=-1))  # (B*T, C)
        combined = torch.cat([dec_sq, enc_combined], dim=-1)  # (B*T, 2C)
        scale = self.excitation(combined).unsqueeze(1)  # (B*T, 1, C)
        return _apply_se_scale(x, scale, self.scale_mode)


class SpatialChannelSE2D(nn.Module):
    """Squeeze-and-Excitation block for 2D feature maps (spatial domain).

    Designed for use inside DPT heads where features are already reshaped
    to (B, C, H, W) format. Applies global average pooling over spatial
    dimensions (Squeeze), learns channel interdependencies via two FC
    layers (Excitation), and rescales the original features (Scale).

    Uses the configured SE scale formulation, consistent with the ChannelSE design.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for the bottleneck (default: 16)
        scale_mode: SE scaling formulation, either "tanh_residual" or "sigmoid"
    """

    def __init__(self, channels, reduction=16, scale_mode="sigmoid"):
        super().__init__()
        _check_se_scale_mode(scale_mode)
        self.scale_mode = scale_mode
        mid_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, channels),
            _make_se_activation(scale_mode),
        )

        # Keep initialization compatible with the selected scale formulation.
        _init_se_projection(self.fc[2], scale_mode)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)          # (B, C) - squeeze over spatial
        scale = self.fc(s).view(b, c, 1, 1)  # (B, C, 1, 1) - excitation
        return _apply_se_scale(x, scale, self.scale_mode)
