import os
import sys
import warnings

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hashlib
import json
import torch
from torch import nn

# --- FIX: Import constants from the new file ---
from discrete_mbrl.constants import DISCRETE_ENCODER_TYPES, CONTINUOUS_ENCODER_TYPES
# -----------------------------------------------

from training_helpers import log_param_updates
from shared.models import *
from shared.trainers import *
from shared.models.iris_models import \
    Encoder as IrisEncoder, Decoder as IrisDecoder, EncoderDecoderConfig

# --- FIX: Removed constant definitions from here ---
# DISCRETE_ENCODER_TYPES = set(['vqvae', 'dae', 'softmax_ae', 'hard_fta_ae'])
# CONTINUOUS_ENCODER_TYPES = set(['ae', 'vae', 'soft_vqvae', 'fta_ae'])
# ----------------------------------------------------

MODEL_VARS = [
    'embedding_dim', 'latent_dim', 'filter_size', 'codebook_size',
    'ae_model_type', 'ae_model_version', 'trans_model_type', 'trans_model_version',
    'trans_hidden', 'trans_depth', 'stochastic', 'extra_info',
    'repr_sparsity', 'sparsity_type', 'vq_trans_1d_conv',
    'trans_reward_overestimate_coef', 'trans_reward_zero_target_coef',
    'trans_reward_zero_margin']
AE_MODEL_VARS = [
    'embedding_dim', 'latent_dim', 'filter_size', 'codebook_size',
    'ae_model_type', 'ae_model_version', 'extra_info', 'repr_sparsity',
    'sparsity_type']

# ... (the rest of the file remains the same) ...


def make_ae_v1(input_dim, embedding_dim, filter_shape=(8, 8)):
    n_channels = input_dim[0]

    if input_dim[1] == 84:
        filters = (8, 5)
        strides = (3, 2)
        extra_padding = 1
    elif input_dim[1] == 56:
        filters = (6, 3)
        strides = (2, 2)
        extra_padding = 1
    else:
        filters = (8, 5)
        strides = (3, 2)
        extra_padding = 1

    encoder_p1 = nn.Sequential(
        nn.Conv2d(n_channels, 16, filters[0], strides[0], extra_padding),
        nn.ReLU(),
        nn.Conv2d(16, 32, filters[1], strides[1]),
        nn.ReLU(),
        nn.Conv2d(32, embedding_dim, 3, 1),
        nn.ReLU())
    test_input = torch.ones([1] + list(input_dim), dtype=torch.float32)
    out_shape = encoder_p1(test_input).shape[1:]
    mid_filter_shape = out_shape[1:]
    print('AE mid filter shape:', mid_filter_shape)

    encoder = nn.Sequential(
        encoder_p1,
        nn.AdaptiveAvgPool2d(filter_shape),
        ResidualBlock(embedding_dim, embedding_dim),
        ResidualBlock(embedding_dim, embedding_dim))
    decoder_p1 = nn.Sequential(
        nn.ReLU(),
        ResidualBlock(embedding_dim, embedding_dim),
        ResidualBlock(embedding_dim, embedding_dim),
        nn.AdaptiveAvgPool2d(mid_filter_shape),
        nn.ConvTranspose2d(embedding_dim, 32, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, filters[1], strides[1]),
        nn.ReLU(),
        nn.ConvTranspose2d(16, n_channels, filters[0], strides[0], extra_padding))

    out_shape = decoder_p1(encoder(test_input)).shape[1:]

    extra_layers = []
    if list(out_shape) != list(input_dim):
        extra_layers.append(nn.AdaptiveAvgPool2d(input_dim[1:]))
    extra_layers.append(nn.ReLU())
    extra_layers.append(ResidualBlock(n_channels, n_channels))
    extra_layers.append(nn.Conv2d(n_channels, n_channels, 1, 1))
    decoder = nn.Sequential(decoder_p1, *extra_layers)

    return encoder, decoder


def make_dense_ae_v2(input_dim, latent_dim=None, hidden_sizes=[512, 512, 256, 256]):
    if latent_dim is None:
        latent_dim = hidden_sizes[-1]
        hidden_sizes = hidden_sizes[:-1]
    n_features = np.array(input_dim).squeeze()
    encoder = nn.Sequential(*create_dense_layers(
        n_features, out_features=latent_dim,
        hidden_sizes=hidden_sizes))
    decoder = nn.Sequential(*create_dense_layers(
        latent_dim, out_features=n_features,
        hidden_sizes=hidden_sizes[::-1]))
    return encoder, decoder


def make_ae_v2(input_dim, embedding_dim=None, filter_size=None):
    embedding_dim = embedding_dim or 128

    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)

    channels = (input_dim[0], 64, 128, embedding_dim)
    filters = (8, 6, 4)
    strides = (2, 2, 2)
    padding = (1, 0, 0)

    # 🚀 FIX: Store original input size for decoder
    original_height, original_width = input_dim[1], input_dim[2]

    encoder_layers = []
    decoder_layers = []

    for i in range(len(filters)):
        encoder_layers.append(nn.Conv2d(
            channels[i], channels[i + 1], filters[i], strides[i], padding[i]))
        encoder_layers.append(nn.ReLU())

    encoder_p1 = nn.Sequential(*encoder_layers)
    test_input = torch.ones([1] + list(input_dim), dtype=torch.float32)
    out_shape = encoder_p1(test_input).shape[1:]
    mid_filter_shape = out_shape[1:]

    if filter_size:
        encoder_layers.append(nn.AdaptiveAvgPool2d(filter_size))
        decoder_layers.append(ResidualBlock(embedding_dim, embedding_dim))
        decoder_layers.append(nn.AdaptiveAvgPool2d(mid_filter_shape))

    for i in reversed(range(len(filters))):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], filters[i], strides[i], padding[i]))
        decoder_layers.append(nn.ReLU())

    # 🚀 FIX: Add adaptive pooling to ensure exact output dimensions
    decoder_layers.append(nn.AdaptiveAvgPool2d((original_height, original_width)))

    encoder = nn.Sequential(*encoder_layers)
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


def make_ae_v3(input_dim, embedding_dim=None, filter_size=None):
    """ Based off of https://arxiv.org/pdf/2209.00588.pdf """
    # https://github.com/eloialonso/iris/blob/main/src/models/tokenizer/nets.py

    # 5 layers of 2 resnet blocks each and self attention
    # All use 64 filters
    # All but last layer downsample by 2 with a conv layer w/ asymetric padding
    # Finally, there is a res , attn, res, normalize, conv (same shape), sequence
    res = input_dim[1]
    config = EncoderDecoderConfig(
        resolution=res,
        in_channels=input_dim[0],
        z_channels=embedding_dim,  # Output embedding dim
        ch=64,  # Channels
        ch_mult=[1, 1, 1, 1, 1],  # Channel size multiplier
        num_res_blocks=2,
        # Resolutions at which attention is applied
        attn_resolutions=[res / (2 ** 2), res / (2 ** 3)],
        out_ch=input_dim[0],
        dropout=0.0
    )

    encoder = IrisEncoder(config)
    decoder = IrisDecoder(config)

    # TODO: Add adaptive pooling to encoder and decoder after testing original model

    return encoder, decoder


def make_nature_ae(input_dim, embedding_dim=None, filter_size=None, vanilla=False):
    embedding_dim = embedding_dim or 64

    if len(input_dim) <= 1:
        raise ValueError('Input dim must be at least 2D for Nature AE')

    channels = (input_dim[0], 32, 64, embedding_dim)
    filters = (8, 4, 3)
    strides = (4, 2, 1)

    if not vanilla and input_dim[1] == 64:
        padding = (2, 0, 0)
    else:
        padding = (0, 0, 0)

    encoder_layers = []
    decoder_layers = []

    for i in range(len(filters)):
        encoder_layers.append(nn.Conv2d(
            channels[i], channels[i + 1], filters[i], strides[i], padding[i]))
        encoder_layers.append(nn.ReLU())
    encoder_p1 = nn.Sequential(*encoder_layers)

    test_input = torch.ones([1] + list(input_dim), dtype=torch.float32)
    out_shape = encoder_p1(test_input).shape[1:]
    mid_filter_shape = out_shape[1:]
    print('AE mid filter shape:', mid_filter_shape)

    if filter_size and not vanilla:
        encoder_layers.append(nn.AdaptiveAvgPool2d(filter_size))
        decoder_layers.append(nn.AdaptiveAvgPool2d(mid_filter_shape))

    for i in reversed(range(len(filters))):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], filters[i], strides[i], padding[i]))
        if i > 0:
            decoder_layers.append(nn.ReLU())

    encoder = nn.Sequential(*encoder_layers)
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


def make_ae_v4(input_dim, embedding_dim=None, filter_size=None):
    """ Based off of https://arxiv.org/pdf/2209.00588.pdf """
    # https://github.com/eloialonso/iris/blob/main/src/models/tokenizer/nets.py

    # 5 layers of 2 resnet blocks each and self attention
    # All use 64 filters
    # All but last layer downsample by 2 with a conv layer w/ asymetric padding
    # Finally, there is a res , attn, res, normalize, conv (same shape), sequence
    res = input_dim[1]
    config = EncoderDecoderConfig(
        resolution=res,
        in_channels=input_dim[0],
        z_channels=embedding_dim,  # Output embedding dim
        ch=64,  # Channels
        ch_mult=[1, 1, 1, 1, 1],  # Channel size multiplier
        downsamples=[True, True, True, False, False],  # Whether to downsample
        num_res_blocks=2,
        # Resolutions at which attention is applied
        attn_resolutions=[res / (2 ** 2), res / (2 ** 3)],
        out_ch=input_dim[0],
        dropout=0.0
    )

    encoder = IrisEncoder(config)
    decoder = IrisDecoder(config)

    # TODO: Add adaptive pooling to encoder and decoder after testing original model

    return encoder, decoder


def make_ae_v5(input_dim, embedding_dim=None, filter_size=None):
    """Strided convolution encoder — no AdaptiveAvgPool2d.

    Uses uniform k=4, s=2, p=1 convolutions for clean 2× downsampling per layer.
    The number of layers is determined automatically so that
    output_spatial = input_spatial / 2^n_layers == filter_size.

    For DoorKey-8x8: (3,64,64) → 3 layers → (embedding_dim, 8, 8) = 64 tokens,
    each mapping to exactly one 8×8-pixel tile (= one grid cell).
    """
    import math
    embedding_dim = embedding_dim or 64

    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)

    H, W = input_dim[1], input_dim[2]
    filter_size = filter_size or 8

    # Compute number of stride-2 layers needed: H / 2^n = filter_size
    n_layers = int(round(math.log2(H / filter_size)))
    assert n_layers >= 1, f"filter_size {filter_size} >= input size {H}, need at least 1 layer"
    expected = H // (2 ** n_layers)
    assert expected == filter_size, (
        f"Input {H} / 2^{n_layers} = {expected}, but filter_size={filter_size}. "
        f"Input must be filter_size × a power of 2.")

    # Build channel progression: gradually widen then project to embedding_dim
    if n_layers == 1:
        channels = [input_dim[0], embedding_dim]
    elif n_layers == 2:
        channels = [input_dim[0], 64, embedding_dim]
    elif n_layers == 3:
        channels = [input_dim[0], 64, 128, embedding_dim]
    elif n_layers == 4:
        channels = [input_dim[0], 32, 64, 128, embedding_dim]
    else:
        # Fallback: linear interpolation
        channels = [input_dim[0]] + [min(64 * (2 ** i), 256) for i in range(n_layers - 1)] + [embedding_dim]

    # Encoder: n stride-2 conv layers, each halving spatial size
    encoder_layers = []
    for i in range(n_layers):
        encoder_layers.append(nn.Conv2d(
            channels[i], channels[i + 1], kernel_size=4, stride=2, padding=1))
        encoder_layers.append(nn.ReLU())

    # Decoder: mirror with ConvTranspose2d, final adaptive pool for exact dims
    decoder_layers = []
    for i in reversed(range(n_layers)):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.ReLU())
    decoder_layers.append(nn.AdaptiveAvgPool2d((H, W)))

    encoder = nn.Sequential(*encoder_layers)
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


class PatchContextEncoderV9(nn.Module):
    """Patch-based encoder with local context refinement (ViT-style).

    Fundamentally different from v5–v8 (all strided-conv based).  Instead of
    overlapping convolutions that mix information across spatial positions,
    this encoder processes each non-overlapping pixel patch *independently*
    through a small MLP, then adds cross-token context via a lightweight
    residual conv block.

    Why this matters for semantic grounding:
      * Strided-conv encoders (v5) have receptive fields spanning multiple
        tiles.  A goal token's embedding is contaminated by its wall
        neighbours, making them hard to separate in VQ space.
      * Patch embedding guarantees each token sees ONLY its own tile's raw
        pixels.  Goal (solid green) and wall (solid grey) produce maximally
        different MLP outputs because their pixel distributions are disjoint.
      * Cross-token context is added AFTER independent embedding, as a
        residual, so it can only enrich — never overwrite — the per-tile
        identity signal.

    Architecture (64 × 64 → 8 × 8, embedding_dim = 64):

        Phase 1 — Independent patch embedding:
          Input (3, 64, 64)
            → Conv2d(3, 256, k=8, s=8)    [non-overlapping, = per-patch linear]
            → GroupNorm(32) → GELU
            → Conv2d(256, 64, k=1)         [project to embedding dim]
            → GELU
            → (64, 8, 8)   each token = one tile, processed independently

        Phase 2 — Local context refinement (residual):
          → Conv2d(64, 64, k=3, pad=1) → GroupNorm(16) → GELU
          → Conv2d(64, 64, k=3, pad=1) → GroupNorm(16)
          → add residual from Phase 1
          → GELU
          → (64, 8, 8)

    Domain-agnostic: no RGB shortcuts, no coordinate grids, no SE attention.
    Standard ViT patch embedding + conv context — works for any visual input.
    """

    def __init__(self, input_dim, embedding_dim=64, filter_size=8):
        super().__init__()
        import math

        if len(input_dim) <= 1:
            raise ValueError("PatchContextEncoderV9 requires image observations")

        C_in, H, W = input_dim
        self.filter_size = filter_size
        patch_size = H // filter_size

        assert H == W, f"Expected square input, got {H}×{W}"
        assert H % filter_size == 0, (
            f"Input {H} not divisible by filter_size {filter_size}")

        mid_dim = min(256, embedding_dim * 4)

        # Phase 1: per-patch independent embedding
        # Conv2d with kernel_size=stride=patch_size is equivalent to
        # unfold → Linear for each non-overlapping patch.
        self.patch_embed = nn.Sequential(
            nn.Conv2d(C_in, mid_dim,
                      kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(min(32, mid_dim), mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, embedding_dim, kernel_size=1),
            nn.GELU(),
        )

        # Phase 2: local context refinement (residual)
        # Two 3×3 conv layers let each token see its 8 neighbours.
        # Applied as a residual so it can only add context, never erase
        # the per-tile identity from Phase 1.
        self.context = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1),
            nn.GroupNorm(min(16, embedding_dim), embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1),
            nn.GroupNorm(min(16, embedding_dim), embedding_dim),
        )

    def forward(self, x):
        h = self.patch_embed(x)                           # (B, D, fs, fs)
        h = h + self.context(h)                           # residual context
        return torch.nn.functional.gelu(h)


def make_ae_v9(input_dim, embedding_dim=None, filter_size=None):
    """Patch-based encoder v9 + local context refinement.

    ViT-style non-overlapping patch embedding.  Each token independently
    encodes its raw pixel tile, then a residual conv block adds cross-token
    context.  See PatchContextEncoderV9 docstring.
    """
    import math
    embedding_dim = embedding_dim or 64

    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)

    H, W = input_dim[1], input_dim[2]
    filter_size = filter_size or 8

    encoder = PatchContextEncoderV9(
        input_dim, embedding_dim=embedding_dim, filter_size=filter_size)

    # Decoder: single ConvTranspose to upsample from filter_size to H,
    # then a refining conv.  Simpler than v5 decoder since we only need
    # one upsampling step (patch_size = H // filter_size).
    patch_size = H // filter_size
    n_layers = int(round(math.log2(H / filter_size)))

    # Re-use v5 style decoder for fair comparison
    if n_layers == 1:
        channels = [input_dim[0], embedding_dim]
    elif n_layers == 2:
        channels = [input_dim[0], 64, embedding_dim]
    elif n_layers == 3:
        channels = [input_dim[0], 64, 128, embedding_dim]
    elif n_layers == 4:
        channels = [input_dim[0], 32, 64, 128, embedding_dim]
    else:
        channels = ([input_dim[0]]
                    + [min(64 * (2 ** i), 256) for i in range(n_layers - 1)]
                    + [embedding_dim])

    decoder_layers = []
    for i in reversed(range(n_layers)):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.ReLU())
    decoder_layers.append(nn.AdaptiveAvgPool2d((H, W)))
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


class ColorCoordEncoderV6(nn.Module):
    """Grid-aligned encoder with explicit pooled color and coordinate paths.

    The v5 encoder fixed spatial alignment, but still relies entirely on the
    conv trunk to preserve rare color cues. For DoorKey-8x8 this can wash out
    the green goal tile because it is visually simple and class-rare.

    This encoder keeps the clean 8x8 alignment from v5 and adds:
      - a pooled RGB shortcut so each token sees its cell-local average color
      - normalized x/y coordinate channels so tokens can use absolute position
      - a 1x1 fusion block after concatenating trunk/color/coord features
    """

    def __init__(self, input_dim, embedding_dim=64, filter_size=8):
        super().__init__()
        if len(input_dim) <= 1:
            raise ValueError('ColorCoordEncoderV6 requires image observations')

        in_channels, height, width = input_dim
        if height != width:
            raise ValueError(f'Expected square inputs, got {(height, width)}')
        if height % filter_size != 0:
            raise ValueError(
                f'Input size {height} must be divisible by filter_size {filter_size}')

        self.filter_size = filter_size
        self.pool_stride = height // filter_size

        color_dim = max(8, embedding_dim // 4)
        coord_dim = max(4, embedding_dim // 8)
        trunk_dim = embedding_dim - color_dim - coord_dim
        if trunk_dim < 16:
            raise ValueError(
                f'embedding_dim={embedding_dim} is too small for v6 feature split')

        self.color_pool = nn.AvgPool2d(kernel_size=self.pool_stride, stride=self.pool_stride)
        self.color_proj = nn.Sequential(
            nn.Conv2d(in_channels, color_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.coord_proj = nn.Sequential(
            nn.Conv2d(2, coord_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels + 2, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, trunk_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1),
            nn.ReLU(),
            ResidualBlock(embedding_dim, embedding_dim),
        )

    def _coord_grid(self, batch_size, device, dtype):
        axis = torch.linspace(-1.0, 1.0, self.filter_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return grid.expand(batch_size, -1, -1, -1)

    def forward(self, x):
        batch_size = x.shape[0]
        coord_hi = self._coord_grid(batch_size, x.device, x.dtype)
        if coord_hi.shape[-1] != x.shape[-1]:
            coord_hi = torch.nn.functional.interpolate(
                coord_hi, size=x.shape[-2:], mode='bilinear', align_corners=False)

        trunk = self.trunk(torch.cat([x, coord_hi], dim=1))
        color = self.color_proj(self.color_pool(x))
        coord_lo = self.coord_proj(self._coord_grid(batch_size, x.device, x.dtype))
        fused = torch.cat([trunk, color, coord_lo], dim=1)
        return self.fuse(fused)


# ── v6 ablation variants ──────────────────────────────────────────────
# v6a: RGB shortcut only (no coordinate grid)
# v6b: Coordinate grid only (no RGB shortcut)
# v6c: Both shortcuts + full-width trunk (no capacity reduction)


class ColorOnlyEncoderV6a(nn.Module):
    """v6 ablation: RGB shortcut only, no coordinate grid.

    Tests whether the pooled RGB path alone is sufficient to capture
    colour-discriminable classes (e.g. goal = green).
    """

    def __init__(self, input_dim, embedding_dim=64, filter_size=8):
        super().__init__()
        if len(input_dim) <= 1:
            raise ValueError('ColorOnlyEncoderV6a requires image observations')

        in_channels, height, width = input_dim
        if height != width:
            raise ValueError(f'Expected square inputs, got {(height, width)}')
        if height % filter_size != 0:
            raise ValueError(
                f'Input size {height} must be divisible by filter_size {filter_size}')

        self.filter_size = filter_size
        self.pool_stride = height // filter_size

        color_dim = max(8, embedding_dim // 4)
        trunk_dim = embedding_dim - color_dim
        assert trunk_dim >= 16, f'embedding_dim={embedding_dim} too small'

        # RGB shortcut
        self.color_pool = nn.AvgPool2d(kernel_size=self.pool_stride,
                                        stride=self.pool_stride)
        self.color_proj = nn.Sequential(
            nn.Conv2d(in_channels, color_dim, kernel_size=1),
            nn.ReLU(),
        )

        # Conv trunk (no coord channels — standard v5 trunk)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, trunk_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1),
            nn.ReLU(),
            ResidualBlock(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        trunk = self.trunk(x)
        color = self.color_proj(self.color_pool(x))
        fused = torch.cat([trunk, color], dim=1)
        return self.fuse(fused)


class CoordOnlyEncoderV6b(nn.Module):
    """v6 ablation: coordinate grid only, no RGB shortcut.

    Tests whether absolute position encoding alone provides the
    discriminability boost, independent of the colour path.
    """

    def __init__(self, input_dim, embedding_dim=64, filter_size=8):
        super().__init__()
        if len(input_dim) <= 1:
            raise ValueError('CoordOnlyEncoderV6b requires image observations')

        in_channels, height, width = input_dim
        if height != width:
            raise ValueError(f'Expected square inputs, got {(height, width)}')
        if height % filter_size != 0:
            raise ValueError(
                f'Input size {height} must be divisible by filter_size {filter_size}')

        self.filter_size = filter_size

        coord_dim = max(4, embedding_dim // 8)
        trunk_dim = embedding_dim - coord_dim
        assert trunk_dim >= 16, f'embedding_dim={embedding_dim} too small'

        # Coordinate projection
        self.coord_proj = nn.Sequential(
            nn.Conv2d(2, coord_dim, kernel_size=1),
            nn.ReLU(),
        )

        # Conv trunk with coord input (same as v6)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels + 2, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, trunk_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1),
            nn.ReLU(),
            ResidualBlock(embedding_dim, embedding_dim),
        )

    def _coord_grid(self, batch_size, device, dtype):
        axis = torch.linspace(-1.0, 1.0, self.filter_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return grid.expand(batch_size, -1, -1, -1)

    def forward(self, x):
        batch_size = x.shape[0]
        coord_hi = self._coord_grid(batch_size, x.device, x.dtype)
        if coord_hi.shape[-1] != x.shape[-1]:
            coord_hi = torch.nn.functional.interpolate(
                coord_hi, size=x.shape[-2:], mode='bilinear', align_corners=False)

        trunk = self.trunk(torch.cat([x, coord_hi], dim=1))
        coord_lo = self.coord_proj(self._coord_grid(batch_size, x.device, x.dtype))
        fused = torch.cat([trunk, coord_lo], dim=1)
        return self.fuse(fused)


class WideTrunkEncoderV6c(nn.Module):
    """v6 ablation: RGB + coords + full-width trunk (no capacity reduction).

    In original v6, the trunk is narrowed to make room for color_dim + coord_dim.
    This variant keeps the trunk at full embedding_dim width and projects the
    concatenation down, testing whether v6's capacity split hurts the trunk.
    """

    def __init__(self, input_dim, embedding_dim=64, filter_size=8):
        super().__init__()
        if len(input_dim) <= 1:
            raise ValueError('WideTrunkEncoderV6c requires image observations')

        in_channels, height, width = input_dim
        if height != width:
            raise ValueError(f'Expected square inputs, got {(height, width)}')
        if height % filter_size != 0:
            raise ValueError(
                f'Input size {height} must be divisible by filter_size {filter_size}')

        self.filter_size = filter_size
        self.pool_stride = height // filter_size

        color_dim = max(8, embedding_dim // 4)
        coord_dim = max(4, embedding_dim // 8)
        # Full-width trunk — no capacity reduction
        trunk_dim = embedding_dim

        # RGB shortcut
        self.color_pool = nn.AvgPool2d(kernel_size=self.pool_stride,
                                        stride=self.pool_stride)
        self.color_proj = nn.Sequential(
            nn.Conv2d(in_channels, color_dim, kernel_size=1),
            nn.ReLU(),
        )

        # Coordinate projection
        self.coord_proj = nn.Sequential(
            nn.Conv2d(2, coord_dim, kernel_size=1),
            nn.ReLU(),
        )

        # Full-width conv trunk with coord input
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels + 2, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, trunk_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Wider concatenation → project down to embedding_dim
        concat_dim = trunk_dim + color_dim + coord_dim  # > embedding_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(concat_dim, embedding_dim, kernel_size=1),
            nn.ReLU(),
            ResidualBlock(embedding_dim, embedding_dim),
        )

    def _coord_grid(self, batch_size, device, dtype):
        axis = torch.linspace(-1.0, 1.0, self.filter_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return grid.expand(batch_size, -1, -1, -1)

    def forward(self, x):
        batch_size = x.shape[0]
        coord_hi = self._coord_grid(batch_size, x.device, x.dtype)
        if coord_hi.shape[-1] != x.shape[-1]:
            coord_hi = torch.nn.functional.interpolate(
                coord_hi, size=x.shape[-2:], mode='bilinear', align_corners=False)

        trunk = self.trunk(torch.cat([x, coord_hi], dim=1))
        color = self.color_proj(self.color_pool(x))
        coord_lo = self.coord_proj(self._coord_grid(batch_size, x.device, x.dtype))
        fused = torch.cat([trunk, color, coord_lo], dim=1)
        return self.fuse(fused)


class GatedInputSkipEncoderV8(nn.Module):
    """Full-capacity trunk + gated input-level skip connection.

    Combines the best properties of v5 and v6 in a domain-agnostic way:

      * The conv trunk is IDENTICAL to v5 (3-layer strided, full embedding_dim
        width).  v5 proved best for structural / textural classes (wall 96.5 %,
        door 99.4 %, key 100 %).

      * A lightweight input-level skip pools the raw observation to the latent
        resolution and projects it to a small feature vector per token.  This
        gives each token direct access to its local statistics (colour, texture)
        without relying on the trunk to preserve them through three stride-2
        layers — the general-purpose version of v6's hand-coded RGB shortcut.

      * A learned **sigmoid gate** controls the per-token, per-channel blend
        of trunk features and skip features.  When the gate is 0 the output
        equals the pure trunk (v5 behaviour); when it opens the skip signal
        is mixed in.  The network can learn to open the gate selectively for
        tokens whose identity depends on raw input statistics (e.g. goal =
        uniform green) while keeping it closed for tokens where the deep
        trunk already provides sufficient discrimination (e.g. wall, door).

    No SE attention, no coordinate grids, no multi-scale skips — each of
    those was tried in v7 and **hurt** dominant-class accuracy (wall 96 → 78 %).
    This design is the minimal intervention that adds input-identity awareness
    to v5 without compromising its proven strengths.

    Architecture (64 × 64 → 8 × 8, embedding_dim = 64):

        Input (C, 64, 64)
          ├─ Trunk (v5):  Conv(C→64, s=2) → Conv(64→128, s=2) → Conv(128→64, s=2)
          │               → f_trunk (64, 8, 8)
          │
          └─ Input skip:  AdaptiveAvgPool2d(8) → Conv1×1(C→16) → ReLU
                          → Conv1×1(16→16) → ReLU  → f_skip (16, 8, 8)

        cat([f_trunk, f_skip]) = (80, 8, 8)
          → gate  = σ(Conv1×1(80→64))               ∈ [0, 1]
          → merge = ReLU(Conv1×1(80→64))
          → out   = gate · merge + (1 − gate) · f_trunk     (gated residual)
          → ResidualBlock → final (64, 8, 8)
    """

    def __init__(self, input_dim, embedding_dim=64, filter_size=8):
        super().__init__()
        import math

        if len(input_dim) <= 1:
            raise ValueError("GatedInputSkipEncoderV8 requires image observations")

        C_in, H, W = input_dim
        self.filter_size = filter_size

        n_layers = int(round(math.log2(H / filter_size)))
        assert n_layers >= 1
        assert H // (2 ** n_layers) == filter_size, (
            f"Input {H} / 2^{n_layers} != filter_size {filter_size}")

        # --- v5-identical trunk (full width) ---
        if n_layers == 1:
            channels = [C_in, embedding_dim]
        elif n_layers == 2:
            channels = [C_in, 64, embedding_dim]
        elif n_layers == 3:
            channels = [C_in, 64, 128, embedding_dim]
        elif n_layers == 4:
            channels = [C_in, 32, 64, 128, embedding_dim]
        else:
            channels = ([C_in]
                        + [min(64 * (2 ** i), 256) for i in range(n_layers - 1)]
                        + [embedding_dim])

        trunk_layers = []
        for i in range(n_layers):
            trunk_layers.append(nn.Conv2d(
                channels[i], channels[i + 1],
                kernel_size=4, stride=2, padding=1))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)

        # --- Input-level skip (domain-agnostic) ---
        skip_dim = max(embedding_dim // 4, 8)          # 16 for D=64
        self.input_pool = nn.AdaptiveAvgPool2d(filter_size)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(C_in, skip_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(skip_dim, skip_dim, kernel_size=1),
            nn.ReLU(),
        )

        # --- Gated fusion ---
        fused_dim = embedding_dim + skip_dim
        self.gate_net = nn.Sequential(
            nn.Conv2d(fused_dim, embedding_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.merge_net = nn.Sequential(
            nn.Conv2d(fused_dim, embedding_dim, kernel_size=1),
            nn.ReLU(),
        )

        # --- Refinement ---
        self.refine = ResidualBlock(embedding_dim, embedding_dim)

    def forward(self, x):
        f_trunk = self.trunk(x)                                     # (B, D, fs, fs)
        f_skip = self.skip_proj(self.input_pool(x))                 # (B, skip, fs, fs)

        combined = torch.cat([f_trunk, f_skip], dim=1)              # (B, D+skip, fs, fs)
        gate = self.gate_net(combined)                              # (B, D, fs, fs) ∈ [0,1]
        merged = self.merge_net(combined)                           # (B, D, fs, fs)

        out = gate * merged + (1.0 - gate) * f_trunk               # gated residual
        out = self.refine(out)
        return out


def make_ae_v8(input_dim, embedding_dim=None, filter_size=None):
    """Full trunk + gated input skip encoder v8.

    Domain-agnostic design: v5's proven trunk at full width, plus a learned
    gated skip from pooled input features.  See GatedInputSkipEncoderV8.
    """
    import math
    embedding_dim = embedding_dim or 64

    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)

    H, W = input_dim[1], input_dim[2]
    filter_size = filter_size or 8

    encoder = GatedInputSkipEncoderV8(
        input_dim, embedding_dim=embedding_dim, filter_size=filter_size)

    # Decoder mirrors v5 for fair comparison
    n_layers = int(round(math.log2(H / filter_size)))
    if n_layers == 1:
        channels = [input_dim[0], embedding_dim]
    elif n_layers == 2:
        channels = [input_dim[0], 64, embedding_dim]
    elif n_layers == 3:
        channels = [input_dim[0], 64, 128, embedding_dim]
    elif n_layers == 4:
        channels = [input_dim[0], 32, 64, 128, embedding_dim]
    else:
        channels = ([input_dim[0]]
                    + [min(64 * (2 ** i), 256) for i in range(n_layers - 1)]
                    + [embedding_dim])

    decoder_layers = []
    for i in reversed(range(n_layers)):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.ReLU())
    decoder_layers.append(nn.AdaptiveAvgPool2d((H, W)))
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018).

    Domain-agnostic: learns per-channel importance weights via global average
    pooling → bottleneck MLP → sigmoid gating.  This lets the encoder
    dynamically emphasise colour-sensitive channels for uniform-texture
    classes (e.g. goal) and texture-sensitive channels for patterned classes
    (e.g. door/key) without any hard-coded domain knowledge.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        scale = x.mean(dim=[2, 3])                     # (B, C)
        scale = self.fc(scale).view(B, C, 1, 1)        # (B, C, 1, 1)
        return x * scale


class MultiScaleEncoderV7(nn.Module):
    """Generalised multi-scale encoder with channel attention.

    Improvements over v6 (which was MiniGrid-specific):
      * No hand-crafted RGB pooling or coordinate grids.
      * Multi-scale skip connections from *every* intermediate layer
        preserve both fine-grained texture (door, key) and coarse colour
        signals (goal) — a domain-agnostic alternative to the explicit
        colour shortcut.
      * Squeeze-and-Excitation (SE) attention lets the network learn which
        channels matter on a per-sample basis, instead of relying on a
        fixed colour / coordinate split.
      * Learnable spatial positional embeddings (à la ViT) provide
        location awareness without assuming a grid-world structure.
      * Full-width backbone (embedding_dim channels in the last conv)
        restores capacity that v6 sacrificed for its colour/coord paths.

    Architecture (for 64×64 → 8×8, embedding_dim=64):

        Input (C, 64, 64)
          → L1: Conv(C→64,  k=4, s=2, p=1) + ReLU   → f1 (64, 32, 32)
          → L2: Conv(64→128, k=4, s=2, p=1) + ReLU   → f2 (128, 16, 16)
          → L3: Conv(128→D,  k=4, s=2, p=1) + ReLU   → f3 (D,   8,  8)

        Skip-1: AvgPool(f1 → 8×8) → Conv1×1(64  → D//4) → s1 (D//4, 8, 8)
        Skip-2: AvgPool(f2 → 8×8) → Conv1×1(128 → D//4) → s2 (D//4, 8, 8)

        Fuse:  cat([f3, s1, s2])  → (D + D//2, 8, 8)
               → Conv1×1 → ReLU  → (D, 8, 8)

        SE attention  → channel re-weighting
        + learnable pos_embed (1, D, 8, 8)
        → ResidualBlock  → output (D, 8, 8)
    """

    def __init__(self, input_dim, embedding_dim=64, filter_size=8):
        super().__init__()
        import math

        if len(input_dim) <= 1:
            raise ValueError("MultiScaleEncoderV7 requires image observations")

        C_in, H, W = input_dim
        self.filter_size = filter_size

        n_layers = int(round(math.log2(H / filter_size)))
        assert n_layers >= 1
        assert H // (2 ** n_layers) == filter_size, (
            f"Input {H} / 2^{n_layers} != filter_size {filter_size}")

        # --- backbone channel progression (same as v5) ---
        if n_layers == 1:
            channels = [C_in, embedding_dim]
        elif n_layers == 2:
            channels = [C_in, 64, embedding_dim]
        elif n_layers == 3:
            channels = [C_in, 64, 128, embedding_dim]
        elif n_layers == 4:
            channels = [C_in, 32, 64, 128, embedding_dim]
        else:
            channels = ([C_in]
                        + [min(64 * (2 ** i), 256) for i in range(n_layers - 1)]
                        + [embedding_dim])

        # --- backbone layers (individually stored for skip access) ---
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1],
                          kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            ))

        # --- multi-scale skip projections (all intermediate layers) ---
        skip_dim = max(embedding_dim // 4, 8)
        self.skip_projs = nn.ModuleList()
        for i in range(n_layers - 1):          # skip from every layer except last
            self.skip_projs.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(filter_size),
                nn.Conv2d(channels[i + 1], skip_dim, kernel_size=1),
                nn.ReLU(),
            ))
        total_skip_dim = skip_dim * (n_layers - 1)

        # --- fusion ---
        fused_dim = embedding_dim + total_skip_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_dim, embedding_dim, kernel_size=1),
            nn.ReLU(),
        )

        # --- channel attention ---
        self.se = SEBlock(embedding_dim, reduction=4)

        # --- learnable positional embedding ---
        self.pos_embed = nn.Parameter(
            torch.randn(1, embedding_dim, filter_size, filter_size) * 0.02)

        # --- final refinement ---
        self.refine = ResidualBlock(embedding_dim, embedding_dim)

    def forward(self, x):
        # Run backbone, collecting intermediate features
        intermediates = []
        h = x
        for layer in self.layers:
            h = layer(h)
            intermediates.append(h)

        backbone_out = intermediates[-1]                       # (B, D, fs, fs)

        # Multi-scale skip connections
        skips = []
        for i, proj in enumerate(self.skip_projs):
            skips.append(proj(intermediates[i]))

        # Fuse
        if skips:
            fused = torch.cat([backbone_out] + skips, dim=1)
        else:
            fused = backbone_out
        out = self.fuse(fused)

        # Channel attention
        out = self.se(out)

        # Positional embedding
        out = out + self.pos_embed

        # Refinement
        out = self.refine(out)
        return out


def make_ae_v7(input_dim, embedding_dim=None, filter_size=None):
    """Multi-scale encoder v7 with SE attention + learned positions.

    Domain-agnostic replacement for v6's MiniGrid-specific colour/coord paths.
    See MultiScaleEncoderV7 docstring for architecture details.
    """
    import math
    embedding_dim = embedding_dim or 64

    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)

    H, W = input_dim[1], input_dim[2]
    filter_size = filter_size or 8

    encoder = MultiScaleEncoderV7(input_dim, embedding_dim=embedding_dim,
                                  filter_size=filter_size)

    # Decoder mirrors v5 for fair comparison (only encoder changes)
    n_layers = int(round(math.log2(H / filter_size)))
    if n_layers == 1:
        channels = [input_dim[0], embedding_dim]
    elif n_layers == 2:
        channels = [input_dim[0], 64, embedding_dim]
    elif n_layers == 3:
        channels = [input_dim[0], 64, 128, embedding_dim]
    elif n_layers == 4:
        channels = [input_dim[0], 32, 64, 128, embedding_dim]
    else:
        channels = ([input_dim[0]]
                    + [min(64 * (2 ** i), 256) for i in range(n_layers - 1)]
                    + [embedding_dim])

    decoder_layers = []
    for i in reversed(range(n_layers)):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.ReLU())
    decoder_layers.append(nn.AdaptiveAvgPool2d((H, W)))
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


def make_ae_v6(input_dim, embedding_dim=None, filter_size=None):
    """Color- and coordinate-aware grid encoder for MiniGrid-style inputs."""
    embedding_dim = embedding_dim or 64

    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)

    H, W = input_dim[1], input_dim[2]
    filter_size = filter_size or 8

    encoder = ColorCoordEncoderV6(input_dim, embedding_dim=embedding_dim, filter_size=filter_size)

    # Decoder mirrors v5 so comparisons isolate the encoder change.
    channels = [input_dim[0], 64, 128, embedding_dim]
    decoder_layers = []
    for i in reversed(range(3)):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.ReLU())
    decoder_layers.append(nn.AdaptiveAvgPool2d((H, W)))
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


def _make_v6_decoder(input_dim, embedding_dim):
    """Shared v5-style decoder for all v6 ablation variants."""
    H, W = input_dim[1], input_dim[2]
    channels = [input_dim[0], 64, 128, embedding_dim]
    decoder_layers = []
    for i in reversed(range(3)):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.ReLU())
    decoder_layers.append(nn.AdaptiveAvgPool2d((H, W)))
    return nn.Sequential(*decoder_layers)


def make_ae_v6a(input_dim, embedding_dim=None, filter_size=None):
    """v6 ablation — RGB shortcut only (no coordinate grid)."""
    embedding_dim = embedding_dim or 64
    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)
    filter_size = filter_size or 8
    encoder = ColorOnlyEncoderV6a(input_dim, embedding_dim=embedding_dim,
                                   filter_size=filter_size)
    decoder = _make_v6_decoder(input_dim, embedding_dim)
    return encoder, decoder


def make_ae_v6b(input_dim, embedding_dim=None, filter_size=None):
    """v6 ablation — coordinate grid only (no RGB shortcut)."""
    embedding_dim = embedding_dim or 64
    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)
    filter_size = filter_size or 8
    encoder = CoordOnlyEncoderV6b(input_dim, embedding_dim=embedding_dim,
                                   filter_size=filter_size)
    decoder = _make_v6_decoder(input_dim, embedding_dim)
    return encoder, decoder


def make_ae_v6c(input_dim, embedding_dim=None, filter_size=None):
    """v6 ablation — RGB + coords + full-width trunk (no capacity reduction)."""
    embedding_dim = embedding_dim or 64
    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)
    filter_size = filter_size or 8
    encoder = WideTrunkEncoderV6c(input_dim, embedding_dim=embedding_dim,
                                   filter_size=filter_size)
    decoder = _make_v6_decoder(input_dim, embedding_dim)
    return encoder, decoder


def make_split_encoder_ae(input_dim, embedding_dim=None, ctx_channels=None, filter_size=None):
    """
    Split-encoder architecture: a lightweight local encoder (small RF) feeds the VQ
    codebook, while a deeper dilated context encoder feeds the decoder.

    Both branches produce spatially aligned feature maps at the same (H_lat, W_lat)
    via two stride-2 downsampling steps. AdaptiveAvgPool2d is applied when
    filter_size is given to ensure exact grid alignment.

    Returns (local_encoder, ctx_encoder, decoder).
    """
    embedding_dim = embedding_dim or 64
    ctx_channels = ctx_channels or 64
    C_in = input_dim[0]
    original_h, original_w = input_dim[1], input_dim[2]

    # ---- local encoder: small RF, 2x stride-2 downsampling ----
    local_layers = [
        nn.Conv2d(C_in, 64, 5, 2, 2),          # k=5, s=2 — first halving
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, 1),             # k=3, s=1 — RF expands slightly
        nn.ReLU(),
        nn.Conv2d(64, embedding_dim, 4, 2, 1),  # k=4, s=2 — second halving
        nn.ReLU(),
    ]

    # ---- context encoder: same spatial downsampling, large RF via dilation ----
    ctx_layers = [
        nn.Conv2d(C_in, 64, 5, 2, 2),                    # k=5, s=2
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, 2, dilation=2),          # d=2, same spatial size
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, 4, dilation=4),          # d=4, same spatial size
        nn.ReLU(),
        nn.Conv2d(64, ctx_channels, 4, 2, 1),             # k=4, s=2 — second halving
        nn.ReLU(),
    ]

    if filter_size is not None:
        local_layers.append(nn.AdaptiveAvgPool2d(filter_size))
        ctx_layers.append(nn.AdaptiveAvgPool2d(filter_size))

    local_encoder = nn.Sequential(*local_layers)
    ctx_encoder = nn.Sequential(*ctx_layers)

    # ---- decoder: 2x ConvTranspose upsampling back to input size ----
    test_input = torch.ones(1, *input_dim, dtype=torch.float32)
    enc_out_shape = local_encoder(test_input).shape[1:]
    print('Split AE — local enc out:', enc_out_shape, '| ctx_channels:', ctx_channels)

    decoder = nn.Sequential(
        nn.ConvTranspose2d(embedding_dim, 64, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 5, 2, 2, output_padding=1),
        nn.ReLU(),
        nn.Conv2d(32, C_in, 3, 1, 1),
        nn.AdaptiveAvgPool2d((original_h, original_w)),
    )

    return local_encoder, ctx_encoder, decoder


def make_ae(input_dim, embedding_dim, filter_size, version='2'):
    version = str(version)
    if version == '1':
        return make_ae_v1(input_dim, embedding_dim, (filter_size, filter_size))
    elif version == '2':
        return make_ae_v2(input_dim, embedding_dim, filter_size)
    elif version == '3':
        return make_ae_v3(input_dim, embedding_dim, filter_size)
    elif version == '4':
        return make_ae_v4(input_dim, embedding_dim, filter_size)
    elif version == '5':
        return make_ae_v5(input_dim, embedding_dim, filter_size)
    elif version == '6':
        return make_ae_v6(input_dim, embedding_dim, filter_size)
    elif version == '6a':
        return make_ae_v6a(input_dim, embedding_dim, filter_size)
    elif version == '6b':
        return make_ae_v6b(input_dim, embedding_dim, filter_size)
    elif version == '6c':
        return make_ae_v6c(input_dim, embedding_dim, filter_size)
    elif version == '7':
        return make_ae_v7(input_dim, embedding_dim, filter_size)
    elif version == '8':
        return make_ae_v8(input_dim, embedding_dim, filter_size)
    elif version == '9':
        return make_ae_v9(input_dim, embedding_dim, filter_size)
    elif version == 'nature_ae':
        return make_nature_ae(input_dim, embedding_dim, filter_size, vanilla=False)
    elif version == 'nature':
        return make_nature_ae(input_dim, embedding_dim, filter_size, vanilla=True)
    raise ValueError(f'Invalid AE version, {version}')


def args_update(args, key, value):
    if args.wandb:
        args.update({key: value}, allow_val_change=True)
    else:
        setattr(args, key, value)

    log_param_updates(args, {key: value})


def construct_ae_model(input_dim, args, load=True, latent_activation=False):
    new_hash = make_model_hash(args, model_vars=AE_MODEL_VARS, exp_type='encoder')
    model = None
    args_update(args, 'ae_model_hash', new_hash)

    if args.ae_model_type not in ('identity', 'flatten', 'local_ctx_vqvae'):
        encoder, decoder = make_ae(
            input_dim, args.embedding_dim, args.filter_size, version=args.ae_model_version)
        test_input = torch.ones(1, *input_dim, dtype=torch.float32)
        encoder_out_shape = encoder(test_input).shape[1:]
        encoder_type = 'dense' if (len(encoder_out_shape) == 1) else 'cnn'

    if args.ae_model_type in CONTINUOUS_ENCODER_TYPES:
        if args.ae_model_type in ('ae', 'vae', 'fta_ae'):
            stochastic = args.ae_model_type == 'vae'
            fta = args.ae_model_type == 'fta_ae'
            fta_params = {
                'tiles': args.fta_tiles,
                'bound_low': args.fta_bound_low,
                'bound_high': args.fta_bound_high,
                'eta': args.fta_eta
            }

            args_update(args, 'codebook_size', None)

            model = AEModel(input_dim, latent_dim=args.latent_dim, encoder=encoder,
                            decoder=decoder, stochastic=stochastic, fta=fta, fta_params=fta_params,
                            latent_activation=latent_activation)
            args_update(args, 'final_latent_dim', model.latent_dim)
            print(f'Constructed {args.ae_model_type.upper()} with ' + \
                  f'{args.final_latent_dim}-dim latent space')

            TrainerClass = AETrainer if args.ae_model_type in ('ae', 'fta_ae') else VAETrainer
            trainer = TrainerClass(model, lr=args.learning_rate, log_freq=-1, grad_clip=args.ae_grad_clip)

        elif args.ae_model_type == 'soft_vqvae':  # VQVAE with quantized latent space
            n_latents = args.latent_dim if encoder_type == 'dense' else None


            model = VQVAEModel(
                input_dim, codebook_size=args.codebook_size, embedding_dim=args.embedding_dim,
                encoder=encoder, decoder=decoder, n_latents=n_latents, quantized_enc=True,
                sparsity=args.repr_sparsity, sparsity_type=args.sparsity_type)
            args_update(args, 'final_latent_dim', model.n_latent_embeds * args.codebook_size)
            print(f'Constructed Soft VQVAE with {model.n_latent_embeds} ' + \
                  f'latents and {args.codebook_size} codebook entries')
            trainer = VQVAETrainer(
                model, lr=args.learning_rate, log_freq=-1, grad_clip=args.ae_grad_clip,
                entropy_penalty_coef=safe_getattr(args, 'entropy_penalty_coef', 0.0),
                code_dropout_rate=safe_getattr(args, 'code_dropout_rate', 0.0),
                mae_mask_ratio=safe_getattr(args, 'mae_mask_ratio', 0.0),
                mae_patch_size=safe_getattr(args, 'mae_patch_size', 4),
                mae_loss_coef=safe_getattr(args, 'mae_loss_coef', 1.0),
            )

        if load:
            load_model(
                model, args, exp_type='encoder', model_vars=AE_MODEL_VARS,
                model_hash=args.ae_model_hash)

    elif args.ae_model_type in DISCRETE_ENCODER_TYPES:
        TrainerClass = VQVAETrainer if args.ae_model_type \
                                       in ('vqvae', 'soft_vqvae', 'local_ctx_vqvae') else AETrainer
        n_latents = None  # default; overridden for types that need it
        if args.ae_model_type != 'local_ctx_vqvae':
            n_latents = args.latent_dim if encoder_type == 'dense' else None

        if args.ae_model_type == 'vqvae':

            model = VQVAEModel(
                input_dim, codebook_size=args.codebook_size, embedding_dim=args.embedding_dim,
                encoder=encoder, decoder=decoder, n_latents=n_latents,
                commitment_cost=getattr(args, 'commitment_cost', 0.25),
                ema_decay=getattr(args, 'ema_decay', 0.99),
                dead_code_threshold=getattr(args, 'dead_code_threshold', 0.0))
            args_update(args, 'final_latent_dim', model.n_latent_embeds * args.codebook_size)
            print(f'Constructed VQVAE with {model.n_latent_embeds} ' + \
                  f'latents and {args.codebook_size} codebook entries')

        elif args.ae_model_type == 'dae':
            model = DAEModel(input_dim, encoder=encoder, decoder=decoder)
            args_update(args, 'final_latent_dim', np.prod(model.encoder_out_shape))
            print(f'Constructed DAE with {np.prod(model.encoder_out_shape[1:])} latents ' + \
                  f'and {model.n_channels} codebook entries')

        elif args.ae_model_type == 'softmax_ae':
            model = SoftmaxAEModel(
                input_dim, codebook_size=args.codebook_size,
                encoder=encoder, decoder=decoder, n_latents=n_latents)
            args_update(args, 'final_latent_dim', np.prod(model.encoder_out_shape))
            print(f'Constructed hard gumbel AE with {model.encoder_out_shape[1:]} latents')

        elif args.ae_model_type == 'hard_fta_ae':
            model = HardFTAAEModel(
                input_dim, codebook_size=args.codebook_size,
                encoder=encoder, decoder=decoder, n_latents=n_latents)
            args_update(args, 'final_latent_dim', np.prod(model.encoder_out_shape))
            print(f'Constructed hard gumbel AE with {model.encoder_out_shape[1:]} latents')

        elif args.ae_model_type == 'local_ctx_vqvae':
            ctx_channels = safe_getattr(args, 'ctx_channels', 64)
            ctx_cond_type = safe_getattr(args, 'ctx_cond_type', 'concat')
            local_encoder, ctx_encoder, _decoder = make_split_encoder_ae(
                input_dim, embedding_dim=args.embedding_dim,
                ctx_channels=ctx_channels, filter_size=args.filter_size)
            model = VQVAEModel(
                input_dim, codebook_size=args.codebook_size, embedding_dim=args.embedding_dim,
                encoder=local_encoder, decoder=_decoder,
                ctx_encoder=ctx_encoder, ctx_cond_type=ctx_cond_type)
            args_update(args, 'final_latent_dim', model.n_latent_embeds * args.codebook_size)
            print(f'Constructed Local-Ctx VQVAE with {model.n_latent_embeds} latents, '
                  f'{args.codebook_size} codebook entries, {ctx_channels} ctx channels '
                  f'({ctx_cond_type} conditioning)')

        if load:
            load_model(
                model, args, exp_type='encoder', model_vars=AE_MODEL_VARS,
                model_hash=args.ae_model_hash)
        if TrainerClass is VQVAETrainer:
            trainer = TrainerClass(
                model, lr=args.learning_rate, log_freq=-1, grad_clip=args.ae_grad_clip,
                entropy_penalty_coef=safe_getattr(args, 'entropy_penalty_coef', 0.0),
                code_dropout_rate=safe_getattr(args, 'code_dropout_rate', 0.0),
                mae_mask_ratio=safe_getattr(args, 'mae_mask_ratio', 0.0),
                mae_patch_size=safe_getattr(args, 'mae_patch_size', 4),
                mae_loss_coef=safe_getattr(args, 'mae_loss_coef', 1.0),
                ctx_aux_coef=safe_getattr(args, 'ctx_aux_coef', 1.0),
            )
        else:
            trainer = TrainerClass(model, lr=args.learning_rate, log_freq=-1, grad_clip=args.ae_grad_clip)

    elif args.ae_model_type == 'flatten':
        model = FlattenModel(input_dim)
        args_update(args, 'final_latent_dim', model.latent_dim)
        print(f'Constructed Flatten model with {model.latent_dim} latents')
        trainer = None

    elif args.ae_model_type == 'identity':
        model = IdentityModel(input_dim, embedding_dim=args.embedding_dim)
        args_update(args, 'final_latent_dim', model.latent_dim)
        print(f'Constructed Identity model with {model.latent_dim} latents')
        trainer = None

    return model, trainer


# Need this because getattr doens't work the same way for wandb args
def safe_getattr(args, attr, default=None):
    try:
        return getattr(args, attr, default)
    except KeyError:
        return default


def construct_trans_model(encoder, args, act_space, load=True):
    new_hash = make_model_hash(args, model_vars=MODEL_VARS, exp_type='trans_model')
    trans_model = None
    args_update(args, 'trans_model_hash', new_hash)

    if args.e2e_loss and args.trans_model_type != 'continuous':
        raise ValueError('End-to-end loss only supported for continuous models!')

    if args.trans_model_type == 'discrete':
        if args.trans_model_version == '1':
            use_soft_embeds = safe_getattr(args, 'use_soft_embeds', False) \
                              or safe_getattr(encoder, 'quantized_enc', False)
            trans_model = DiscreteTransitionModel(
                encoder.n_latent_embeds, encoder.n_embeddings, encoder.embedding_dim,
                act_space, hidden_sizes=[args.trans_hidden] * args.trans_depth,
                stochastic=args.stochastic, stoch_hidden_sizes=[256, 256],
                discretizer_hidden_sizes=[256], use_soft_embeds=use_soft_embeds,
                return_logits=safe_getattr(encoder, 'quantized_enc', False))
        args_update(args, 'final_latent_dim',
                    encoder.n_latent_embeds * encoder.n_embeddings)
        if load:
            load_model(trans_model, args, exp_type='trans_model',
                       model_vars=MODEL_VARS, model_hash=args.trans_model_hash)
        trans_trainer = DiscreteTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, grad_clip=args.ae_grad_clip)

    elif args.trans_model_type == 'continuous':
        if args.trans_model_version == '1':
            trans_model = ContinuousTransitionModel(
                encoder.latent_dim, act_space,
                hidden_sizes=[args.trans_hidden] * args.trans_depth,
                stochastic=args.stochastic,
                stoch_hidden_sizes=[256, 256],
                discretizer_hidden_sizes=[256]
            )
        args_update(args, 'final_latent_dim', encoder.latent_dim)
        if load:
            load_model(trans_model, args, exp_type='trans_model',
                       model_vars=MODEL_VARS, model_hash=args.trans_model_hash)
        trans_trainer = ContinuousTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, grad_clip=args.ae_grad_clip, e2e_loss=args.e2e_loss,
            reward_overestimate_coef=safe_getattr(args, 'trans_reward_overestimate_coef', 0.0),
            reward_zero_target_coef=safe_getattr(args, 'trans_reward_zero_target_coef', 0.0),
            reward_zero_margin=safe_getattr(args, 'trans_reward_zero_margin', 0.0))

    elif args.trans_model_type == 'shared_vq':
        # Don't track gradients if quantizer is external
        def logits_to_state(logits):
            logits = logits.view(
                logits.shape[0], encoder.n_embeddings, encoder.n_latent_embeds)
            with torch.no_grad():
                mask = encoder.sparsity_mask if encoder.sparsity_enabled else None
                quantized = encoder.quantizer(logits, mask)[1]
            states = quantized.view(logits.shape[0], -1)
            return states

        if args.trans_model_version == '1':
            trans_model = ContinuousTransitionModel(
                encoder.latent_dim, act_space,
                hidden_sizes=[args.trans_hidden] * args.trans_depth,
                stochastic=args.stochastic,
                stoch_hidden_sizes=[256, 256],
                discretizer_hidden_sizes=[256],
                logits_to_state_func=logits_to_state
            )
        args_update(args, 'final_latent_dim', encoder.latent_dim)
        if load:
            load_model(trans_model, args, exp_type='trans_model',
                       model_vars=MODEL_VARS, model_hash=args.trans_model_hash)
        trans_trainer = ContinuousTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, grad_clip=args.ae_grad_clip,
            reward_overestimate_coef=safe_getattr(args, 'trans_reward_overestimate_coef', 0.0),
            reward_zero_target_coef=safe_getattr(args, 'trans_reward_zero_target_coef', 0.0),
            reward_zero_margin=safe_getattr(args, 'trans_reward_zero_margin', 0.0))


    elif args.trans_model_type == 'universal_vq':
        if args.trans_model_version == '1':

            use_soft_embeds = safe_getattr(args, 'use_soft_embeds', False) \
                              or safe_getattr(encoder, 'quantized_enc', False)
            embed_snap_enc = encoder if args.vq_trans_state_snap else None

            if args.extra_info and 'scale_embeds' in args.extra_info:
                codebook = encoder.get_codebook()
                zeros = torch.zeros_like(codebook)
                zeros_count = (codebook.isclose(zeros)).sum(dim=1)
                print(codebook.abs().sum(dim=1))

                if (zeros_count < codebook.shape[1] - 1).any():
                    raise ValueError('Codebook must have one or less non-zero per row!')
                elif (zeros_count == codebook.shape[1]).any():
                    n_zero_rows = (zeros_count == codebook.shape[1]).sum()
                    warnings.warn(f'Codebook has {n_zero_rows} rows with all zeros!')

                scale_factor = 1.0 / codebook.sum(dim=1)
                scale_factor = scale_factor.unsqueeze(0)
            else:
                scale_factor = None

            embed_grad_hook = args.extra_info and 'embed_grad_hook' in args.extra_info
            rand_mask = args.extra_info and 'rand_mask' in args.extra_info

            trans_model = UniversalVQTransitionModel(
                encoder.n_latent_embeds, encoder.n_embeddings, encoder.embedding_dim,
                act_space, hidden_sizes=[args.trans_hidden] * args.trans_depth,
                stochastic=args.stochastic, stoch_hidden_sizes=[256, 256],
                discretizer_hidden_sizes=[256], use_soft_embeds=use_soft_embeds,
                use_1d_conv=args.vq_trans_1d_conv, embed_snap_encoder=embed_snap_enc,
                embed_scale_factor=scale_factor, embed_grad_hook=embed_grad_hook,
                rand_mask=rand_mask)
        args_update(args, 'final_latent_dim',
                    encoder.n_latent_embeds * encoder.n_embeddings)
        if load:
            load_model(trans_model, args, exp_type='trans_model',
                       model_vars=MODEL_VARS, model_hash=args.trans_model_hash)
        trans_trainer = UniversalVQTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, loss_type=args.vq_trans_loss_type, grad_clip=args.ae_grad_clip)


    elif args.trans_model_type == 'transformer':
        if args.trans_model_version == '1':
            trans_model = TransformerTransitionModel(
                encoder.codebook_size, encoder.embedding_dim, act_space,
                num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                dim_feedforward=1024, dropout=0.2, stochastic=args.stochastic)
        args_update(args, 'final_latent_dim',
                    encoder.n_latent_embeds * encoder.codebook_size)
        if load:
            load_model(trans_model, args, exp_type='trans_model',
                       model_vars=MODEL_VARS, model_hash=args.trans_model_hash)
        trans_trainer = TransformerTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            grad_clip=args.ae_grad_clip)

    elif args.trans_model_type == 'transformerdec':
        if args.trans_model_version == '1':
            trans_model = TransformerDecTransitionModel(
                encoder.codebook_size, encoder.embedding_dim, act_space,
                num_heads=4, num_decoder_layers=6, dim_feedforward=256,
                dropout=0.1, stochastic=args.stochastic)
        args_update(args, 'final_latent_dim',
                    encoder.n_latent_embeds * encoder.codebook_size)
        if load:
            load_model(trans_model, args, exp_type='trans_model',
                       model_vars=MODEL_VARS, model_hash=args.trans_model_hash)
        trans_trainer = TransformerTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            grad_clip=args.ae_grad_clip)

    else:
        raise ValueError(f'No trans_model_type, "{args.trans_model_type}"!')

    return trans_model, trans_trainer


def make_model_hash(args=None, model_vars=MODEL_VARS, **kwargs):
    """MD5 hash of a dictionary."""
    args_dict = vars(args)
    args_dict = args_dict.get('_items', args_dict)
    if args is not None:
        for model_param in model_vars:
            if model_param in args_dict:
                kwargs[model_param] = args_dict[model_param]
    dhash = hashlib.md5()
    kwargs = {k: int(v) if isinstance(v, (np.int32, np.int64)) \
        else v for k, v in kwargs.items()}
    encoded = json.dumps(dict(kwargs), sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


MODEL_SAVE_FORMAT = '{base}/models/{env}/model_{hash}.pt'


def _model_path(args, model_hash):
    base = getattr(args, 'model_dir', '.') or '.'
    path = MODEL_SAVE_FORMAT.format(
        base=base, env=args.env_name, hash=model_hash)
    return path.replace(':', '-')


def save_model(model, args, model_hash=None, model_vars=MODEL_VARS, **kwargs):
    if model_hash is None:
        model_hash = make_model_hash(args, model_vars=model_vars, **kwargs)
    save_path = _model_path(args, model_hash)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to "{save_path}"')
    return model_hash


def load_model(
        model, args, model_hash=None, return_hash=False,
        model_vars=MODEL_VARS, **kwargs):
    if model_hash is None:
        model_hash = make_model_hash(args, model_vars=model_vars, **kwargs)
    file_path = _model_path(args, model_hash)
    if not os.path.exists(file_path):
        print(f'No model found at "{file_path}", not loading')
        model = None
    else:
        print(f'Model found at "{file_path}", loading')
        try:
            model.load_state_dict(torch.load(file_path, map_location=args.device), strict=False)
        except RuntimeError as e:
            print(f'Failed to load model at {file_path}!')
            raise e

    if return_hash:
        return model, model_hash
    return model
