"""Depth-guided lifting and BEV projection -- the paper's core substitution for LSS.

Moved from the released `Model/model.py`. The lifting arithmetic is preserved exactly; what
changed is that the BEV window comes from a shared `BevGrid` and the depth clamp comes from
config, because the two are coupled and the released code hardcoded both (see
`unidepthlss/geometry.py`).

One block was also lifted out of `forward()` into its own `lift_to_ego` method. That is a pure
code move -- verified bit-for-bit by `dev/capture_forward_reference.py` -- done so the stage is
addressable by name for `torch.compile`, which cannot target a span of statements inside a
larger `forward()`.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

from unidepthlss.geometry import BevGrid

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def scatter_add(src, index, *, dim=0, out=None, dim_size=None):
    """Pure-torch scatter-add.

    The released code tried `torch_scatter.scatter_add` first and fell back to this; its own
    environment.yml called that dependency optional and the reference evaluation ran without it,
    so this is the path that produced the published numbers. Keeping only the fallback drops a
    CUDA-built dependency and keeps the stage compilable -- `torch_scatter`'s scatter ops have no
    fake-tensor kernel and hard-crash Dynamo.
    """
    if out is None:
        size = list(src.shape)
        size[dim] = dim_size or int(index.max()) + 1
        out = src.new_zeros(size)
    return out.index_add_(dim, index, src)


@contextlib.contextmanager
def _efficient_attention():
    """Disable PyTorch's fused MultiheadAttention fastpath for the enclosed block.

    The BEV refinement transformer attends over *every* BEV cell, so its sequence length is the
    grid area: 16,384 tokens at 128x128, but 40,000 at the standard 200x200 window. PyTorch's
    "BetterTransformer" fastpath (`torch._transformer_encoder_layer_fwd`) materializes the full
    attention matrix, which at 40,000 tokens is tens of GB; the ordinary path routes through
    `scaled_dot_product_attention`, which picks a memory-efficient kernel instead.

    Measured on an RTX 5090, whole model, 448x798 input, batch 1, 200x200 grid:

        fp32, fastpath on    27.48 GB     <- OOMs a 32 GB card in a real eval loop
        fp32, fastpath off    6.28 GB
        bf16, either          5.94 GB

    So this is what makes the comparable window runnable at full precision at all. Under bf16 it
    costs nothing, because SDPA already picks a flash kernel there.

    The toggle is process-global, so it is scoped to the call and restored afterwards rather
    than set once at import.

    **It is not bit-identical.** SDPA reduces in a different order than the fused kernel, so
    enabling this changes the low bits of `bev_trans`'s output and everything downstream --
    `dev/capture_forward_reference.py` reports the divergence. The class default is therefore
    False (bit-exact with the release); the shipped config sets it True, because without it the
    comparable 200x200 window cannot be evaluated in fp32 on a 32 GB card at all. The measured
    effect on Vehicle IoU is recorded in docs/baseline.md.
    """
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def _position_encoding(height: int, width: int, channels: int, device):
    half = channels // 2
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).float()
    divisor = torch.exp(
        torch.arange(0, half, 2, device=device)
        * (-math.log(10000.0) / half)
    )
    encoding = []
    for axis in range(2):
        position = grid[..., axis].reshape(1, -1, 1)
        sin_cos = torch.stack(
            (torch.sin(position * divisor), torch.cos(position * divisor)),
            dim=3,
        ).flatten(2)
        encoding.append(sin_cos)
    encoding = torch.cat(encoding, dim=2)
    if encoding.shape[-1] < channels:
        encoding = F.pad(encoding, (0, channels - encoding.shape[-1]))
    return encoding[0]


class LiftSplatProjector(nn.Module):
    def __init__(
        self,
        *,
        img_height: int = 294,
        img_width: int = 518,
        grid: BevGrid | None = None,
        feature_channels: int = 128,
        depth_min: float = 1.0,
        depth_max: float = 50.0,
        use_transformer: bool = True,
        efficient_attention: bool = False,
        strict_freeze: bool = False,
        backbone_name: str = "lpiccinelli/unidepth-v2-vitl14",
    ):
        super().__init__()
        self.H = img_height
        self.W = img_width
        self.grid = grid if grid is not None else BevGrid()
        self.bev_h = self.grid.h
        self.bev_w = self.grid.w
        self.voxel_z = self.grid.voxel_z
        self.C = feature_channels

        # Kept as buffers holding exactly the values the released code used, so the pooling
        # arithmetic below is unchanged expression-for-expression. bx[0]/bx[1] are never read
        # (only the z origin is) but are carried to keep the tensor's shape and meaning intact.
        self.register_buffer(
            "dx",
            torch.tensor(
                (self.grid.res_h, self.grid.res_w, self.grid.dz)
            ).view(1, 1, 3),
        )
        self.register_buffer(
            "bx",
            torch.tensor(
                (-self.grid.half_h, -self.grid.half_w, self.grid.z_min)
            ).view(1, 1, 3),
        )
        self.depth_min = depth_min
        self.depth_max = depth_max

        from unidepth.models.unidepthv2 import UniDepthV2

        self.backbone = UniDepthV2.from_pretrained(backbone_name)
        self.strict_freeze = strict_freeze
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self._enforce_freeze()

        self.feat_adapter = nn.Sequential(
            nn.Linear(1024, feature_channels),
            nn.GELU(),
            nn.Linear(feature_channels, feature_channels),
        )

        self.use_transformer = use_transformer
        self.efficient_attention = efficient_attention
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=feature_channels,
                nhead=4,
                dropout=0.1,
                batch_first=True,
            )
            self.bev_trans = nn.TransformerEncoder(layer, num_layers=2)

        self._frustum = {}

    def _enforce_freeze(self):
        """Re-assert the backbone freeze that `UniDepthV2.eval()` silently undoes.

        `DinoVisionTransformer.train()` (unidepth/models/backbones/dinov2.py) ends by
        *unconditionally* reassigning `requires_grad` on `cls_token`, `pos_embed` and `norm`
        from its own `frozen_stages`/`use_norm` fields. It ignores any external
        `requires_grad_(False)`, and `.eval()` routes through `train(False)` -- so the freeze
        applied one line earlier in `__init__` is reverted, and reverted again on every
        subsequent `.train()` call.

        The effect is that 1,405,952 backbone parameters (`pos_embed` dominating, at 1,402,880)
        stay trainable. The committed `script/train.ipynb` builds its optimizer from
        `p for p in model.parameters() if p.requires_grad`, so running the released code as
        committed trains 2,813,633 parameters rather than the 1.41M the paper reports.

        The released *checkpoint*, though, was not produced that way: its optimizer state covers
        exactly 33 tensors / 1,407,681 elements -- feat_adapter + bev_trans + seg, the intended
        set -- and its `format` field reads "compact_without_frozen_backbone_v1". So the paper's
        figure is right for the actual run, and this is a gap between the published code and the
        published result rather than an error in the result.

        The class default is False, matching the released `Model/model.py` literally; the
        shipped config sets it True, matching the run behind the numbers. It makes no difference
        to a forward pass, so evaluating a released checkpoint is unaffected either way.
        """
        if self.strict_freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self._enforce_freeze()
        return self

    def _make_frustum(self, *, device, Hf: int, Wf: int):
        key = (device, Hf, Wf)
        if key not in self._frustum:
            scale_x = self.W / Wf
            scale_y = self.H / Hf
            x = (torch.arange(Wf, device=device) + 0.5) * scale_x
            y = (torch.arange(Hf, device=device) + 0.5) * scale_y
            x, y = torch.meshgrid(x, y, indexing="xy")
            self._frustum[key] = torch.stack(
                (x, y, torch.ones_like(x), torch.ones_like(x)), dim=-1
            ).reshape(-1, 4)
        return self._frustum[key]

    def lift_to_ego(self, depth, intrinsics, extrinsics, frustum, flat_batch, point_count):
        """Back-project one metric depth per feature cell, then map it into the ego frame.

        Extracted verbatim from the released `forward()`. This is the paper's substitution for
        LSS's depth-bin distribution: a single 3D point per feature cell, placed at the depth
        UniDepthV2 predicted.
        """
        flat_intrinsics = intrinsics.reshape(flat_batch, 3, 3)
        focal_x = flat_intrinsics[:, 0, 0]
        focal_y = flat_intrinsics[:, 1, 1]
        center_x = flat_intrinsics[:, 0, 2]
        center_y = flat_intrinsics[:, 1, 2]
        pixel_x = frustum[:, 0].expand(flat_batch, point_count)
        pixel_y = frustum[:, 1].expand(flat_batch, point_count)
        depth = depth.reshape(flat_batch, point_count)

        camera_xyz = torch.stack(
            (
                (pixel_x - center_x[:, None]) / focal_x[:, None] * depth,
                (pixel_y - center_y[:, None]) / focal_y[:, None] * depth,
                depth,
            ),
            dim=-1,
        )
        flat_extrinsics = extrinsics.reshape(flat_batch, 4, 4)
        ego_xyz = (
            torch.bmm(camera_xyz, flat_extrinsics[:, :3, :3].transpose(1, 2))
            + flat_extrinsics[:, :3, 3].unsqueeze(1)
        )
        return ego_xyz

    def _voxel_pool(self, xyz, features, batch_size: int):
        cameras_per_batch, channels, point_count = features.shape
        dx = self.dx.to(xyz.device, xyz.dtype)
        bx = self.bx.to(xyz.device, xyz.dtype)

        half_x = self.bev_h * dx[0, 0, 0] / 2
        half_y = self.bev_w * dx[0, 0, 1] / 2
        row = ((half_x - xyz[..., 0]) / dx[0, 0, 0]).round().long()
        col = ((half_y - xyz[..., 1]) / dx[0, 0, 1]).round().long()
        height = ((xyz[..., 2] - bx[0, 0, 2]) / dx[0, 0, 2]).round().long()

        valid = (
            (row >= 0)
            & (row < self.bev_h)
            & (col >= 0)
            & (col < self.bev_w)
            & (height >= 0)
            & (height < self.voxel_z)
        )
        if not valid.any():
            return features.new_zeros(
                batch_size,
                channels,
                self.voxel_z,
                self.bev_h,
                self.bev_w,
            )

        pooled_features = features.permute(0, 2, 1)[valid]
        row, col, height = row[valid], col[valid], height[valid]
        batch = (
            torch.arange(cameras_per_batch, device=features.device)
            .view(-1, 1)
            .expand(-1, point_count)[valid]
            // (cameras_per_batch // batch_size)
        )
        linear_index = (
            ((batch * self.voxel_z + height) * self.bev_h + row)
            * self.bev_w
            + col
        )
        output_size = (
            batch_size * self.voxel_z * self.bev_h * self.bev_w
        )
        volume = scatter_add(
            pooled_features,
            linear_index,
            dim=0,
            dim_size=output_size,
        )
        counts = scatter_add(
            torch.ones_like(pooled_features[:, 0]),
            linear_index,
            dim=0,
            dim_size=output_size,
        ).clamp_min(1).unsqueeze(-1)
        volume = (volume / counts).view(
            batch_size,
            self.voxel_z,
            self.bev_h,
            self.bev_w,
            channels,
        )
        return volume.permute(0, 4, 1, 2, 3)

    def forward(self, images, intrinsics, extrinsics):
        batch_size, camera_count, _, height, width = images.shape
        flat_batch = batch_size * camera_count
        rgb = images.reshape(flat_batch, 3, height, width)
        rgb = TF.normalize(rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        with torch.no_grad():
            encoder_inputs, encoder_outputs = self.backbone.encode_decode(
                {"image": rgb, "camera": None}, image_metas=[]
            )

        depth_full = encoder_outputs["depth"]
        encoder_features = encoder_inputs["features"][-1]
        feature_height, feature_width = encoder_features.shape[1:3]

        features = self.feat_adapter(encoder_features).permute(0, 3, 1, 2)
        depth = F.interpolate(
            depth_full,
            size=(feature_height, feature_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).clamp(self.depth_min, self.depth_max)

        point_count = feature_height * feature_width
        frustum = self._make_frustum(
            device=images.device, Hf=feature_height, Wf=feature_width
        )
        ego_xyz = self.lift_to_ego(
            depth, intrinsics, extrinsics, frustum, flat_batch, point_count
        )

        bev = self._voxel_pool(
            ego_xyz,
            features.reshape(flat_batch, self.C, point_count),
            batch_size,
        ).max(dim=2).values

        if self.use_transformer:
            bev_height, bev_width = bev.shape[2:]
            position = _position_encoding(
                bev_height,
                bev_width,
                self.C,
                bev.device,
            )
            tokens = bev.flatten(2).transpose(1, 2) + position
            with _efficient_attention() if self.efficient_attention else contextlib.nullcontext():
                refined = self.bev_trans(tokens)
            bev = refined.transpose(1, 2).reshape(
                batch_size,
                self.C,
                bev_height,
                bev_width,
            )
        return bev
