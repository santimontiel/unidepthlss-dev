"""The BEV window: one definition, consumed by everything that touches the grid.

In the released code this was defined twice, independently -- `bev_size`/`bev_res` in the
dataset (which rasterizes the labels) and `bev_h`/`bev_w`/`dx`/`bx` in `LiftSplatProjector`
(which pools the features) -- with nothing coupling them. The training notebook happened to pass
matching values, so the two agreed by convention rather than by construction. `BevGrid` makes
that structural: the label rasterizer and the voxel pooler read the same object, so they cannot
drift.

**Why this matters beyond tidiness.** The released configuration is 128 x 128 cells at 0.5 m,
i.e. a 64 x 64 m window (+/-32 m). Every camera-only baseline UniDepth-LSS is compared against
in the paper's Table 1 reports the standard CVT/LSS "Setting 1" window: 200 x 200 at 0.5 m, i.e.
100 x 100 m. The smaller window excludes the 32-50 m ring -- where camera-only depth degrades
most and vehicles are small, distant and often occluded -- from both the true positives and the
false negatives, which inflates IoU. `configs/data/bev/` ships both windows for exactly this
reason; see `paper.yaml` vs `standard.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BevGrid:
    """A rectangular BEV window plus the height range collapsed into it.

    Defaults reproduce the released model exactly (128 x 128 over 64 x 64 m, z in [-4, 4) in
    eight 1 m bins) so that constructing a model with no explicit grid behaves as the published
    code did. The shipped Hydra config always passes an explicit grid, and its default is
    `standard`, not these values -- see the module docstring.
    """

    h: int = 128
    """Rows, along longitudinal (forward) ego x."""

    w: int = 128
    """Columns, along lateral (left) ego y."""

    h_meters: float = 64.0
    w_meters: float = 64.0

    z_min: float = -4.0
    z_max: float = 4.0
    voxel_z: int = 8

    @property
    def res_h(self) -> float:
        """Metres per cell along x."""
        return self.h_meters / self.h

    @property
    def res_w(self) -> float:
        """Metres per cell along y."""
        return self.w_meters / self.w

    @property
    def half_h(self) -> float:
        return self.h_meters / 2.0

    @property
    def half_w(self) -> float:
        return self.w_meters / 2.0

    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / self.voxel_z

    def __post_init__(self) -> None:
        if self.h <= 0 or self.w <= 0 or self.voxel_z <= 0:
            raise ValueError(f"BevGrid needs positive cell counts, got {self}")
        if self.h_meters <= 0 or self.w_meters <= 0:
            raise ValueError(f"BevGrid needs a positive extent, got {self}")
        if self.z_max <= self.z_min:
            raise ValueError(f"BevGrid needs z_max > z_min, got {self}")

    def describe(self) -> str:
        return (
            f"{self.h}x{self.w} cells over {self.h_meters:g}x{self.w_meters:g} m "
            f"(+/-{self.half_h:g} m, {self.res_h:g} m/cell), "
            f"z [{self.z_min:g}, {self.z_max:g}) in {self.voxel_z} bins"
        )


def required_depth_max(grid: BevGrid) -> float:
    """The smallest depth clamp that can still reach the far edge of `grid`.

    Depth is measured along the camera ray, and the lifting stage clamps it before
    back-projection, so a clamp below this leaves the outer part of the window structurally
    unreachable -- it would read as empty no matter what the backbone predicted. Widening the
    grid without raising the clamp is therefore a silent failure, which is why
    `configs/data/bev/*.yaml` carries `depth_max` alongside the window rather than in the model
    config.
    """
    return max(grid.half_h, grid.half_w)
