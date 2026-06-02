"""Spatial saliency from the deep foul detector ("where the net looks").

For an action the deep net flags as a foul, this occludes a grid of spatial
regions in the live view and measures how much each occlusion lowers the net's
"foulness" score. The region whose removal hurts the foul score most is where
the model is looking -- the foul location. The saliency is computed in the
model's cropped input space and mapped back to the original broadcast frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropGeometry:
    """Maps the model's square crop back to original-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def uv_to_point(self, u: float, v: float) -> tuple[float, float]:
        return (self.x1 + u * (self.x2 - self.x1), self.y1 + v * (self.y2 - self.y1))

    def as_box(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


def crop_geometry(
    pre_model: str,
    orig_h: int,
    orig_w: int,
    resize_shorter: int = 256,
    crop_size: int = 224,
) -> CropGeometry:
    """Rectangle in original-frame coords that the model's input crop covers.

    Torchvision video backbones (e.g. mvit_v2_s) resize the shorter side to
    ``resize_shorter`` then center-crop ``crop_size``.
    """
    scale = resize_shorter / float(min(orig_h, orig_w))
    resized_w = orig_w * scale
    resized_h = orig_h * scale
    crop_x0 = (resized_w - crop_size) / 2.0
    crop_y0 = (resized_h - crop_size) / 2.0
    return CropGeometry(
        x1=crop_x0 / scale,
        y1=crop_y0 / scale,
        x2=(crop_x0 + crop_size) / scale,
        y2=(crop_y0 + crop_size) / scale,
    )


def compute_occlusion_saliency(
    model,
    mvclips,
    view: int = 0,
    grid: int = 6,
    occ_value: float = 0.0,
    chunk: int = 12,
) -> np.ndarray:
    """Return a ``grid x grid`` foul-score drop map for the chosen view.

    ``mvclips`` is the batched model input (batch size 1). Higher values mark
    regions whose occlusion lowers the net's foulness most.
    """
    import torch

    device = mvclips.device
    height, width = mvclips.shape[-2], mvclips.shape[-1]
    cell_h = height / grid
    cell_w = width / grid

    with torch.no_grad():
        base = _foul_score(model, mvclips)

    cells = [(gy, gx) for gy in range(grid) for gx in range(grid)]
    drops = np.zeros(len(cells), dtype=np.float32)

    for start in range(0, len(cells), chunk):
        batch_cells = cells[start : start + chunk]
        n = len(batch_cells)
        masked = mvclips.repeat(n, *([1] * (mvclips.dim() - 1))).clone()
        for k, (gy, gx) in enumerate(batch_cells):
            y0, y1 = int(round(gy * cell_h)), int(round((gy + 1) * cell_h))
            x0, x1 = int(round(gx * cell_w)), int(round((gx + 1) * cell_w))
            masked[k, view, ..., y0:y1, x0:x1] = occ_value
        with torch.no_grad():
            scores = _foul_score(model, masked, reduce=False)
        drops[start : start + n] = (base - scores).detach().cpu().numpy()

    return drops.reshape(grid, grid)


def _foul_score(model, mvclips, reduce: bool = True):
    import torch

    offence_logits, _, _ = model(mvclips, None)
    probs = torch.softmax(offence_logits, dim=-1)
    foulness = 1.0 - probs[:, 0]
    if reduce:
        return float(foulness[0].item())
    return foulness


def peak_uv(saliency: np.ndarray) -> tuple[float, float]:
    """Center (u, v) in [0, 1] of the strongest saliency cell."""
    grid_h, grid_w = saliency.shape
    flat = int(np.argmax(saliency))
    gy, gx = divmod(flat, grid_w)
    return ((gx + 0.5) / grid_w, (gy + 0.5) / grid_h)


def weighted_centroid_uv(saliency: np.ndarray, power: float = 3.0) -> tuple[float, float]:
    """Intensity-weighted center (u, v) in [0, 1] of the saliency.

    More stable than :func:`peak_uv` when there are several hotspots. ``power``
    sharpens the weighting toward the strongest cells; falls back to the peak if
    the map is empty.
    """
    grid_h, grid_w = saliency.shape
    weights = np.clip(saliency.astype(np.float64), 0.0, None)
    if weights.max() <= 1e-9:
        return peak_uv(saliency)
    weights = (weights / weights.max()) ** power
    total = weights.sum()
    if total <= 1e-9:
        return peak_uv(saliency)
    ys, xs = np.mgrid[0:grid_h, 0:grid_w]
    u = float((weights * (xs + 0.5)).sum() / total) / grid_w
    v = float((weights * (ys + 0.5)).sum() / total) / grid_h
    return (u, v)
