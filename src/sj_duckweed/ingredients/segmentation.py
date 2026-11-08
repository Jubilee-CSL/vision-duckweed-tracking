import importlib
import importlib.util
import logging

import cv2
import numpy as np
from sacred import Ingredient

segmentation = Ingredient("segmentation")

logger = logging.getLogger(__name__)

__all__ = [
    "segmentation",
    "get_img_contour",
    "get_img_contour_cellpose",
    "is_cellpose_available",
]

# Debug overlay colours (BGR, drawn on the BGR image before conversion for display)
_DEBUG_COLOURS = {
    "kept": (0, 255, 0),          # green
    "too small": (255, 128, 0),   # blue
    "too large": (255, 0, 255),   # magenta
    "low circularity": (0, 0, 255),  # red
}


@segmentation.config
def config():
    min_area_px = 10
    max_area_px = 800
    min_circularity = 0.2
    threshold_green = 180
    cellpose_diameter = 8
    debug_segmentation = True


def _plot_debug(img, panels, kept, rejected, title):
    """Show intermediate images plus a contour overlay coloured by filter outcome.

    panels: list of (title, image, cmap) shown before the overlay.
    rejected: list of (contour, reason) where reason is a key of _DEBUG_COLOURS.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    overlay = img.copy()
    counts = {k: 0 for k in _DEBUG_COLOURS}
    for cnt, reason in rejected:
        cv2.drawContours(overlay, [cnt], -1, _DEBUG_COLOURS[reason], 1)
        counts[reason] += 1
    cv2.drawContours(overlay, kept, -1, _DEBUG_COLOURS["kept"], 1)
    counts["kept"] = len(kept)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(6 * (len(panels) + 1), 6),
                             sharex=True, sharey=True)
    for ax, (panel_title, panel_img, cmap) in zip(axes, panels):
        ax.imshow(panel_img, cmap=cmap)
        ax.set_title(panel_title)
    axes[-1].imshow(overlay)
    axes[-1].set_title("Contours")
    axes[-1].legend(
        handles=[
            Patch(color=np.array(c[::-1]) / 255, label=f"{name} ({counts[name]})")
            for name, c in _DEBUG_COLOURS.items()
        ],
        loc="upper right",
        fontsize=8,
    )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


@segmentation.capture
def get_img_contour(img, min_area_px, max_area_px, min_circularity, threshold_green, debug_segmentation):
    """ExG segmentation — returns contours matching area and circularity filters."""
    b, g, r = cv2.split(img)
    exg = 2 * g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16)
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(exg, threshold_green, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    rejected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            rejected.append((cnt, "too small"))
            continue
        if area > max_area_px:
            rejected.append((cnt, "too large"))
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            rejected.append((cnt, "too small"))
            continue
        if 4 * np.pi * area / (perimeter**2) >= min_circularity:
            valid.append(cnt)
        else:
            rejected.append((cnt, "low circularity"))

    logger.debug(
        "ExG: %d contours, %d kept, %d rejected", len(contours), len(valid), len(rejected)
    )
    if debug_segmentation:
        _plot_debug(
            img,
            panels=[
                ("ExG (normalised)", exg, "gray"),
                (f"Mask (threshold={threshold_green})", mask, "gray"),
            ],
            kept=valid,
            rejected=rejected,
            title=(
                f"ExG segmentation — area [{min_area_px}, {max_area_px}] px, "
                f"circularity ≥ {min_circularity}"
            ),
        )
    return valid


def get_img_contour_cellpose(
    img,
    diameter=8,
    min_area_px=10,
    max_area_px=300,
    min_circularity=0.5,
    debug_segmentation=False,
):
    """Cellpose segmentation — returns area-filtered contours.

    If Cellpose is unavailable or inference fails, returns an empty list.
    """
    try:
        models = importlib.import_module("cellpose.models")
    except Exception:
        logger.warning("Cellpose is not installed; AI segmentation disabled.")
        return []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        try:
            use_gpu = bool(models.use_gpu())
        except Exception:
            use_gpu = False

        model = models.CellposeModel(gpu=use_gpu, model_type="cyto2")
        masks, flows, styles = model.eval(img_rgb, diameter=diameter, channels=[2, 0])
    except Exception as exc:
        logger.warning("Cellpose inference failed: %s", exc)
        return []

    valid_contours = []
    rejected = []
    num_cells = masks.max()
    for i in range(1, num_cells + 1):
        cell_mask = np.uint8(masks == i) * 255
        contours, _ = cv2.findContours(
            cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                rejected.append((cnt, "too small"))
                continue
            if area > max_area_px:
                rejected.append((cnt, "too large"))
                continue
            valid_contours.append(cnt)

    logger.debug(
        "Cellpose: %d cells, %d kept, %d rejected",
        num_cells, len(valid_contours), len(rejected),
    )
    if debug_segmentation:
        _plot_debug(
            img,
            panels=[
                ("Input", img_rgb, None),
                (f"Cellpose masks ({num_cells} cells)", masks, "nipy_spectral"),
            ],
            kept=valid_contours,
            rejected=rejected,
            title=f"Cellpose segmentation — diameter={diameter}, area [{min_area_px}, {max_area_px}] px",
        )
    return valid_contours


def is_cellpose_available():
    """Return True when Cellpose can be imported in the current environment."""
    return importlib.util.find_spec("cellpose") is not None