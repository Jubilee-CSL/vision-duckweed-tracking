from dataclasses import dataclass

import cv2
import numpy as np
from sacred import Ingredient

float_detection = Ingredient("float_detection")


@dataclass
class FloatCircle:
    points: np.ndarray  # 4 cardinal image points for PnP
    center_px: tuple  # (x, y) pixel centre
    radius_px: float  # enclosing circle radius in pixels
    contour: np.ndarray  # contour of detected float in image space


@float_detection.config
def config():
    threshold_blue = 150
    min_area_px = 250
    min_circularity = 0.7
    debug_float=True




@float_detection.capture
def get_float_points(img, threshold_blue, min_area_px, min_circularity, debug_float) -> FloatCircle:
    """ExB segmentation — returns FloatCircle with cardinal PnP points, centre and radius."""
    r, g, b = cv2.split(img)
    exb = 2 * b.astype(np.int16) - g.astype(np.int16) - r.astype(np.int16)
    exb = cv2.normalize(exb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(exb, threshold_blue, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    rejected = []  # (contour, reason) — only used for debug
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            rejected.append((cnt, f"area {area:.0f}"))
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            rejected.append((cnt, "perim 0"))
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity >= min_circularity:
            valid_contours.append(cnt)
        else:
            rejected.append((cnt, f"circ {circularity:.2f}"))

    biggest = max(valid_contours, key=cv2.contourArea) if valid_contours else None

    if debug_float:
        _plot_float_debug(img, exb, mask, valid_contours, rejected, biggest, threshold_blue)

    if biggest is None:
        raise ValueError("No float detected — adjust float_detection config.")

    (x, y), radius = cv2.minEnclosingCircle(biggest)
    points = np.array(
        [
            [x - radius, y],
            [x, y - radius],
            [x + radius, y],
            [x, y + radius],
        ],
        dtype=np.float32,
    )
    return FloatCircle(
        points=points,
        center_px=(int(x), int(y)),
        radius_px=float(radius),
        contour=biggest,
    )


def _plot_float_debug(img, exb, mask, valid_contours, rejected, biggest, threshold_blue):
    """Show ExB channel, binary mask and contour overlay (red=rejected, yellow=valid, green=selected)."""
    import matplotlib.pyplot as plt

    overlay = img.copy()
    for cnt, reason in rejected:
        cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 1)
        bx, by, _, _ = cv2.boundingRect(cnt)
        cv2.putText(overlay, reason, (bx, max(by - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.drawContours(overlay, valid_contours, -1, (255, 255, 0), 2)
    if biggest is not None:
        cv2.drawContours(overlay, [biggest], -1, (0, 255, 0), 2)
        (x, y), radius = cv2.minEnclosingCircle(biggest)
        cv2.circle(overlay, (int(x), int(y)), int(radius), (0, 255, 255), 1)
        cv2.drawMarker(overlay, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        for px, py in [(x - radius, y), (x, y - radius), (x + radius, y), (x, y + radius)]:
            cv2.circle(overlay, (int(px), int(py)), 4, (255, 0, 255), -1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(exb, cmap="gray")
    axes[0].set_title("ExB (normalised)")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Mask (threshold={threshold_blue})")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Contours: {len(valid_contours)} valid / {len(rejected)} rejected"
        + ("" if biggest is not None else " — NONE SELECTED")
    )
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
