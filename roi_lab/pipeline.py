"""Crown halves ROI clone.

Orange boxes, blue finish line, pink edge, yellow dots, and the percent.
Copied from the crown path that runs on the machine:

- operations/crown_round/vision/halves_roi.py  (quad from star + girdle)
- hardware/symmetry.py  (box thickness from the girdle chord)
- operations/crown_round/vision/halves_progress.py  (warp, clip, draw)
- operations/crown_round/vision/halves_progress_math.py  (percent)

The photo does not contain the 8 star points or the 16 girdle points.
On the machine those come from the 3D symmetry draw. This folder takes them
as input, then runs the same box and line steps.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from line_detect import roi_line_detection_pipeline

# Crown hardware defaults (mes/operation_hardware.py crown block).
OFFSET_PX = 15.0
ENDPOINT_SHIFT_PERCENT = 30.0
MIN_INSET_PERCENT = 7.0
INCREASE_PERCENTAGE = 99.9
PERP_MIN_PX = 20.0
PERP_CHORD_SCALE = 0.30
PERP_CAP_VS_SEGMENT = 0.55
PERP_EXTRA_PX = 0.0
LINE_THICKNESS = 2
FONT_SCALE = 0.5 * max(1, LINE_THICKNESS)
TEXT_OFFSET = max(8, 8 * LINE_THICKNESS)

Point = Tuple[float, float]
Quad = Tuple[Point, Point, Point, Point]


def _odd_girdle_index(roi_idx: int) -> int:
    if roi_idx % 2 == 0:
        return 1 + 2 * (roi_idx // 2)
    girdle_idx = (3 + 2 * (roi_idx // 2)) % 16
    if girdle_idx % 2 == 0:
        girdle_idx = (girdle_idx + 1) % 16
    return girdle_idx


def star_girdle_lengths(star_points, girdle_points) -> List[Optional[float]]:
    lengths: List[Optional[float]] = []
    for roi_idx in range(16):
        star_idx = roi_idx // 2
        girdle_idx = _odd_girdle_index(roi_idx)
        if star_idx >= len(star_points) or girdle_idx >= len(girdle_points):
            lengths.append(None)
            continue
        star_pt = star_points[star_idx]
        girdle_pt = girdle_points[girdle_idx]
        if star_pt is None or girdle_pt is None or len(star_pt) < 2 or len(girdle_pt) < 2:
            lengths.append(None)
            continue
        length = math.hypot(float(girdle_pt[0]) - float(star_pt[0]), float(girdle_pt[1]) - float(star_pt[1]))
        lengths.append(length if length >= 1e-6 else None)
    return lengths


def perpendicular_height_from_girdle(girdle_points, girdle_idx: int, star_girdle_length_px: float) -> float:
    """Box thickness in pixels. Same formula as symmetry._crown_half_perpendicular_height_from_girdle_px."""
    n = len(girdle_points) if girdle_points else 0
    fallback = float(max(PERP_MIN_PX, 0.22 * max(star_girdle_length_px, PERP_MIN_PX)) + PERP_EXTRA_PX)
    if n < 3:
        return fallback

    def xy(i):
        p = girdle_points[int(i) % n]
        if p is None or len(p) < 2:
            return None
        return float(p[0]), float(p[1])

    im = int(girdle_idx) % n
    pa, pb, pc = xy(im - 1), xy(im), xy(im + 1)
    if pa is None or pb is None or pc is None:
        return fallback
    chord_skip = math.hypot(pc[0] - pa[0], pc[1] - pa[1])
    e1 = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
    e2 = math.hypot(pc[0] - pb[0], pc[1] - pb[1])
    chord_local = max(chord_skip, 0.5 * (e1 + e2))
    height = max(PERP_MIN_PX, PERP_CHORD_SCALE * chord_local)
    if star_girdle_length_px > 1e-6:
        height = min(height, PERP_CAP_VS_SEGMENT * star_girdle_length_px)
    height = height + PERP_EXTRA_PX
    return float(max(PERP_MIN_PX + PERP_EXTRA_PX, height))


def map_quads(star_points, girdle_points, shift_per_roi: Sequence[float]) -> List[Optional[Quad]]:
    """16 orange-box quads. Same steps as map_crown_halves_quad_rois with height 'girdle'."""
    result: List[Optional[Quad]] = []
    for roi_idx in range(16):
        star_idx = roi_idx // 2
        girdle_idx = _odd_girdle_index(roi_idx)
        if star_idx >= len(star_points) or girdle_idx >= len(girdle_points):
            result.append(None)
            continue
        star_pt = star_points[star_idx]
        girdle_pt = girdle_points[girdle_idx]
        if star_pt is None or girdle_pt is None or len(star_pt) < 2 or len(girdle_pt) < 2:
            result.append(None)
            continue
        sx, sy = float(star_pt[0]), float(star_pt[1])
        gx, gy = float(girdle_pt[0]), float(girdle_pt[1])
        dx, dy = gx - sx, gy - sy
        length = math.hypot(dx, dy)
        original_length = length
        if length < 1e-6:
            result.append(None)
            continue
        shift_px = float(shift_per_roi[roi_idx]) if roi_idx < len(shift_per_roi) else 0.0
        shift_px = max(0.0, min(shift_px, length * 0.5 - 1e-6))
        if shift_px > 1e-6:
            ux, uy = dx / length, dy / length
            sx += shift_px * ux
            sy += shift_px * uy
            gx -= shift_px * ux
            gy -= shift_px * uy
            dx, dy = gx - sx, gy - sy
            length = math.hypot(dx, dy)
            if length < 1e-6:
                result.append(None)
                continue
        perp_x, perp_y = -dy / length, dx / length
        star_left = (int(round(sx - OFFSET_PX * perp_x)), int(round(sy - OFFSET_PX * perp_y)))
        star_right = (int(round(sx + OFFSET_PX * perp_x)), int(round(sy + OFFSET_PX * perp_y)))
        girdle_left = (int(round(gx - OFFSET_PX * perp_x)), int(round(gy - OFFSET_PX * perp_y)))
        girdle_right = (int(round(gx + OFFSET_PX * perp_x)), int(round(gy + OFFSET_PX * perp_y)))

        is_odd = (roi_idx + 1) % 2 == 1
        if is_odd:
            e1, e2 = girdle_left, star_left
            e3, e4 = star_right, girdle_right
        else:
            e1, e2 = star_right, girdle_right
            e3, e4 = star_left, girdle_left
        ex, ey = e4[0] - e3[0], e4[1] - e3[1]
        side = math.hypot(ex, ey)
        if side >= 1e-6:
            dpx, dpy = ex / side, ey / side
            out_x = (e1[0] + e2[0]) / 2.0 - (e3[0] + e4[0]) / 2.0
            out_y = (e1[1] + e2[1]) / 2.0 - (e3[1] + e4[1]) / 2.0
            height = math.hypot(out_x, out_y)
            if height >= 1e-6:
                nx, ny = out_x / height, out_y / height
                new_height = perpendicular_height_from_girdle(girdle_points, girdle_idx, original_length)
                t1 = (e1[0] - e3[0]) * dpx + (e1[1] - e3[1]) * dpy
                t2 = (e2[0] - e3[0]) * dpx + (e2[1] - e3[1]) * dpy
                new_e1 = (e3[0] + t1 * dpx + nx * new_height, e3[1] + t1 * dpy + ny * new_height)
                new_e2 = (e3[0] + t2 * dpx + nx * new_height, e3[1] + t2 * dpy + ny * new_height)
                if is_odd:
                    girdle_left = (int(round(new_e1[0])), int(round(new_e1[1])))
                    star_left = (int(round(new_e2[0])), int(round(new_e2[1])))
                else:
                    star_right = (int(round(new_e1[0])), int(round(new_e1[1])))
                    girdle_right = (int(round(new_e2[0])), int(round(new_e2[1])))
        result.append((star_left, star_right, girdle_left, girdle_right))
    return result


def clip_line_to_rect(mx, my, dx, dy, w_f, h_f):
    eps = 1e-10
    hits = []
    if abs(dx) >= eps:
        for x_edge, t in ((0.0, -mx / dx), (w_f, (w_f - mx) / dx)):
            y = my + t * dy
            if 0 <= y <= h_f:
                hits.append((t, x_edge, y))
    if abs(dy) >= eps:
        for t in (-my / dy, (h_f - my) / dy):
            x = mx + t * dx
            y = my + t * dy
            if 0 <= x <= w_f and 0 <= y <= h_f:
                hits.append((t, x, y))
    if len(hits) < 2:
        return None, None
    hits.sort(key=lambda item: item[0])
    t_lo, t_hi = hits[0][0], hits[-1][0]
    return (mx + t_lo * dx, my + t_lo * dy), (mx + t_hi * dx, my + t_hi * dy)


def progress_percent(mid_outward, finish_center, mid_detected) -> Tuple[float, float]:
    ax, ay = float(finish_center[0]), float(finish_center[1])
    mx, my = float(mid_outward[0]), float(mid_outward[1])
    dx, dy = float(mid_detected[0]), float(mid_detected[1])
    center_dist = math.hypot(dx - ax, dy - ay)
    total = math.hypot(ax - mx, ay - my)
    if total < 1e-10:
        return 0.0, center_dist
    unit_dx = (ax - mx) / total
    unit_dy = (ay - my) / total
    s = (dx - mx) * unit_dx + (dy - my) * unit_dy
    t = s / (total + 1e-12)
    return max(0.0, min(120.0, 100.0 * t)), center_dist


def _as_points(raw, count: int, name: str):
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError(f"Need {count} {name} points, got {0 if not isinstance(raw, list) else len(raw)}.")
    points = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(f"Each {name} point must be [x, y].")
        points.append((float(item[0]), float(item[1])))
    return points


def _shift_pixels(lengths, previous: Optional[Dict[int, float]]) -> List[float]:
    shifts = []
    for roi_no in range(1, 17):
        length = lengths[roi_no - 1] if lengths[roi_no - 1] is not None else 100.0
        reduce_px = length * (ENDPOINT_SHIFT_PERCENT / 100.0)
        min_inset_px = length * (MIN_INSET_PERCENT / 100.0)
        pct = 0.0 if not previous else float(previous.get(roi_no, previous.get(str(roi_no), 0.0)) or 0.0)
        if previous is None or all(float(previous.get(i, previous.get(str(i), 0.0)) or 0.0) == 0 for i in range(1, 17)):
            shifts.append(reduce_px)
        else:
            shifts.append(reduce_px if pct < INCREASE_PERCENTAGE else min_inset_px)
    return shifts


def draw_pink_on_rect(image_bgr: np.ndarray):
    """Draw the pink edge on one already-straightened ROI rectangle.

    The input is the gray warp lookup saves. No star or girdle points.
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        raise ValueError("Image is empty.")
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    h, w = image_bgr.shape[:2]
    line = _detect_line(image_bgr)
    overlay = image_bgr.copy()
    if line is None:
        return overlay, None
    p1, p2 = line
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    mx = (float(p1[0]) + float(p2[0])) / 2.0
    my = (float(p1[1]) + float(p2[1])) / 2.0
    left_f, right_f = clip_line_to_rect(mx, my, dx, dy, float(w), float(h))
    if left_f is None or right_f is None:
        a = (int(round(p1[0])), int(round(p1[1])))
        b = (int(round(p2[0])), int(round(p2[1])))
    else:
        a = (int(round(left_f[0])), int(round(left_f[1])))
        b = (int(round(right_f[0])), int(round(right_f[1])))
    cv2.line(overlay, a, b, (255, 0, 255), LINE_THICKNESS, cv2.LINE_AA)
    return overlay, [list(a), list(b)]


def _detect_line(roi_content):
    line = roi_line_detection_pipeline(
        roi_content, return_steps=False, canny_low=40, canny_high=200, hough_threshold=70, min_line_length=20, max_line_gap=10
    )
    if line is None:
        line = roi_line_detection_pipeline(
            roi_content, return_steps=False, canny_low=20, canny_high=150, hough_threshold=20, min_line_length=10, max_line_gap=18
        )
    if line is None:
        line = roi_line_detection_pipeline(
            roi_content, return_steps=False, canny_low=15, canny_high=180, hough_threshold=15, min_line_length=8, max_line_gap=25, use_contour_mask=False
        )
    return line


def run_crown_roi(image_bgr: np.ndarray, star_points, girdle_points, previous_progress: Optional[dict] = None):
    """Draw the current crown ROI overlay. Returns (image, progress by half 1..16)."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        raise ValueError("Image is empty.")
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    star_points = _as_points(list(star_points), 8, "star")
    girdle_points = _as_points(list(girdle_points), 16, "girdle")

    lengths = star_girdle_lengths(star_points, girdle_points)
    shifts = _shift_pixels(lengths, previous_progress)
    mapped = map_quads(star_points, girdle_points, shifts)

    overlay = image_bgr.copy()
    for roi_idx, quad in enumerate(mapped):
        if quad is None:
            continue
        star_left, star_right, girdle_left, girdle_right = quad
        pts = np.array([star_left, star_right, girdle_right, girdle_left], dtype=np.int32)
        cv2.polylines(overlay, [pts], True, (0, 165, 255), 1, cv2.LINE_AA)
        if (roi_idx + 1) % 2 == 1:
            e1, e2 = girdle_left, star_left
            e3, e4 = star_right, girdle_right
        else:
            e1, e2 = star_right, girdle_right
            e3, e4 = star_left, girdle_left
        for label, pt in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
            cv2.putText(overlay, label, (int(pt[0]) + 2, int(pt[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cx = int((star_left[0] + star_right[0] + girdle_left[0] + girdle_right[0]) / 4)
        cy = int((star_left[1] + star_right[1] + girdle_left[1] + girdle_right[1]) / 4)
        cv2.putText(overlay, str(roi_idx + 1), (cx - 4, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    n_sp = len(star_points)
    stone_center = (
        sum(p[0] for p in star_points) / n_sp,
        sum(p[1] for p in star_points) / n_sp,
    )

    def vertex_angle(p):
        return math.atan2(p[1] - stone_center[1], p[0] - stone_center[0])

    progress = {i: 0.0 for i in range(1, 17)}
    dots = []
    for roi_idx, quad in enumerate(mapped):
        if quad is None:
            continue
        roi_num = roi_idx + 1
        star_left, star_right, girdle_left, girdle_right = quad
        star_idx = roi_idx // 2
        girdle_idx = _odd_girdle_index(roi_idx)
        star_pt = star_points[star_idx]
        girdle_pt = girdle_points[girdle_idx]
        cv2.line(
            overlay,
            (int(round(star_pt[0])), int(round(star_pt[1]))),
            (int(round(girdle_pt[0])), int(round(girdle_pt[1]))),
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        vertices = [star_left, star_right, girdle_right, girdle_left]
        sorted_by_dist = sorted(vertices, key=lambda p: math.hypot(p[0] - stone_center[0], p[1] - stone_center[1]))
        inner_left, inner_right = sorted(sorted_by_dist[:2], key=vertex_angle)
        outer_left, outer_right = sorted(sorted_by_dist[2:], key=vertex_angle)
        mid_outward = ((outer_left[0] + outer_right[0]) / 2.0, (outer_left[1] + outer_right[1]) / 2.0)
        src = np.array([inner_left, inner_right, outer_right, outer_left], dtype=np.float32)
        w = max(
            1,
            int(round(max(
                math.hypot(inner_right[0] - inner_left[0], inner_right[1] - inner_left[1]),
                math.hypot(outer_right[0] - outer_left[0], outer_right[1] - outer_left[1]),
            ))),
        )
        h = max(
            1,
            int(round(max(
                math.hypot(outer_right[0] - inner_right[0], outer_right[1] - inner_right[1]),
                math.hypot(outer_left[0] - inner_left[0], outer_left[1] - inner_left[1]),
            ))),
        )
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        roi_content = cv2.warpPerspective(image_bgr, matrix, (w, h))
        line = _detect_line(roi_content)
        if line is None:
            cv2.putText(overlay, "—", (int((star_left[0] + star_right[0] + girdle_left[0] + girdle_right[0]) / 4) + TEXT_OFFSET,
                                       int((star_left[1] + star_right[1] + girdle_left[1] + girdle_right[1]) / 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, (0, 165, 255), max(1, LINE_THICKNESS))
            continue
        p1, p2 = line
        inv = np.linalg.inv(matrix)
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        mx = (float(p1[0]) + float(p2[0])) / 2.0
        my = (float(p1[1]) + float(p2[1])) / 2.0
        left_f, right_f = clip_line_to_rect(mx, my, dx, dy, float(w), float(h))
        if left_f is not None and right_f is not None:
            roi_pts = np.array([[left_f[0], left_f[1]], [right_f[0], right_f[1]]], dtype=np.float32).reshape(2, 1, 2)
        else:
            roi_pts = np.array([[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]], dtype=np.float32).reshape(2, 1, 2)
        img_pts = cv2.perspectiveTransform(roi_pts, inv)
        a = (int(round(img_pts[0, 0, 0])), int(round(img_pts[0, 0, 1])))
        b = (int(round(img_pts[1, 0, 0])), int(round(img_pts[1, 0, 1])))
        cv2.line(overlay, a, b, (255, 0, 255), LINE_THICKNESS, cv2.LINE_AA)
        finish = ((float(star_pt[0]) + float(girdle_pt[0])) / 2.0, (float(star_pt[1]) + float(girdle_pt[1])) / 2.0)
        mid_detected = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        stored, _dist = progress_percent(mid_outward, finish, mid_detected)
        progress[roi_num] = float(stored)
        dots.append((finish, mid_detected))
        cx = int((star_left[0] + star_right[0] + girdle_left[0] + girdle_right[0]) / 4)
        cy = int((star_left[1] + star_right[1] + girdle_left[1] + girdle_right[1]) / 4)
        if stored < 100:
            cv2.putText(
                overlay, f"{stored:.1f}%", (cx + TEXT_OFFSET, cy),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, (0, 0, 255), max(1, LINE_THICKNESS),
            )
    for finish, detected in dots:
        for px, py in (finish, detected):
            center = (int(round(px)), int(round(py)))
            cv2.circle(overlay, center, 4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(overlay, center, 3, (0, 255, 255), -1, cv2.LINE_AA)
    return overlay, progress


def ring_points(cx, cy, girdle_radius, star_ratio):
    """8 star points and 16 girdle points on one circle. star_ratio is star radius / girdle radius."""
    ratio = max(0.2, min(0.95, float(star_ratio)))
    radius = float(girdle_radius)
    star, girdle = [], []
    for i in range(16):
        angle = math.radians(i * 22.5 - 90.0)
        girdle.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    for i in range(8):
        angle = math.radians((2 * i + 1) * 22.5 - 90.0)
        star.append((cx + radius * ratio * math.cos(angle), cy + radius * ratio * math.sin(angle)))
    return star, girdle


def suggest_ring(image_bgr, star_ratio=0.63):
    """Find the stone on the uploaded photo and build star/girdle points around it.

    A symmetry picture is detected from the orange overlay. A plain stone picture
    is detected from the bright pixels. The girdle radius is the outer edge of that mask.
    """
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("Need a color image.")
    blue, green, red = cv2.split(image_bgr)
    orange = (
        (red > 150)
        & (green > 70)
        & (green < 220)
        & (blue < 100)
        & (red.astype(np.int16) > green.astype(np.int16) + 25)
    )
    ys, xs = np.where(orange)
    source = "orange symmetry circle"
    if xs.size < 300:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray > 45)
        source = "bright stone"
    if xs.size < 300:
        raise ValueError("The stone was not found. Click the center of the stone, then click the girdle.")
    cx = float(np.median(xs))
    cy = float(np.median(ys))
    distance = np.hypot(xs.astype(np.float64) - cx, ys.astype(np.float64) - cy)
    radius = float(np.percentile(distance, 94))
    short_side = float(min(image_bgr.shape[0], image_bgr.shape[1]))
    if radius < 0.08 * short_side:
        raise ValueError("The detected stone is too small. Click the center, then the girdle.")
    star, girdle = ring_points(cx, cy, radius, star_ratio)
    return star, girdle, {
        "center": [round(cx, 1), round(cy, 1)],
        "girdle_radius_px": round(radius, 1),
        "source": source,
    }


def demo_frame():
    """A round test stone plus the 8 star and 16 girdle points that match it."""
    size = 640
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    cv2.circle(image, (center, center), 250, (30, 30, 30), -1)
    cv2.circle(image, (center, center), 200, (230, 230, 230), -1)
    star, girdle = [], []
    for i in range(16):
        angle = math.radians(i * 22.5 - 90)
        girdle.append((center + 230 * math.cos(angle), center + 230 * math.sin(angle)))
    for i in range(8):
        angle = math.radians((2 * i + 1) * 22.5 - 90)
        star.append((center + 145 * math.cos(angle), center + 145 * math.sin(angle)))
    return image, star, girdle
