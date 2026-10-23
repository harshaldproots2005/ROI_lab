"""Crown ROI line finder. Copied from hardware/symmetry.py _roi_line_detection_pipeline."""
import math
import cv2
import numpy as np

def roi_line_detection_pipeline(roi_bgr,
                                 canny_low=30, canny_high=110,
                                 return_steps=False, hough_threshold=50, min_line_length=50, max_line_gap=40, use_contour_mask=True,
                                 peak_min=25.0, inlier_band_frac=0.08, row_margin_px=0,
                                 suppress_dust=False,
                                 dust_bright_thresh=100,
                                 dust_open_ksize=5,
                                 dust_min_cc_area_frac=0.003,
                                 force_horizontal=False,
                                 subpixel=False):
    """
    Industrial Peak Intrinsic Profiling Pipeline.
    Tracks tilted boundaries column-by-column based on high-contrast transition peaks,
    bypassing structural spatial assumptions and fixed masks.

    ``peak_min`` / ``inlier_band_frac`` / ``row_margin_px`` default to the values
    Crown halves has always used. Pavilion's late quad passes its own, because its
    ROI reaches past the destination into empty background where the warp border
    itself carries a strong vertical gradient.

    ``suppress_dust`` (Pavilion late-quad / analysis): threshold → drop floating
    dark islands (kachra on the bright side) → inpaint those pixels, then run
    gray Sobel column peaks on the cleaned ROI. Binary is debug-only; jagged
    white→black fronts from bloom must not own the chord.

    ``force_horizontal`` (Pavilion halves analysis): warp x is the mesh radial, so
    a true PH–PH rib is horizontal. After column peaks + inlier gate, pin the
    chord to median peak row instead of a free Huber tilt (mid dust/bloom).

    ``subpixel`` (Pavilion late quad): refine each column's peak row with a
    parabola through its neighbours, and return the Huber fit's end rows as
    floats. Whole-row peaks make a nearly flat edge a staircase (one step
    mid-chord fits as ~1.5/width rad), and whole-row ends quantise the tilt to
    atan(1 / width) — both ~0.4° on a Pavilion quad, which the angle correction
    then chases as if it were real. Crown keeps whole rows.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        empty_steps = {k: None for k in ['gray', 'enhanced', 'thresh', 'canny', 'mask', 'edges_masked', 'contour', 'contour_with_line', 'edges_bitwise', 'edges_binary', 'best_contour', 'dust_mask']}
        return (None, empty_steps) if return_steps else None

    # 1) Grayscale and Blur
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    dust_mask = None
    material_binary = None
    work_gray = gray

    # Rows this close to the top/bottom belong to the warp frame, not the stone.
    row_lo = max(0, int(row_margin_px))
    row_hi = max(row_lo + 1, h - int(row_margin_px))
    peak_floor = float(peak_min)

    if suppress_dust:
        # Threshold → wipe floating dark islands on white (and tiny bright
        # islands). Inpaint so Sobel cannot latch onto kachra; do not use the
        # cleaned binary as the column front — bloom makes that silhouette jagged.
        gray_blur = cv2.bilateralFilter(gray, 7, 60, 60)
        bright_t = int(max(1, min(254, int(dust_bright_thresh))))
        _, raw_bright = cv2.threshold(gray_blur, bright_t, 255, cv2.THRESH_BINARY)
        material_binary = raw_bright.copy()
        material_binary = cv2.morphologyEx(
            material_binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
        )
        open_k = max(3, int(dust_open_ksize))
        if open_k % 2 == 0:
            open_k += 1
        material_binary = cv2.morphologyEx(
            material_binary,
            cv2.MORPH_OPEN,
            np.ones((open_k, open_k), np.uint8),
            iterations=1,
        )
        min_cc_area = max(16, int(float(dust_min_cc_area_frac) * float(w) * float(h)))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            material_binary, connectivity=8
        )
        for label_idx in range(1, n_labels):
            if int(stats[label_idx, cv2.CC_STAT_AREA]) < min_cc_area:
                material_binary[labels == label_idx] = 0

        # Bright islands that OPEN/CC removed (legacy crown-style wipe).
        bright_dust = cv2.bitwise_and(raw_bright, cv2.bitwise_not(material_binary))

        # Floating dark islands on white: keep the main black body (touches
        # bottom of warp, else largest), paint every other dark CC white.
        dark = cv2.bitwise_not(material_binary)
        n_dark, dark_labels, dark_stats, _ = cv2.connectedComponentsWithStats(
            dark, connectivity=8
        )
        keep_dark = -1
        keep_score = -1
        for label_idx in range(1, n_dark):
            area = int(dark_stats[label_idx, cv2.CC_STAT_AREA])
            if area < 4:
                continue
            ys = np.where(dark_labels == label_idx)[0]
            if ys.size == 0:
                continue
            touches_bottom = int(ys.max()) >= (h - 2)
            score = area + (1_000_000 if touches_bottom else 0)
            if score > keep_score:
                keep_score = score
                keep_dark = label_idx
        if keep_dark < 0 and n_dark > 1:
            keep_dark = 1 + int(np.argmax(dark_stats[1:, cv2.CC_STAT_AREA]))

        cleaned_binary = np.full((h, w), 255, dtype=np.uint8)
        dark_dust = np.zeros((h, w), dtype=np.uint8)
        if keep_dark > 0:
            cleaned_binary[dark_labels == keep_dark] = 0
            dark_dust[(dark_labels > 0) & (dark_labels != keep_dark)] = 255
        else:
            cleaned_binary = material_binary.copy()

        material_binary = cleaned_binary
        dust_mask = cv2.bitwise_or(bright_dust, dark_dust)
        if int(np.count_nonzero(dust_mask)) > 0:
            work_gray = cv2.inpaint(gray, dust_mask, 3, cv2.INPAINT_TELEA)
        else:
            work_gray = gray

    blurred = cv2.GaussianBlur(work_gray, (5, 5), 0)

    # 2) COLUMN-BY-COLUMN PEAK TRANSITION GRADIENT TRACKING (gray Sobel)
    tracked_points = []
    sobely_raw = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobely = np.abs(sobely_raw)
    if dust_mask is not None and int(np.count_nonzero(dust_mask)) > 0:
        abs_sobely = abs_sobely.copy()
        abs_sobely[dust_mask > 0] = 0.0

    for col in range(2, w - 2):
        column_profile = abs_sobely[row_lo:row_hi, col]
        if column_profile.size == 0:
            continue

        best_y = row_lo + int(np.argmax(column_profile))
        peak_value = float(abs_sobely[best_y, col])

        if peak_value <= peak_floor:
            continue
        if suppress_dust:
            col_med = float(np.median(column_profile))
            if peak_value < col_med + (0.5 * peak_floor):
                continue
        peak_y = float(best_y)
        if subpixel and row_lo < best_y < row_hi - 1:
            above = float(abs_sobely[best_y - 1, col])
            below = float(abs_sobely[best_y + 1, col])
            curvature = above - 2.0 * peak_value + below
            if curvature < 0.0:
                peak_y += max(-0.5, min(0.5, 0.5 * (above - below) / curvature))
        tracked_points.append([col, peak_y])

    # Convert to numpy float32 array for tracking
    if len(tracked_points) < 10:
        # Fallback to standard Canny if the transition is overly smooth/unpolished
        roi_edges = cv2.Canny(blurred, canny_low, canny_high)
        if dust_mask is not None and int(np.count_nonzero(dust_mask)) > 0:
            roi_edges = cv2.bitwise_and(roi_edges, cv2.bitwise_not(dust_mask))
        edge_pts = np.argwhere(roi_edges > 0)
        edge_pts = [p for p in edge_pts if row_lo <= p[0] < row_hi]
        if len(edge_pts) >= 5:
            pts = np.float32([[p[1], p[0]] for p in edge_pts])
        else:
            return (None, {k: gray for k in ['gray', 'canny', 'thresh']}) if return_steps else None
    else:
        pts = np.array(tracked_points, dtype=np.float32)

    # 3) TWO-PASS GEOMETRIC OUTLIER REMOVAL (Eliminates localized debris jumps)
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    norm = math.hypot(vx, vy)
    filtered_pts = []
    MAX_DISTANCE_FROM_LINE = max(4.0, float(inlier_band_frac) * h) # Strict line variance gate

    if norm > 0:
        for pt in pts:
            dist = abs(vy * (pt[0] - x0) - vx * (pt[1] - y0)) / norm
            if dist <= MAX_DISTANCE_FROM_LINE:
                filtered_pts.append(pt)

    if len(filtered_pts) >= 5:
        final_pts = np.array(filtered_pts, dtype=np.float32)
    else:
        final_pts = pts

    # 4) Chord: free Huber, or radial-only (horizontal) for halves analysis.
    if force_horizontal:
        # Warp x = mesh radial → true PH–PH rib is horizontal. Median peak row
        # ignores mid-plane tilt from dust/bloom while keeping column-gradient y.
        y_med = float(np.median(final_pts[:, 1]))
        y_i = int(round(max(0, min(h - 1, y_med))))
        p1 = (0, y_i)
        p2 = (w - 1, y_i)
    else:
        [vx, vy, x0, y0] = cv2.fitLine(final_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

        # Project edge points completely out to the margins of the ROI matrix width
        t0 = (0 - x0) / vx if abs(vx) > 1e-6 else 0
        t1 = ((w - 1) - x0) / vx if abs(vx) > 1e-6 else 0

        if subpixel:
            p1 = (0.0, float(y0 + t0 * vy))
            p2 = (float(w - 1), float(y0 + t1 * vy))
        else:
            p1 = (0, int(round(y0 + t0 * vy)))
            p2 = (w - 1, int(round(y0 + t1 * vy)))
    line_result = (p1, p2)

    # 5) Build debug steps visualization
    if return_steps:
        # Generate a clean diagnostic mask showcasing tracking targets
        diagnostic_mask = np.zeros_like(gray)
        for pt in final_pts:
            diagnostic_mask[int(pt[1]), int(pt[0])] = 255
            
        contour_with_line = cv2.cvtColor(diagnostic_mask, cv2.COLOR_GRAY2BGR)
        cv2.line(
            contour_with_line,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (0, 0, 255), 2, cv2.LINE_AA,
        )
        if material_binary is not None:
            binary = material_binary
        else:
            _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

        steps_dict = {
            'gray': gray,
            'enhanced': work_gray,
            'thresh': binary,
            'edges_bitwise': diagnostic_mask,
            'edges_binary': binary,
            'canny': diagnostic_mask,
            'mask': dust_mask,
            'dust_mask': dust_mask,
            'edges_masked': diagnostic_mask,
            'contour': contour_with_line,
            'contour_with_line': contour_with_line,
            'best_contour': None,
        }
        return line_result, steps_dict

    return line_result
