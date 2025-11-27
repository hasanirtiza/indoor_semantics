import argparse
import json
import os

import numpy as np
import cv2


def load_splat_centers(splats_path):
  # load json
    with open(splats_path, "r") as f:
        data = json.load(f)

    centers = []
    for s in data:
        m = s.get("mean", None)
        if m is None or len(m) != 3:
            continue
        centers.append(m)

    if len(centers) == 0:
        raise ValueError("No valid splat centers found in {}".format(splats_path))

    centers = np.asarray(centers, dtype=np.float32)
    return centers


def filter_floor_ceiling_height_band(
    pts,
    floor_percentile=2.0,
    ceil_percentile=98.0,
    margin=0.1,
):

    z = pts[:, 2]
    z_floor = np.percentile(z, floor_percentile)
    z_ceil = np.percentile(z, ceil_percentile)

    mask = (z > (z_floor + margin)) & (z < (z_ceil - margin))
    filtered_pts = pts[mask]
    return filtered_pts, mask

def filter_xy_percentile(pts, low=1.0, high=99.0):
    x = pts[:,0]
    y = pts[:,1]

    x_min = np.percentile(x, low)
    x_max = np.percentile(x, high)
    y_min = np.percentile(y, low)
    y_max = np.percentile(y, high)

    mask = (
        (x >= x_min) & (x <= x_max) &
        (y >= y_min) & (y <= y_max)
    )
    return pts[mask], mask

def create_occupancy_grid(
    pts_xy,
    resolution=0.05,
    padding=0.5,
):
    if pts_xy.shape[0] == 0:
        raise ValueError("No points left after floor/ceiling filtering.")

    min_xy = pts_xy.min(axis=0) - padding
    max_xy = pts_xy.max(axis=0) + padding

    size_xy = (max_xy - min_xy) / resolution
    size_xy = np.ceil(size_xy).astype(int)  # [W, H] but we will map carefully

    width = int(size_xy[0])
    height = int(size_xy[1])

    occ = np.zeros((height, width), dtype=np.uint8)
    counts = np.zeros((height, width), dtype=np.int32)

    # Map points to grid indices
    grid_xy = (pts_xy - min_xy) / resolution
    grid_xy = np.floor(grid_xy).astype(int)

    # grid_xy[:,0] -> col (x), grid_xy[:,1] -> row (y)
    # Make sure indices are in bounds
    valid = (
        (grid_xy[:, 0] >= 0) & (grid_xy[:, 0] < width) &
        (grid_xy[:, 1] >= 0) & (grid_xy[:, 1] < height)
    )
    grid_xy = grid_xy[valid]

    # Accumulate observation counts
    for x_pix, y_pix in grid_xy:
        counts[y_pix, x_pix] += 1

    # Occupancy: any cell with at least 1 observation is occupied
    occ[counts > 0] = 255

    return occ, counts, min_xy, resolution


def clean_occupancy(
    occ,
    resolution,
    min_region_area_m2=0.1,
):
   
    # Morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    occ_closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel, iterations=2)
    occ_open = cv2.morphologyEx(occ_closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        occ_open, connectivity=8
    )

    if num_labels <= 1:
        # Only background
        return occ_open

    min_area_pix = int(min_region_area_m2 / (resolution ** 2))

    cleaned = np.zeros_like(occ_open)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area_pix:
            cleaned[labels == label] = 255

    return cleaned


def detect_wall_lines(
    occ,
    min_line_length_pix=30,
    max_line_gap_pix=10,
    hough_threshold=50,
):
   
    # Edges for Hough
    edges = cv2.Canny(occ, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length_pix,
        maxLineGap=max_line_gap_pix,
    )

    return lines


def lines_to_geojson(
    lines,
    origin_xy,
    resolution,
):
  
    if lines is None or len(lines) == 0:
        return {
            "type": "FeatureCollection",
            "features": []
        }

    features = []
    for i, l in enumerate(lines):
        x1_pix, y1_pix, x2_pix, y2_pix = l[0]

        # pixel -> world
        x1 = origin_xy[0] + x1_pix * resolution
        y1 = origin_xy[1] + y1_pix * resolution
        x2 = origin_xy[0] + x2_pix * resolution
        y2 = origin_xy[1] + y2_pix * resolution

        feature = {
            "type": "Feature",
            "properties": {
                "id": int(i),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(x1), float(y1)],
                    [float(x2), float(y2)],
                ],
            },
        }
        features.append(feature)

    geo = {
        "type": "FeatureCollection",
        "features": features
    }
    return geo


def save_floorplan_png(occ, out_path):
   
    occ_u8 = occ.astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, occ_u8)


def save_geojson(geojson_obj, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(geojson_obj, f, indent=2)


def compute_floorplan_metrics(occ, resolution):
    
    # Ensure binary
    occ_bin = (occ > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        occ_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return 0.0, 0.0

    areas_pix = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas_pix))
    main_contour = contours[idx]

    area_pix = areas_pix[idx]
    perimeter_pix = cv2.arcLength(main_contour, True)

    # Convert to metric
    area_m2 = area_pix * (resolution ** 2)
    perimeter_m = perimeter_pix * resolution

    return float(area_m2), float(perimeter_m)


def compute_coverage_metrics(counts, occ_clean, min_observations):
   
    mask_floor = (occ_clean > 0)

    num_floor_cells = int(mask_floor.sum())
    if num_floor_cells == 0:
        return {
            "min_observations": int(min_observations),
            "num_floor_cells": 0,
            "num_floor_cells_well_observed": 0,
            "completeness_percent": 0.0,
            "max_observations_per_cell": 0,
            "mean_observations_per_visited_cell": 0.0,
        }

    counts_floor = counts[mask_floor]
    num_floor_cells_well = int((counts_floor >= min_observations).sum())
    completeness = 100.0 * float(num_floor_cells_well) / float(num_floor_cells)

    max_obs = int(counts_floor.max())
    visited = counts_floor[counts_floor > 0]
    if visited.size > 0:
        mean_obs = float(visited.mean())
    else:
        mean_obs = 0.0

    metrics = {
        "min_observations": int(min_observations),
        "num_floor_cells": num_floor_cells,
        "num_floor_cells_well_observed": num_floor_cells_well,
        "completeness_percent": completeness,
        "max_observations_per_cell": max_obs,
        "mean_observations_per_visited_cell": mean_obs,
    }
    return metrics


def save_coverage_heatmap(counts, occ_clean, out_path):
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    max_count = int(counts.max())
    if max_count > 0:
        vis = (counts.astype(np.float32) / float(max_count) * 255.0).astype(np.uint8)
    else:
        vis = np.zeros_like(counts, dtype=np.uint8)

    # Color map
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    # Mask outside floor to black
    mask_floor = (occ_clean > 0)
    vis_color[~mask_floor] = 0

    cv2.imwrite(out_path, vis_color)


def save_metrics_json(out_path, resolution, area_m2, perimeter_m, coverage_metrics):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    metrics = {
        "resolution_m": float(resolution),
        "area_m2": float(area_m2),
        "perimeter_m": float(perimeter_m),
        "coverage": coverage_metrics,
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


def splats2floorplan(splats_json, resolution=0.05, floor_percentile=2.0, out_floorplan="results/floorplan.png"):

    # some default params
    min_observations = 3
    hough_threshold = 50
    max_line_gap = 10
    min_line_length = 30
    min_region_area = 0.1
    z_margin = 0.1
    ceil_percentile = 98.0
    out_coverage_heatmap = "./results/coverage_heatmap.png"
    out_metrics = "./results/metrics.json"
    out_walls = "./results/walls.geojson"


    centers = load_splat_centers(splats_json)

    centers_filtered, _ = filter_floor_ceiling_height_band(
        centers,
        floor_percentile=floor_percentile,
        ceil_percentile=ceil_percentile,
        margin=z_margin,
    )
  
    centers_filtered, _ = filter_xy_percentile(
    centers_filtered,
    low=0.8,
    high=97.0
    )

    pts_xy = centers_filtered[:, :2]  # (x,y)
    occ, counts, origin_xy, res = create_occupancy_grid(
        pts_xy,
        resolution=resolution,
        padding=0.5,
    )

    occ_clean = clean_occupancy(
        occ,
        resolution=res,
        min_region_area_m2=min_region_area,
    )

    area_m2, perimeter_m = compute_floorplan_metrics(occ_clean, res)

    coverage_metrics = compute_coverage_metrics(
        counts, occ_clean, min_observations
    )

    lines = detect_wall_lines(
        occ_clean,
        min_line_length_pix=min_line_length,
        max_line_gap_pix=max_line_gap,
        hough_threshold=hough_threshold,
    )

    geo = lines_to_geojson(lines, origin_xy, res)

    save_floorplan_png(occ_clean, out_floorplan)
    save_geojson(geo, out_walls)
    save_coverage_heatmap(counts, occ_clean, out_coverage_heatmap)
    save_metrics_json(out_metrics, res, area_m2, perimeter_m, coverage_metrics)

    print("[OK] Saved floorplan to {}".format(out_floorplan))
    print("[OK] Saved walls GeoJSON to {}".format(out_walls))
    print("[OK] Saved coverage heatmap to {}".format(out_coverage_heatmap))
    print("[OK] Saved metrics JSON to {}".format(out_metrics))

    if lines is None:
        print("[WARN] No wall lines detected. Try lowering hough_threshold or min_line_length.")

    print("Floorplan area      ≈ {:.2f} m²".format(area_m2))
    print("Floorplan perimeter ≈ {:.2f} m".format(perimeter_m))
    print(
        "Coverage completeness (≥ {} obs) ≈ {:.1f}%".format(
            coverage_metrics["min_observations"],
            coverage_metrics["completeness_percent"],
        )
    )


