import os
import re
import shutil
import json
import argparse
import numpy as np
from PIL import Image
import open3d as o3d

from viz import load_npz, load_grasps, grasp_to_T, make_gripper, make_grasp_markers, make_robot_base_marker, assign_grasp_colors, draw_grasp_labels, render_to_file, render_to_array, render_and_composite, render_interactive
from viz_with_arm import make_franka_geometry


def cam_index_from_name(camera_name):
    m = re.search(r"(\d+)$", camera_name)
    return int(m.group(1)) if m else 0


def build_world_geometries(pcd, grasps, top_k=10, robot_geoms=None, color_assignments=None, include_pcd=True):
    """Return all geometries in world frame. Open3D camera extrinsic handles the projection."""
    geometries = [pcd] if include_pcd else []
    ranked = sorted(grasps, key=lambda x: x["score"], reverse=True)[:top_k]
    ranked_cas = [color_assignments[i] for i in range(len(ranked))] if color_assignments else [None] * len(ranked)
    for g, ca in zip(ranked, ranked_cas):
        color = ca["color"] if ca else None
        geometries.extend(make_gripper(grasp_to_T(g), g["width"], g["depth"], g["height"], color=color))
    if color_assignments:
        geometries.extend(make_grasp_markers(ranked, ranked_cas))
    for geom in (robot_geoms or []):
        geometries.append(geom)
    return geometries


def render_pov(pcd, cam, grasps=None, top_k=10, robot_geoms=None):
    """
    Render a pointcloud from a single camera's POV.

    Args:
        pcd: o3d.geometry.PointCloud in world frame.
        cam: calibration dict with 'K', 'T_cam_from_world', 'image_size'.
        grasps: optional list of grasp dicts (same format as grasps.json).
        top_k: max grasps to render.
        robot_geoms: optional list of extra o3d geometries in world frame.

    Returns:
        (H, W, 3) uint8 numpy array of the rendered view.
    """
    geometries = build_world_geometries(pcd, grasps or [], top_k=top_k, robot_geoms=robot_geoms)
    T_cam_from_world = np.array(cam["T_cam_from_world"], dtype=np.float64)
    K_mat = np.array(cam["K"], dtype=np.float64)
    width, height = cam["image_size"]
    return render_to_array(geometries, width, height, K_mat, T_cam_from_world)


def main(args):
    pcd = load_npz(args.npz)
    grasps = load_grasps(args.grasp_json)

    robot_geoms = []
    if args.robot_calib:
        with open(args.robot_calib) as f:
            calib = json.load(f)
        T_world_from_base = np.array(calib["T_world_from_base"], dtype=np.float64)
        robot_geoms.append(make_robot_base_marker(T_world_from_base))
        robot_geoms.extend(make_franka_geometry(T_world_from_base))

    if not args.sim:
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)
        os.makedirs(args.out_dir)

    # Build geometries once in world frame; render per camera via extrinsic
    geometries = build_world_geometries(pcd, grasps, top_k=10, robot_geoms=robot_geoms)

    for cam_path in args.cams:
        cam = json.load(open(cam_path))
        T_cam_from_world = np.array(cam["T_cam_from_world"], dtype=np.float64)
        K_mat = np.array(cam["K"], dtype=np.float64)
        width, height = cam["image_size"]
        out_path = os.path.join(args.out_dir, f"{cam['camera_name']}.png")

        if args.sim:
            render_interactive(geometries, window_name=cam["camera_name"])
        elif args.color_dir:
            cam_idx = cam_index_from_name(cam["camera_name"])
            photo_path = os.path.join(args.color_dir, str(cam_idx), "0.png")
            composite = render_and_composite(geometries, photo_path, width, height, K_mat, T_cam_from_world)
            Image.fromarray(composite).save(out_path)
            print("saved:", out_path)
        else:
            render_to_file(geometries, out_path, width, height, K_mat, T_cam_from_world)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--grasp_json", required=True)
    parser.add_argument("--cams", nargs="+", required=True)
    parser.add_argument("--out_dir", default="renders")
    parser.add_argument("--color_dir", default=None,
                        help="root of real photos: <color_dir>/<cam_idx>/0.png. "
                             "If provided, composites render over real photo.")
    parser.add_argument("--robot_calib", default=None, help="path to robot_calib_result.json to overlay robot base frame")
    parser.add_argument("--sim", action="store_true", help="open interactive O3D viewer per camera instead of saving images")

    main(parser.parse_args())
