import os
import shutil
import json
import argparse
import numpy as np
import open3d as o3d

from viz import load_npz, load_grasps, grasp_to_T, make_gripper, make_robot_base_marker, render_to_file, render_interactive
from viz_with_arm import make_franka_geometry


def build_cam_geometries(pcd, grasps, cam, top_k=10, robot_geoms=None):
    T_cam_from_world = np.linalg.inv(np.array(cam["T_world_from_cam"], dtype=np.float64))

    pcd_cam = o3d.geometry.PointCloud(pcd)
    pcd_cam.transform(T_cam_from_world)

    geometries = [pcd_cam]
    for g in sorted(grasps, key=lambda x: x["score"], reverse=True)[:top_k]:
        T_c = T_cam_from_world @ grasp_to_T(g)
        geometries.extend(make_gripper(T_c, g["width"], g["depth"], g["height"]))

    for geom in (robot_geoms or []):
        g = o3d.geometry.TriangleMesh(geom)
        g.transform(T_cam_from_world)
        geometries.append(g)

    return geometries, T_cam_from_world


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

    for cam_path in args.cams:
        cam = json.load(open(cam_path))
        geometries, T_cam_from_world = build_cam_geometries(pcd, grasps, cam, top_k=10, robot_geoms=robot_geoms)

        if args.sim:
            render_interactive(geometries, window_name=cam["camera_name"])
        else:
            K_mat = np.array(cam["K"], dtype=np.float64)
            width, height = cam["image_size"]
            out_path = os.path.join(args.out_dir, f"{cam['camera_name']}.png")
            render_to_file(geometries, out_path, width, height, K_mat, T_cam_from_world)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--grasp_json", required=True)
    parser.add_argument("--cams", nargs="+", required=True)
    parser.add_argument("--out_dir", default="renders")
    parser.add_argument("--robot_calib", default=None, help="path to robot_calib_result.json to overlay robot base frame")
    parser.add_argument("--sim", action="store_true", help="open interactive O3D viewer per camera instead of saving images")

    main(parser.parse_args())
