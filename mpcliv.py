import os
import re
import shutil
import json
import numpy as np
import open3d as o3d
from PIL import Image


# ---------- load NPZ ----------
def load_npz(npz_path):
    data = np.load(npz_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["points"])
    pcd.colors = o3d.utility.Vector3dVector(data["colors"])
    return pcd


# ---------- load grasps ----------
def load_grasps(path):
    with open(path, "r") as f:
        return json.load(f)


def grasp_to_T(g):
    return np.array(g["pose_matrix_4x4"], dtype=np.float32)


# ---------- gripper ----------
def make_gripper(T, width, depth, height):
    R = T[:3, :3]
    t = T[:3, 3]

    def box(center, size, color):
        b = o3d.geometry.TriangleMesh.create_box(*size)
        b.compute_vertex_normals()
        b.translate(-np.array(size) / 2)
        b.rotate(R, center=(0, 0, 0))
        b.translate(t + R @ center)
        b.paint_uniform_color(color)
        return b

    finger = np.array([depth, 0.01, height])

    return [
        box(np.array([depth/2, -width/2, 0]), finger, [1, 0, 0]),
        box(np.array([depth/2,  width/2, 0]), finger, [1, 0, 0]),
    ]


# ---------- camera index from name (e.g. "realsense_2" -> 2) ----------
def cam_index_from_name(camera_name):
    m = re.search(r"(\d+)$", camera_name)
    return int(m.group(1)) if m else 0


# ---------- render + composite ----------
def render_cam_windows(pcd, grasps, cam, photo_path, out_path, top_k=10):

    width, height = cam["image_size"]
    K_mat = np.array(cam["K"], dtype=np.float64)
    T_world_from_cam = np.array(cam["T_world_from_cam"], dtype=np.float64)
    T_cam_from_world = np.linalg.inv(T_world_from_cam)

    pcd_cam = o3d.geometry.PointCloud(pcd)
    pcd_cam.transform(T_cam_from_world)

    geometries = [pcd_cam]
    for g in sorted(grasps, key=lambda x: x["score"], reverse=True)[:top_k]:
        T_w = grasp_to_T(g)
        T_c = T_cam_from_world @ T_w
        geometries.extend(make_gripper(T_c, g["width"], g["depth"], g["height"]))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    for g in geometries:
        vis.add_geometry(g)
    vis.get_render_option().background_color = np.array([0., 0., 0.])
    vis.poll_events()
    vis.update_renderer()

    # apply actual camera intrinsics so the render matches the real photo
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intrinsic
    params.extrinsic = T_cam_from_world
    vis.get_view_control().convert_from_pinhole_camera_parameters(
        params, allow_arbitrary=True
    )

    vis.poll_events()
    vis.update_renderer()

    # capture render as float buffer [H, W, 3] in [0, 1]
    render_buf = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    render_f = np.asarray(render_buf)          # [H, W, 3], float32 [0, 1]
    render_u8 = (render_f * 255).astype(np.uint8)

    # load real photo
    photo = np.array(Image.open(photo_path).convert("RGB"))
    if photo.shape[:2] != (height, width):
        photo = np.array(
            Image.fromarray(photo).resize((width, height), Image.LANCZOS)
        )

    # composite: render foreground (non-black) over real photo
    fg_mask = render_f.max(axis=2) > (10 / 255.0)   # [H, W] bool
    composite = photo.copy()
    composite[fg_mask] = render_u8[fg_mask]

    Image.fromarray(composite).save(out_path)
    print("saved:", out_path)


# ---------- main ----------
def main(npz_path, grasp_json, cam_jsons, color_dir, out_dir, top_k=10):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    pcd = load_npz(npz_path)
    grasps = load_grasps(grasp_json)

    for cam_path in cam_jsons:
        cam = json.load(open(cam_path))

        cam_idx = cam_index_from_name(cam["camera_name"])
        photo_path = os.path.join(color_dir, str(cam_idx), "0.png")

        out_path = os.path.join(out_dir, f"{cam['camera_name']}.png")

        render_cam_windows(pcd, grasps, cam, photo_path, out_path, top_k=top_k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--grasp_json", required=True)
    parser.add_argument("--cams", nargs="+", required=True)
    parser.add_argument("--color_dir", default="Rope_1/color",
                        help="root of color images: <color_dir>/<cam_idx>/0.png")
    parser.add_argument("--out_dir", default="renders_with_photo")
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()
    main(args.npz, args.grasp_json, args.cams, args.color_dir, args.out_dir, args.top_k)
