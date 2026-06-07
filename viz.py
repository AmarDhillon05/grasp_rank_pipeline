"""
viz.py — shared visualization primitives for point cloud + grasp rendering.

Provides loaders, gripper geometry builders, and two render backends:
  - render_to_file: hidden Visualizer window → PNG, using real camera K + extrinsics
  - render_interactive: open3d.draw_geometries interactive viewer

Imported by mcliv.py and mpcliv.py. Also provides make_robot_base_marker
for overlaying the calibrated robot base frame in the scene.
"""

import colorsys
import json
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont


def load_npz(npz_path):
    """Load a point cloud from an NPZ file with 'points' and 'colors' arrays."""
    data = np.load(npz_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["points"].astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(data["colors"].astype(np.float32))
    return pcd


def load_grasps(path, top_k=None):
    """
    Load and sort grasps from a JSON file by score (descending).

    Args:
        path: path to grasps JSON (list of dicts with 'score', 'pose_matrix_4x4', etc.)
        top_k: if set, return only the top-k highest-scoring grasps.

    Returns:
        List of grasp dicts sorted by score descending.
    """
    with open(path) as f:
        grasps = json.load(f)
    grasps = sorted(grasps, key=lambda g: g["score"], reverse=True)
    if top_k is not None:
        grasps = grasps[:top_k]
    return grasps


def grasp_to_T(g):
    """Extract the 4x4 pose matrix from a grasp dict."""
    return np.array(g["pose_matrix_4x4"], dtype=np.float32)


def assign_grasp_colors(grasps):
    """
    Assign a distinct high-contrast color and short label to each grasp.

    Colors are evenly spaced around the HSV hue wheel at full saturation/value,
    giving maximally distinct colors regardless of how many grasps there are.

    Returns a list of dicts (parallel to `grasps`) with:
        'label': str  e.g. "G0", "G1", ...
        'color': [r, g, b] floats in [0, 1]
    """
    n = len(grasps)
    assignments = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.90, 0.95)
        assignments.append({"label": f"G{i}", "color": [r, g, b]})
    return assignments


def draw_grasp_labels(composite, grasps, color_assignments, K_mat, T_cam_from_world):
    """
    Project each grasp origin into 2D and draw a colored dot + label on the image.

    Args:
        composite: (H, W, 3) uint8 numpy array.
        grasps: list of grasp dicts (same order as color_assignments).
        color_assignments: list of dicts from assign_grasp_colors().
        K_mat: (3, 3) camera intrinsic matrix.
        T_cam_from_world: (4, 4) camera extrinsic matrix.

    Returns:
        (H, W, 3) uint8 numpy array with labels drawn on top.
    """
    img = Image.fromarray(composite)
    draw = ImageDraw.Draw(img)
    h_img, w_img = composite.shape[:2]
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    R = T_cam_from_world[:3, :3]
    t = T_cam_from_world[:3, 3]

    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    for g, ca in zip(grasps, color_assignments):
        p_world = grasp_to_T(g)[:3, 3]
        p_cam = R @ p_world + t
        if p_cam[2] <= 0:
            continue
        u = int(fx * p_cam[0] / p_cam[2] + cx)
        v = int(fy * p_cam[1] / p_cam[2] + cy)
        if not (0 <= u < w_img and 0 <= v < h_img):
            continue
        fill = tuple(int(c * 255) for c in ca["color"])
        r = 6
        draw.ellipse([u - r, v - r, u + r, v + r], fill=fill, outline=(0, 0, 0), width=1)
        draw.text((u + r + 2, v - 8), ca["label"], fill=(0, 0, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((u + r + 2, v - 8), ca["label"], fill=fill, font=font)

    return np.array(img)


def draw_grasp_keypoints(composite, grasps, color_assignments, K_mat, T_cam_from_world):
    """
    Project gripper keypoints to 2D and draw bold colored dots — no lines, no labels.

    Fingertips get the largest dots (r=8), finger bases medium (r=5), wrist small (r=4).
    Intended as an overlay on top of a 3D render to make grasp positions pop.
    """
    img = Image.fromarray(composite)
    draw = ImageDraw.Draw(img)
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    R_ext = T_cam_from_world[:3, :3]
    t_ext = T_cam_from_world[:3, 3]

    def project(p_world):
        p = R_ext @ p_world + t_ext
        if p[2] <= 0:
            return None
        return (fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy)

    for g, ca in zip(grasps, color_assignments):
        T = grasp_to_T(g)
        Rg, tg = T[:3, :3], T[:3, 3]
        w, d = g["width"], g["depth"]
        fill = tuple(int(c * 255) for c in ca["color"])

        keypoints = [
            (tg + Rg @ np.array([d,  -w / 2, 0]), 8),  # left fingertip
            (tg + Rg @ np.array([d,   w / 2, 0]), 8),  # right fingertip
            (tg + Rg @ np.array([0,  -w / 2, 0]), 5),  # left base
            (tg + Rg @ np.array([0,   w / 2, 0]), 5),  # right base
            (tg,                                   4),  # wrist
        ]
        for p_world, r in keypoints:
            pt = project(p_world)
            if pt:
                u, v = pt
                draw.ellipse([u - r, v - r, u + r, v + r],
                             fill=fill, outline=(0, 0, 0), width=1)

    return np.array(img)


def draw_grasp_schematic(composite, grasps, color_assignments, K_mat, T_cam_from_world, line_width=3):
    """
    Draw a 2D projected schematic of each gripper directly on a photo — no Open3D needed.

    For each grasp draws:
      - Bold filled dots at the two fingertips
      - Smaller dots at the two finger bases and wrist
      - A U-shaped line (fingertip → base → base → fingertip) for the jaw
      - A thin spoke from the wrist to the centre of the palm bar
      - A label at the wrist

    Args:
        composite: (H, W, 3) uint8 numpy array (the background photo).
        grasps: list of grasp dicts.
        color_assignments: list of dicts from assign_grasp_colors().
        K_mat: (3, 3) camera intrinsic matrix.
        T_cam_from_world: (4, 4) camera extrinsic matrix.

    Returns:
        (H, W, 3) uint8 numpy array with schematics drawn on top.
    """
    img = Image.fromarray(composite)
    draw = ImageDraw.Draw(img)
    h_img, w_img = composite.shape[:2]
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    R_ext = T_cam_from_world[:3, :3]
    t_ext = T_cam_from_world[:3, 3]

    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    def project(p_world):
        p = R_ext @ p_world + t_ext
        if p[2] <= 0:
            return None
        return (fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy)

    def dot(pt, radius, fill):
        u, v = pt
        draw.ellipse([u - radius, v - radius, u + radius, v + radius],
                     fill=fill, outline=(0, 0, 0), width=1)

    def rect_line(a, b, thickness, fill):
        """Draw a flat-capped filled rectangle from point a to point b."""
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return
        # perpendicular unit vector scaled to half-thickness
        px = -dy / length * thickness / 2
        py =  dx / length * thickness / 2
        corners = [(ax + px, ay + py), (ax - px, ay - py),
                   (bx - px, by - py), (bx + px, by + py)]
        draw.polygon(corners, fill=fill, outline=(0, 0, 0))

    for g, ca in zip(grasps, color_assignments):
        T = grasp_to_T(g)
        Rg, tg = T[:3, :3], T[:3, 3]
        w, d = g["width"], g["depth"]

        # Five keypoints in world frame
        lf_tip  = tg + Rg @ np.array([d,  -w / 2, 0])
        lf_base = tg + Rg @ np.array([0,  -w / 2, 0])
        rf_base = tg + Rg @ np.array([0,   w / 2, 0])
        rf_tip  = tg + Rg @ np.array([d,   w / 2, 0])
        wrist   = tg

        pts = [project(p) for p in (lf_tip, lf_base, rf_base, rf_tip, wrist)]
        lft, lfb, rfb, rft, wpt = pts

        fill = tuple(int(c * 255) for c in ca["color"])

        # U-shaped jaw outline + wrist spoke
        for a, b in [(lft, lfb), (lfb, rfb), (rfb, rft)]:
            if a and b:
                rect_line(a, b, line_width, fill)
        if lfb and rfb and wpt:
            mid = ((lfb[0] + rfb[0]) / 2, (lfb[1] + rfb[1]) / 2)
            rect_line(wpt, mid, max(1, line_width - 2), fill)

        # Dots: large at fingertips, medium at bases, small at wrist
        for pt, r in ((lft, 8), (rft, 8), (lfb, 5), (rfb, 5), (wpt, 4)):
            if pt:
                dot(pt, r, fill)

        # Label next to wrist
        if wpt and 0 <= int(wpt[0]) < w_img and 0 <= int(wpt[1]) < h_img:
            tx, ty = wpt[0] + 10, wpt[1] - 8
            draw.text((tx, ty), ca["label"], fill=(0, 0, 0), font=font,
                      stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((tx, ty), ca["label"], fill=fill, font=font)

    return np.array(img)


def make_grasp_markers(grasps, color_assignments):
    """
    Build small Open3D spheres at each gripper keypoint in world frame.

    Renders inside the same Open3D scene as the gripper rods, so the markers
    are guaranteed to be pixel-perfect with the rendered grippers — no separate
    2D projection needed.

    Keypoint radii (metres):
        fingertips  0.005   (largest — the grasping contacts)
        finger bases 0.004
        wrist        0.003
    """
    meshes = []
    for g, ca in zip(grasps, color_assignments):
        T = grasp_to_T(g).astype(np.float64)
        Rg, tg = T[:3, :3], T[:3, 3]
        w, d = g["width"], g["depth"]
        color = ca["color"]

        for p_local, radius in [
            (np.array([d,    -w / 2, 0.0]), 0.005),   # left fingertip
            (np.array([d,     w / 2, 0.0]), 0.005),   # right fingertip
            (np.array([0.0, -w / 2, 0.0]), 0.004),   # left base
            (np.array([0.0,  w / 2, 0.0]), 0.004),   # right base
            (np.array([0.0,   0.0,  0.0]), 0.003),   # wrist
        ]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(color)
            sphere.translate(tg + Rg @ p_local)
            meshes.append(sphere)

    return meshes


def make_gripper(T, width, depth, height, color=None):
    """
    Build a solid parallel-jaw gripper mesh at pose T.

    Returns three TriangleMesh objects: [left finger, right finger, base].
    If color ([r, g, b] floats) is provided, fingers use that color and the
    base uses a 50% darkened version. Otherwise defaults to red fingers / blue base.

    Args:
        T: (4, 4) gripper pose matrix.
        width: jaw opening width (m).
        depth: finger depth / reach (m).
        height: finger height (m).
        color: optional [r, g, b] floats in [0, 1] for this gripper.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    finger_color = color if color is not None else [0.9, 0.2, 0.2]
    base_color = [c * 0.5 for c in finger_color] if color is not None else [0.2, 0.2, 0.9]

    def box(center_local, size, clr):
        b = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
        b.compute_vertex_normals()
        b.translate(-np.array(size) / 2)
        b.rotate(R, center=(0, 0, 0))
        b.translate(t + R @ center_local)
        b.paint_uniform_color(clr)
        return b

    t = 0.003  # rod cross-section (3 mm)
    finger_size = np.array([depth, t, t])
    return [
        box(np.array([depth / 2, -width / 2, 0]), finger_size, finger_color),
        box(np.array([depth / 2,  width / 2, 0]), finger_size, finger_color),
        box(np.array([0, 0, 0]), np.array([t, width, t]), base_color),
    ]


def _make_visualizer(geometries, width, height, K_mat, T_cam_from_world):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    for g in geometries:
        vis.add_geometry(g)
    vis.get_render_option().background_color = np.array([0., 0., 0.])
    vis.poll_events()
    vis.update_renderer()
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intrinsic
    params.extrinsic = T_cam_from_world
    vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    return vis


def render_to_array(geometries, width, height, K_mat, T_cam_from_world):
    """Render world-frame geometries from a camera POV. Returns (H, W, 3) uint8 numpy array."""
    vis = _make_visualizer(geometries, width, height, K_mat, T_cam_from_world)
    render_f = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (render_f * 255).astype(np.uint8)


def render_and_composite(geometries, photo_path, width, height, K_mat, T_cam_from_world):
    """
    Render world-frame geometries from a camera POV and composite over a real photo.

    Non-black render pixels (float value > 10/255 in any channel) replace the
    corresponding photo pixels. Returns (H, W, 3) uint8 composite.
    """
    vis = _make_visualizer(geometries, width, height, K_mat, T_cam_from_world)
    render_f = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    render_u8 = (render_f * 255).astype(np.uint8)

    photo = np.array(Image.open(photo_path).convert("RGB"))
    if photo.shape[:2] != (height, width):
        photo = np.array(Image.fromarray(photo).resize((width, height), Image.LANCZOS))

    fg_mask = render_f.max(axis=2) > (10 / 255.0)
    composite = photo.copy()
    composite[fg_mask] = render_u8[fg_mask]
    return composite


def render_to_file(geometries, out_path, width, height, K_mat, T_cam_from_world):
    """Render world-frame geometries from a camera POV to a PNG file."""
    vis = _make_visualizer(geometries, width, height, K_mat, T_cam_from_world)
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()
    print("saved:", out_path)


def make_robot_base_marker(T_world_from_base, size=0.25):
    """
    Build a coordinate frame + orange sphere marking the robot base origin.

    The coordinate frame axes (RGB = XYZ) show base orientation.
    The sphere makes the base origin easy to spot at a glance.

    Args:
        T_world_from_base: (4, 4) robot base → world transform.
        size: coordinate frame axis length in metres.

    Returns:
        A single TriangleMesh in world frame.
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T_world_from_base)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([1.0, 0.5, 0.0])
    sphere.transform(T_world_from_base)

    return frame + sphere


def render_interactive(geometries, window_name=""):
    """Open an interactive Open3D viewer window (blocking until closed)."""
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
