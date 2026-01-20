# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import trimesh, logging, importlib, scipy
import torch
import torch.nn.functional as F
import torchvision
import nvdiffrast.torch as dr
import open3d as o3d
import cv2
import numpy as np
from PIL import Image
from transformations import *
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


import kornia

try:
    import warp as wp
    wp.init()
except:
    raise ImportError("Please install warp from https://github.com/NVIDIA/warp")

BAD_DEPTH = 99
BAD_COLOR = 0

glcam_in_cvcam = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
).astype(float)

COLOR_MAP = np.array(
    [
        [0, 0, 0],  # Ignore
        [128, 0, 0],  # Background
        [0, 128, 0],  # Wall
        [128, 128, 0],  # Floor
        [0, 0, 128],  # Ceiling
        [128, 0, 128],  # Table
        [0, 128, 128],  # Chair
        [128, 128, 128],  # Window
        [64, 0, 0],  # Door
        [192, 0, 0],  # Monitor
        [64, 128, 0],  # 11th
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
    ]
)


def set_logging_format(level=logging.INFO):
    importlib.reload(logging)
    FORMAT = "[%(funcName)s()] %(message)s"
    logging.basicConfig(level=level, format=FORMAT)


set_logging_format()


def make_mesh_tensors(
    mesh: trimesh.Trimesh, device="cuda", max_tex_size=None
) -> Dict[str, torch.Tensor]:
    """Convert a trimesh mesh to a dictionary of torch tensors for rendering.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to convert.
    device : str, optional
        Device to place the tensors on, by default 'cuda'
    max_tex_size : int, optional
        Maximum texture size, by default None

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing the mesh tensors.
        pos: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        vnormals: Vertex normals (N, 3)
    """
    mesh_tensors = {}
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        img = np.array(mesh.visual.material.image.convert("RGB"))
        img = img[..., :3]
        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size > max_tex_size:
                scale = 1 / max_size * max_tex_size
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        mesh_tensors["tex"] = (
            torch.as_tensor(img, device=device, dtype=torch.float)[None] / 255.0
        )
        mesh_tensors["uv_idx"] = torch.as_tensor(
            mesh.faces, device=device, dtype=torch.int
        )
        uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors["uv"] = uv
    else:
        if mesh.visual.vertex_colors is None:
            mesh.visual.vertex_colors = np.tile(
                np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1)
            )
        mesh_tensors["vertex_color"] = (
            torch.as_tensor(
                mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float
            )
            / 255.0
        )

    mesh_tensors.update(
        {
            "pos": torch.tensor(mesh.vertices, device=device, dtype=torch.float),
            "faces": torch.tensor(mesh.faces, device=device, dtype=torch.int),
            "vnormals": torch.tensor(
                mesh.vertex_normals, device=device, dtype=torch.float
            ),
            "diameter" : torch.tensor(compute_mesh_diameter(mesh=mesh), device=device, dtype=torch.float)
        }
    )
    return mesh_tensors


def depth2xyzmap(depth: np.ndarray, K: np.ndarray, uvs=None) -> np.ndarray:
    """
    Convert a depth map to a 3D XYZ coordinate map using camera intrinsics.

    This function transforms depth values into 3D Cartesian coordinates (X, Y, Z)
    using the camera's intrinsic matrix. It handles both full depth maps and
    specific pixel coordinates.

    Parameters
    ----------
    depth : np.ndarray
        Depth map of shape (H, W) containing depth values in meters.

    K : np.ndarray
        Camera intrinsic matrix of shape (3, 3).

    uvs : np.ndarray, optional
        Array of pixel coordinates of shape (N, 2) in format [u, v] (column, row).
        If None, all pixels in the depth map are converted. Default is None.

    Returns
    -------
    np.ndarray
        XYZ coordinate map of shape (H, W, 3) where each pixel contains its
        3D coordinate [X, Y, Z]. Invalid depth values are set to [0, 0, 0].
    """
    invalid_mask = depth < 0.001
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(
            np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij"
        )
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0

    return xyz_map


def compute_mesh_diameter(
    model_pts: np.ndarray | None = None,
    mesh: trimesh.Trimesh | None = None,
    n_sample=1000,
) -> float:
    """Compute the diameter of a 3D mesh or a set of points.

    Parameters
    ----------
    model_pts : np.ndarray | None, optional
        _description_, by default None
    mesh : trimesh.Trimesh | None, optional
        _description_, by default None
    n_sample : int, optional
        _description_, by default 1000

    Returns
    -------
    float
        The diameter of the mesh or point set.
    """
    if mesh is not None:
        u, s, vh = scipy.linalg.svd(mesh.vertices, full_matrices=False)
        pts = u @ s
        diameter = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        return float(diameter)

    if n_sample is None:
        pts = model_pts
    else:
        ids = np.random.choice(
            len(model_pts), size=min(n_sample, len(model_pts)), replace=False
        )
        pts = model_pts[ids]
    dists = np.linalg.norm(pts[None] - pts[:, None], axis=-1)
    diameter = dists.max()
    return diameter


def nvdiffrast_render(
    K=None,
    H=None,
    W=None,
    ob_in_cams=None,
    glctx=None,
    context="cuda",
    get_normal=False,
    mesh_tensors=None,
    mesh=None,
    projection_mat=None,
    bbox2d=None,
    output_size=None,
    use_light=False,
    light_color=None,
    light_dir=np.array([0, 0, 1]),
    light_pos=np.array([0, 0, 0]),
    w_ambient=0.8,
    w_diffuse=0.5,
    extra={},
):
    """Just plain rendering, not support any gradient
    @K: (3,3) np array
    @ob_in_cams: (N,4,4) torch tensor, openCV camera
    @projection_mat: np array (4,4)
    @output_size: (height, width)
    @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
    @light_dir: in cam space
    @light_pos: in cam space
    """
    if glctx is None:
        if context == "gl":
            glctx = dr.RasterizeGLContext()
        elif context == "cuda":
            glctx = dr.RasterizeCudaContext()
        else:
            raise NotImplementedError
        logging.info("created context")

    if mesh_tensors is None:
        mesh_tensors = make_mesh_tensors(mesh)
    pos = mesh_tensors["pos"]
    vnormals = mesh_tensors["vnormals"]
    pos_idx = mesh_tensors["faces"]
    has_tex = "tex" in mesh_tensors

    ob_in_glcams = (
        torch.tensor(glcam_in_cvcam, device="cuda", dtype=torch.float)[None]
        @ ob_in_cams
    )
    if projection_mat is None:
        projection_mat = projection_matrix_from_intrinsics(
            K, height=H, width=W, znear=0.001, zfar=100
        )
    projection_mat = torch.as_tensor(
        projection_mat.reshape(-1, 4, 4), device="cuda", dtype=torch.float
    )
    mtx = projection_mat @ ob_in_glcams

    if output_size is None:
        output_size = np.asarray([H, W])

    pts_cam = transform_pts(pos, ob_in_cams)
    pos_homo = to_homo_torch(pos)
    pos_clip = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
    if bbox2d is not None:
        l = bbox2d[:, 0]
        t = H - bbox2d[:, 1]
        r = bbox2d[:, 2]
        b = H - bbox2d[:, 3]
        tf = (
            torch.eye(4, dtype=torch.float, device="cuda")
            .reshape(1, 4, 4)
            .expand(len(ob_in_cams), 4, 4)
            .contiguous()
        )
        tf[:, 0, 0] = W / (r - l)
        tf[:, 1, 1] = H / (t - b)
        tf[:, 3, 0] = (W - r - l) / (r - l)
        tf[:, 3, 1] = (H - t - b) / (t - b)
        pos_clip = pos_clip @ tf
    rast_out, _ = dr.rasterize(
        glctx, pos_clip, pos_idx, resolution=np.asarray(output_size)
    )
    xyz_map, _ = dr.interpolate(pts_cam, rast_out, pos_idx)
    depth = xyz_map[..., 2]
    if has_tex:
        texc, _ = dr.interpolate(mesh_tensors["uv"], rast_out, mesh_tensors["uv_idx"])
        color = dr.texture(mesh_tensors["tex"], texc, filter_mode="linear")
    else:
        color, _ = dr.interpolate(mesh_tensors["vertex_color"], rast_out, pos_idx)

    if use_light:
        get_normal = True
    if get_normal:
        vnormals_cam = transform_dirs(vnormals, ob_in_cams)
        normal_map, _ = dr.interpolate(vnormals_cam, rast_out, pos_idx)
        normal_map = F.normalize(normal_map, dim=-1)
        normal_map = torch.flip(normal_map, dims=[1])
    else:
        normal_map = None

    if use_light:
        if light_dir is not None:
            light_dir_neg = -torch.as_tensor(
                light_dir, dtype=torch.float, device="cuda"
            )
        else:
            light_dir_neg = (
                torch.as_tensor(light_pos, dtype=torch.float, device="cuda").reshape(
                    1, 1, 3
                )
                - pts_cam
            )
        diffuse_intensity = (
            (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1))
            .sum(dim=-1)
            .clip(0, 1)[..., None]
        )
        diffuse_intensity_map, _ = dr.interpolate(
            diffuse_intensity, rast_out, pos_idx
        )  # (N_pose, H, W, 1)
        if light_color is None:
            light_color = color
        else:
            light_color = torch.as_tensor(light_color, device="cuda", dtype=torch.float)
        color = color * w_ambient + diffuse_intensity_map * light_color * w_diffuse

    color = color.clip(0, 1)
    color = color * torch.clamp(
        rast_out[..., -1:], 0, 1
    )  # Mask out background using alpha
    color = torch.flip(color, dims=[1])  # Flip Y coordinates
    depth = torch.flip(depth, dims=[1])
    extra["xyz_map"] = torch.flip(xyz_map, dims=[1])
    return color, depth, normal_map


def set_seed(random_seed):
    import torch, random

    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def make_grid_image(imgs, nrow, padding=5, pad_value=255):
    """
    @imgs: (B,H,W,C) np array
    @nrow: num of images per row
    """
    grid = torchvision.utils.make_grid(
        torch.as_tensor(np.asarray(imgs)).permute(0, 3, 1, 2),
        nrow=nrow,
        padding=padding,
        pad_value=pad_value,
    )
    grid = grid.permute(1, 2, 0).contiguous().data.cpu().numpy().astype(np.uint8)
    return grid


if wp is not None:

    @wp.kernel(enable_backward=False)
    def bilateral_filter_depth_kernel(
        depth: wp.array(dtype=float, ndim=2),
        out: wp.array(dtype=float, ndim=2),
        radius: int,
        zfar: float,
        sigmaD: float,
        sigmaR: float,
    ):
        h, w = wp.tid()
        H = depth.shape[0]
        W = depth.shape[1]
        if w >= W or h >= H:
            return
        out[h, w] = 0.0
        mean_depth = float(0.0)
        num_valid = int(0)
        for u in range(w - radius, w + radius + 1):
            if u < 0 or u >= W:
                continue
            for v in range(h - radius, h + radius + 1):
                if v < 0 or v >= H:
                    continue
                cur_depth = depth[v, u]
                if cur_depth >= 0.001 and cur_depth < zfar:
                    num_valid += 1
                    mean_depth += cur_depth
        if num_valid == 0:
            return
        mean_depth /= float(num_valid)

        depthCenter = depth[h, w]
        sum_weight = float(0.0)
        sum = float(0.0)
        for u in range(w - radius, w + radius + 1):
            if u < 0 or u >= W:
                continue
            for v in range(h - radius, h + radius + 1):
                if v < 0 or v >= H:
                    continue
                cur_depth = depth[v, u]
                if (
                    cur_depth >= 0.001
                    and cur_depth < zfar
                    and abs(cur_depth - mean_depth) < 0.01
                ):
                    weight = wp.exp(
                        -float((u - w) * (u - w) + (h - v) * (h - v))
                        / (2.0 * sigmaD * sigmaD)
                        - (depthCenter - cur_depth)
                        * (depthCenter - cur_depth)
                        / (2.0 * sigmaR * sigmaR)
                    )
                    sum_weight += weight
                    sum += weight * cur_depth
        if sum_weight > 0 and num_valid > 0:
            out[h, w] = sum / sum_weight

    def bilateral_filter_depth(
        depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device="cuda"
    ):
        if isinstance(depth, np.ndarray):
            depth_wp = wp.array(depth, dtype=float, device=device)
        else:
            depth_wp = wp.from_torch(depth)
        out_wp = wp.zeros(depth.shape, dtype=float, device=device)
        wp.launch(
            kernel=bilateral_filter_depth_kernel,
            device=device,
            dim=[depth.shape[0], depth.shape[1]],
            inputs=[depth_wp, out_wp, radius, zfar, sigmaD, sigmaR],
        )
        depth_out = wp.to_torch(out_wp)

        if isinstance(depth, np.ndarray):
            depth_out = depth_out.data.cpu().numpy()
        return depth_out

    @wp.kernel(enable_backward=False)
    def erode_depth_kernel(
        depth: wp.array(dtype=float, ndim=2),
        out: wp.array(dtype=float, ndim=2),
        radius: int,
        depth_diff_thres: float,
        ratio_thres: float,
        zfar: float,
    ):
        h, w = wp.tid()
        H = depth.shape[0]
        W = depth.shape[1]
        if w >= W or h >= H:
            return
        d_ori = depth[h, w]
        if d_ori < 0.001 or d_ori >= zfar:
            out[h, w] = 0.0
        bad_cnt = float(0)
        total = float(0)
        for u in range(w - radius, w + radius + 1):
            if u < 0 or u >= W:
                continue
            for v in range(h - radius, h + radius + 1):
                if v < 0 or v >= H:
                    continue
                cur_depth = depth[v, u]
                total += 1.0
                if (
                    cur_depth < 0.001
                    or cur_depth >= zfar
                    or abs(cur_depth - d_ori) > depth_diff_thres
                ):
                    bad_cnt += 1.0
        if bad_cnt / total > ratio_thres:
            out[h, w] = 0.0
        else:
            out[h, w] = d_ori

    def erode_depth(
        depth,
        radius=2,
        depth_diff_thres=0.001,
        ratio_thres=0.8,
        zfar=100,
        device="cuda",
    ):
        depth_wp = wp.from_torch(
            torch.as_tensor(depth, dtype=torch.float, device=device)
        )
        out_wp = wp.zeros(depth.shape, dtype=float, device=device)
        wp.launch(
            kernel=erode_depth_kernel,
            device=device,
            dim=[depth.shape[0], depth.shape[1]],
            inputs=[depth_wp, out_wp, radius, depth_diff_thres, ratio_thres, zfar],
        )
        depth_out = wp.to_torch(out_wp)

        if isinstance(depth, np.ndarray):
            depth_out = depth_out.data.cpu().numpy()
        return depth_out


def depth2xyzmap_batch(depths, Ks, zfar):
    """
    @depths: torch tensor (B,H,W)
    @Ks: torch tensor (B,3,3)
    """
    bs = depths.shape[0]
    invalid_mask = (depths < 0.001) | (depths > zfar)
    H, W = depths.shape[-2:]
    vs, us = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij")
    vs = vs.reshape(-1).float().cuda()[None].expand(bs, -1)
    us = us.reshape(-1).float().cuda()[None].expand(bs, -1)
    zs = depths.reshape(bs, -1)
    Ks = Ks[:, None].expand(bs, zs.shape[-1], 3, 3)
    xs = (us - Ks[..., 0, 2]) * zs / Ks[..., 0, 0]  # (B,N)
    ys = (vs - Ks[..., 1, 2]) * zs / Ks[..., 1, 1]
    pts = torch.stack([xs, ys, zs], dim=-1)  # (B,N,3)
    xyz_maps = pts.reshape(bs, H, W, 3)
    xyz_maps[invalid_mask] = 0
    return xyz_maps





def depth_to_vis(depth, zmin=None, zmax=None, mode="rgb", inverse=True):
    if zmin is None:
        zmin = depth.min()
    if zmax is None:
        zmax = depth.max()

    if inverse:
        invalid = depth < 0.001
        vis = zmin / (depth + 1e-8)
        vis[invalid] = 0
    else:
        depth = depth.clip(zmin, zmax)
        invalid = (depth == zmin) | (depth == zmax)
        vis = (depth - zmin) / (zmax - zmin)
        vis[invalid] = 1

    if mode == "gray":
        vis = (vis * 255).clip(0, 255).astype(np.uint8)
    elif mode == "rgb":
        vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)[
            ..., ::-1
        ]
    else:
        raise RuntimeError

    return vis


def sample_views_icosphere(n_views: int,
                           subdivisions: int | None = None,
                           radius: float = 1) -> np.ndarray:
    """ 
    Sample camera poses uniformly on an icosphere.

    Parameters
    ----------
    n_views : int
        Number of views to sample
    subdivisions : int | None, optional
        Number of subdivisions for the icosphere, by default None
    radius : float, optional
        Radius of the icosphere, by default 1

    Returns
    -------
    np.ndarray
        Array of camera poses centered arround the origin (N, 4, 4)
    """
    if subdivisions is not None:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    else:
        subdivision = 1
        while 1:
            mesh = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
            if mesh.vertices.shape[0] >= n_views:
                break
            subdivision += 1
    cam_in_obs = np.tile(np.eye(4)[None], (len(mesh.vertices), 1, 1))
    cam_in_obs[:, :3, 3] = mesh.vertices
    up = np.array([0, 0, 1])
    z_axis = -cam_in_obs[:, :3, 3]  # (N,3)
    z_axis /= np.linalg.norm(z_axis, axis=-1).reshape(-1, 1)
    x_axis = np.cross(up.reshape(1, 3), z_axis)
    invalid = (x_axis == 0).all(axis=-1)
    x_axis[invalid] = [1, 0, 0]
    x_axis /= np.linalg.norm(x_axis, axis=-1).reshape(-1, 1)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=-1).reshape(-1, 1)
    cam_in_obs[:, :3, 0] = x_axis
    cam_in_obs[:, :3, 1] = y_axis
    cam_in_obs[:, :3, 2] = z_axis
    return cam_in_obs


def to_homo(pts):
    """
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def to_homo_torch(pts):
    """
    @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
    """
    ones = torch.ones((*pts.shape[:-1], 1), dtype=torch.float, device=pts.device)
    homo = torch.cat((pts, ones), dim=-1)
    return homo


def transform_pts(pts: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """ Transform 2d or 3d points
    @pts: (...,N_pts,3)
    @tf: (...,4,4)

    Parameters
    ----------
    pts : torch.Tensor
        Points to transform, (...,N_pts,3)
    tf : torch.Tensor
        Transformation matrices, (...,4,4)

    Returns
    -------
    _type_
        _description_
    """
    
    if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]


def transform_dirs(dirs, tf):
    """
    @dirs: (...,3)
    @tf: (...,4,4)
    """
    if len(tf.shape) >= 3 and tf.shape[-3] != dirs.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :3, :3] @ dirs[..., None])[..., 0]


def compute_crop_window_tf_batch(
    poses: torch.Tensor,
    K: np.ndarray,
    crop_ratio=1.2,
    out_size=None,
    mesh_diameter=None,
) -> torch.Tensor:
    """
    Project the points and find the cropping transform

    Parameters
    ----------
    poses : torch.Tensor
        ob_in_cams poses (B,4,4)
    K : np.ndarray
        Camera intrinsic matrix
    crop_ratio : float, optional
        Scale to apply to the tightly enclosing roi, by default 1.2
    out_size : _type_, optional
        The desired size of the final crop, by default None
    mesh_diameter : _type_, optional
        The maximum diameter of the mesh, by default None

    Returns
    -------
    torch.Tensor
        The batch of transformation matrices to crop and resize the images. (B,3,3)
    """
    # Print types and shapes for debugging

    def compute_tf_batch(left, right, top, bottom):
        B = len(left)
        left = left.round()
        right = right.round()
        top = top.round()
        bottom = bottom.round()

        tf = torch.eye(3)[None].expand(B, -1, -1).contiguous()
        tf[:, 0, 2] = -left
        tf[:, 1, 2] = -top
        new_tf = torch.eye(3)[None].expand(B, -1, -1).contiguous()
        new_tf[:, 0, 0] = out_size[0] / (right - left)
        new_tf[:, 1, 1] = out_size[1] / (bottom - top)
        tf = new_tf @ tf
        return tf

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    B = len(poses)
    K_tensor = torch.as_tensor(K, dtype=torch.float, device=poses.device)

    # Set default tensor type to match poses
    
    radius = mesh_diameter * crop_ratio / 2 # in meter
    offsets = torch.tensor(
        [0, 0, 0, radius, 0, 0, -radius, 0, 0, 0, radius, 0, 0, -radius, 0]
    ).reshape(-1, 3) # (5, 3)
    
    # Get Mesh center point and bbox corners in camera space
    pts = poses[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3)
    # Project points to image plane
    projected = (K_tensor @ pts.reshape(-1, 3).T).T # (B*5, 3)
    # Normalize to get pixel coordinates
    uvs = projected[:, :2] / projected[:, 2:3]
    uvs = uvs.reshape(B, -1, 2)
    center = uvs[:, 0]  # (B,2)
    radius = torch.abs(uvs - center.reshape(-1, 1, 2)).reshape(B, -1).max(axis=-1)[0].reshape(-1)  # (B)
    
    # Box corners in pixel coordinates
    left = center[:, 0] - radius
    right = center[:, 0] + radius
    top = center[:, 1] - radius
    bottom = center[:, 1] + radius
    tfs = compute_tf_batch(left, right, top, bottom)
    return tfs



def cv_draw_text(
    img,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=None,
    line_spacing=1.5,
):
    H, W = img.shape[:2]
    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]

        ############# Ensure inside image
        while uv_bottom_left_i[0] < 0:
            uv_bottom_left_i[0] += 1
        while uv_bottom_left_i[0] + w >= W:
            uv_bottom_left_i[0] -= 1
        while uv_bottom_left_i[1] >= H:
            uv_bottom_left_i[1] -= 1
        while uv_bottom_left_i[1] - h < 0:
            uv_bottom_left_i[1] += 1

        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        uv_top_left[1] = uv_bottom_left_i[1] - h + h * line_spacing
    return img


def trimesh_add_pure_colored_texture(
    mesh, color=np.array([255, 255, 255]), resolution=5
):
    tex_img = np.tile(color.reshape(1, 1, 3), (resolution, resolution, 1)).astype(
        np.uint8
    )
    mesh = mesh.unwrap()
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=mesh.visual.uv, image=Image.fromarray(tex_img)
    )
    return mesh


def project_3d_to_2d(pt, K, ob_in_cam):
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def draw_xyz_axis(
    color,
    ob_in_cam,
    scale=0.1,
    K=np.eye(3),
    thickness=3,
    transparency=0,
    is_input_rgb=False,
):
    """
    @color: BGR
    """
    if is_input_rgb:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    xx = np.array([1, 0, 0, 1]).astype(float)
    yy = np.array([0, 1, 0, 1]).astype(float)
    zz = np.array([0, 0, 1, 1]).astype(float)
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale
    origin = tuple(project_3d_to_2d(np.array([0, 0, 0, 1]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = color.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        xx,
        color=(0, 0, 255),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        yy,
        color=(0, 255, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        zz,
        color=(255, 0, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    return tmp


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0, 255, 0), linewidth=2):
    """Revised from 6pack dataset/inference_dataset_nocs.py::projection
    @bbox: (2,3) min/max
    @line_color: RGB
    """
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def draw_line3d(start, end, img):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts = (ob_in_cam @ to_homo(pts).T).T[:, :3]  # (2,3)
        projected = (K @ pts.T).T
        uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(
            int
        )  # (2,2)
        img = cv2.line(
            img,
            uv[0].tolist(),
            uv[1].tolist(),
            color=line_color,
            thickness=linewidth,
            lineType=cv2.LINE_AA,
        )
        return img

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start + np.array([xmax - xmin, 0, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start + np.array([0, ymax - ymin, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start + np.array([0, 0, zmax - zmin])
            img = draw_line3d(start, end, img)

    return img


def projection_matrix_from_intrinsics(
    K, height, width, znear, zfar, window_coords="y_down"
):
    """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

    Ref:
    1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
    2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param x0 The X coordinate of the camera image origin (typically 0).
    :param y0: The Y coordinate of the camera image origin (typically 0).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: 4x4 ndarray with the OpenGL projection matrix.
    """
    x0 = 0
    y0 = 0
    w = width
    h = height
    nc = znear
    fc = zfar

    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same.
    if window_coords == "y_up":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )

    # Draw the images upright and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords.
    elif window_coords == "y_down":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )
    else:
        raise NotImplementedError

    return proj


def symmetry_tfs_from_info(info, rot_angle_discrete=5):
    symmetry_tfs = [np.eye(4)]
    if "symmetries_discrete" in info:
        tfs = np.array(info["symmetries_discrete"]).reshape(-1, 4, 4)
        tfs[..., :3, 3] *= 0.001
        symmetry_tfs = [np.eye(4)]
        symmetry_tfs += list(tfs)
    if "symmetries_continuous" in info:
        axis = np.array(info["symmetries_continuous"][0]["axis"]).reshape(3)
        offset = info["symmetries_continuous"][0]["offset"]
        rxs = [0]
        rys = [0]
        rzs = [0]
        if axis[0] > 0:
            rxs = np.arange(0, 360, rot_angle_discrete) / 180.0 * np.pi
        elif axis[1] > 0:
            rys = np.arange(0, 360, rot_angle_discrete) / 180.0 * np.pi
        elif axis[2] > 0:
            rzs = np.arange(0, 360, rot_angle_discrete) / 180.0 * np.pi
        for rx in rxs:
            for ry in rys:
                for rz in rzs:
                    tf = euler_matrix(rx, ry, rz)
                    tf[:3, 3] = offset
                    symmetry_tfs.append(tf)
    if len(symmetry_tfs) == 0:
        symmetry_tfs = [np.eye(4)]
    symmetry_tfs = np.array(symmetry_tfs)
    return symmetry_tfs





def egocentric_delta_pose_to_pose(A_in_cam, trans_delta, rot_mat_delta):
    """Used for Pose Refinement. Given the object's two poses in camera, convert them to relative poses in camera's egocentric view
    @A_in_cam: (B,4,4) torch tensor
    """
    B_in_cam = (
        torch.eye(4, dtype=torch.float, device=A_in_cam.device)[None]
        .expand(len(A_in_cam), -1, -1)
        .contiguous()
    )
    B_in_cam[:, :3, 3] = A_in_cam[:, :3, 3] + trans_delta
    B_in_cam[:, :3, :3] = rot_mat_delta @ A_in_cam[:, :3, :3]
    return B_in_cam







def transform_depth_to_xyzmap_utils(batch, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    H, W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda() / 2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()  # (B,3,3)
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()


    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    invalid = batch.xyz_mapAs[:, 2:3] < 0.1
    batch.xyz_mapAs = batch.xyz_mapAs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
    
    batch.xyz_mapAs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
    invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapAs) >= 2)
    batch.xyz_mapAs[invalid.expand(bs, 3, -1, -1)] = 0



    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    invalid = batch.xyz_mapBs[:, 2:3] < 0.1
    batch.xyz_mapBs = batch.xyz_mapBs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
    
    batch.xyz_mapBs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
    invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapBs) >= 2)
    batch.xyz_mapBs[invalid.expand(bs, 3, -1, -1)] = 0

    return batch

@torch.inference_mode()
def make_crop_data_batch_utils(
    render_size: Tuple[int, int],
    ob_in_cams: torch.Tensor,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    K: np.ndarray,
    crop_ratio: float,
    xyz_map: torch.Tensor,
    mesh_tensors,
    cfg=None,
    glctx=None,
):
    """ 

    Parameters
    ----------
    render_size : Tuple[int, int]
        The input size of image (H, W) passed to the network
    ob_in_cams : torch.Tensor
        (B,4,4) object poses in camera frame
    rgb : torch.Tensor
        The RGB image (H,W,3)
    depth : torch.Tensor
        The depth image (H,W)
    K : np.ndarray
        The camera intrinsic matrix (3,3)
    crop_ratio : float
        The ratio to crop the image around the object
    xyz_map : torch.Tensor
        Depth converted point cloud map (H,W,3)
    cfg : _type_, optional
        _description_, by default None
    glctx : _type_, optional
        _description_, by default None
    mesh_tensors : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    
    H, W = depth.shape[:2]
    B = len(ob_in_cams)
    poseA = ob_in_cams.clone()
    
    # Compute cropping transforms
    tf_to_crops = compute_crop_window_tf_batch(
        poses=ob_in_cams,
        K=K,
        crop_ratio=crop_ratio,
        out_size=(render_size[1], render_size[0]),
        mesh_diameter=mesh_tensors["diameter"],
    )
    
    # Prepare 2D bounding boxes for rendering
    bbox2d_crop = torch.as_tensor(
        np.array(
            [0, 0, cfg["input_resize"][0] - 1, cfg["input_resize"][1] - 1]
        ).reshape(2, 2),
        device="cuda",
        dtype=torch.float,
    )
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1, 4)

    bs = 512
    rgb_rs = []
    depth_rs = []
    normal_rs = []
    xyz_map_rs = []

    for b in range(0, len(poseA), bs):
        extra = {}
        rgb_r, depth_r, normal_r = nvdiffrast_render(
            K=K,
            H=H,
            W=W,
            ob_in_cams=poseA[b : b + bs],
            context="cuda",
            get_normal=cfg["use_normal"],
            glctx=glctx,
            mesh_tensors=mesh_tensors,
            output_size=cfg["input_resize"],
            bbox2d=bbox2d_ori[b : b + bs],
            use_light=True,
            extra=extra,
        )
        rgb_rs.append(rgb_r)
        depth_rs.append(depth_r[..., None])
        normal_rs.append(normal_r)
        xyz_map_rs.append(extra["xyz_map"])
        
    rgb_rs = torch.cat(rgb_rs, dim=0).permute(0, 3, 1, 2) * 255 # (num_views, 3, H, W)
    depth_rs = torch.cat(depth_rs, dim=0).permute(0, 3, 1, 2)  # (B,1,H,W)
    xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0, 3, 1, 2)  # (B,3,H,W)
    Ks = torch.as_tensor(K, device="cuda", dtype=torch.float).reshape(1, 3, 3)

    rgbBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(rgb, dtype=torch.float, device="cuda")
        .permute(2, 0, 1)[None]
        .expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="bilinear",
        align_corners=False,
    )
    depthBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(depth, dtype=torch.float, device="cuda")[None, None].expand(
            B, -1, -1, -1
        ),
        tf_to_crops,
        dsize=render_size,
        mode="nearest",
        align_corners=False,
    )
    xyz_mapBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(xyz_map, device="cuda", dtype=torch.float)
        .permute(2, 0, 1)[None]
        .expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="nearest",
        align_corners=False,
    )  # (B,3,H,W)
    
    rgbAs = rgb_rs
    depthAs = depth_rs
    xyz_mapAs = xyz_map_rs


    mesh_diameters = (
        torch.ones((len(rgbAs)), dtype=torch.float, device="cuda") * mesh_tensors["diameter"]
    )
    pose_data = BatchPoseData(
        rgbAs=rgbAs.float() / 255.0,
        rgbBs=rgbBs.float() / 255.0,
        depthAs=depthAs,
        depthBs=depthBs,
        poseA=poseA,
        poseB=None,
        xyz_mapAs=xyz_mapAs,
        xyz_mapBs=xyz_mapBs,
        tf_to_crops=tf_to_crops,
        Ks=Ks,
        mesh_diameters=mesh_diameters,
    )
    pose_data = transform_depth_to_xyzmap_utils(batch=pose_data, H_ori=H, W_ori=W, bound=1)

    return pose_data

@dataclass
class BatchPoseData:
    """
    rgbs: (bsz, 3, h, w) torch tensor uint8
    depths: (bsz, h, w) float32
    bboxes: (bsz, 4) int
    K: (bsz, 3, 3) float32
    """

    rgbs: torch.Tensor = None
    object_datas = None
    bboxes: torch.Tensor = None
    K: torch.Tensor = None
    depths: Optional[torch.Tensor] = None
    rgbAs = None
    rgbBs = None
    depthAs = None
    depthBs = None
    normalAs = None
    normalBs = None
    poseA = None  #(B,4,4)
    poseB = None
    targets = None  # Score targets, torch tensor (B)
    
    def __init__(self, rgbAs=None, rgbBs=None, depthAs=None, depthBs=None, normalAs=None, normalBs=None, maskAs=None, maskBs=None, poseA=None, poseB=None, xyz_mapAs=None, xyz_mapBs=None, tf_to_crops=None, Ks=None, crop_masks=None, model_pts=None, mesh_diameters=None, labels=None):
        self.rgbAs = rgbAs
        self.rgbBs = rgbBs
        self.depthAs = depthAs
        self.depthBs = depthBs
        self.normalAs = normalAs
        self.normalBs = normalBs
        self.poseA = poseA
        self.poseB = poseB
        self.maskAs = maskAs
        self.maskBs = maskBs
        self.xyz_mapAs = xyz_mapAs
        self.xyz_mapBs = xyz_mapBs
        self.tf_to_crops = tf_to_crops
        self.crop_masks = crop_masks
        self.Ks = Ks
        self.model_pts = model_pts
        self.mesh_diameters = mesh_diameters
        self.labels = labels


def rotation_geodesic_distance(R1, R2):
    cos = (np.trace(R1 @ R2.T) - 1) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def cluster_poses(angle_diff, dist_diff, poses_in, symmetry_tfs):
    print(f"num original candidates = {len(poses_in)}")
    poses_out = []
    if len(poses_in) == 0:
        return poses_out

    poses_out.append(poses_in[0])

    radian_thres = angle_diff / 180.0 * np.pi

    for i in range(1, len(poses_in)):
        isnew = True
        cur_pose = poses_in[i]
        for cluster in poses_out:
            t0 = cluster[0:3, 3]
            t1 = cur_pose[0:3, 3]

            if np.linalg.norm(t0 - t1) >= dist_diff:
                continue

            for tf in symmetry_tfs:
                cur_pose_tmp = cur_pose @ tf
                rot_diff = rotation_geodesic_distance(cur_pose_tmp[0:3, 0:3], cluster[0:3, 0:3])
                if rot_diff < radian_thres:
                    isnew = False
                    break

            if not isnew:
                break

        if isnew:
            poses_out.append(poses_in[i])

    print(f"num of pose after clustering: {len(poses_out)}")
    # Convert to np.array
    poses_out = np.stack(poses_out, axis=0)
    
    return poses_out



def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )
    
def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h
        
def so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)
    
    R = axis_angle_to_matrix(log_rot)

    return R
