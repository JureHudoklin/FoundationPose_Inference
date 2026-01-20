# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import logging
import numpy as np
import torch
import torch.nn.functional as F
import uuid
import trimesh
import nvdiffrast.torch as dr

from .Utils import (
    compute_mesh_diameter,
    sample_views_icosphere,
    euler_matrix,
    cluster_poses,
    depth2xyzmap,
    depth2xyzmap_batch,
    erode_depth,
    bilateral_filter_depth,
    make_mesh_tensors,
    nvdiffrast_render,
    toOpen3dCloud,
    so3_exp_map,
)
from .predict_score import ScorePredictor
from .predict_pose_refine import PoseRefinePredictor


class FoundationPose:
    def __init__(
        self,
        mesh,
        scorer: ScorePredictor,
        refiner: PoseRefinePredictor,
        symmetry_tfs=None,
    ):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.mesh_ori = mesh

        self.reset_object(mesh=mesh, symmetry_tfs=symmetry_tfs)
        self.rot_grid = self._make_rotation_grid()
        self.pose_hypotheses = None

        self.glctx = dr.RasterizeCudaContext()

        self.scorer = scorer
        self.refiner = refiner


        self.pose_last = None  # Used for tracking; per the centered mesh

    def reset_object(
        self, mesh: trimesh.Trimesh, symmetry_tfs: torch.Tensor | None = None
    ) -> None:
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2

        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)

        self.vox_size = max(self.diameter / 20.0, 0.003)
        logging.info(f"self.diameter:{self.diameter}, vox_size:{self.vox_size}")
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # Deg
        pcd = toOpen3dCloud(mesh.vertices, normals=mesh.vertex_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)
        self.max_xyz = np.asarray(pcd.points).max(axis=0)
        self.min_xyz = np.asarray(pcd.points).min(axis=0)
        self.pts = torch.tensor(
            np.asarray(pcd.points), dtype=torch.float32, device="cuda"
        )
        self.normals = F.normalize(
            torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device="cuda"),
            dim=-1,
        )
        logging.info(f"self.pts:{self.pts.shape}")
        self.mesh_path = None
        self.mesh = mesh
        if self.mesh is not None:
            self.mesh_path = f"/tmp/{uuid.uuid4()}.obj"
            self.mesh.export(self.mesh_path)
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(
                symmetry_tfs, device="cuda", dtype=torch.float
            )

        logging.info("reset done")

    def get_tf_to_centered_mesh(self) -> torch.Tensor:
        """

        Returns
        -------
        torch.Tensor
            Transformation matrix to center the mesh (4, 4)
        """
        tf_to_center = torch.eye(4, dtype=torch.float, device="cuda")
        tf_to_center[:3, 3] = -torch.as_tensor(
            self.model_center, device="cuda", dtype=torch.float
        )
        return tf_to_center

    def _make_rotation_grid(self, min_n_views=40, inplane_step=60) -> torch.Tensor:
        """
        Make rotation grid by sampling views on icosphere and in-plane rotations

        Parameters
        ----------
        min_n_views : int, optional
            Number of views sampled on the icosphere, by default 40
        inplane_step : int, optional
            Step size in degrees for in-plane rotation, by default 60

        Returns
        -------
        torch.Tensor
            Generated rotation grid of shape (N, 4, 4)
        """

        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        rot_grid = []
        # For each view, sample in-plane rotations
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        # Cluster the rotations to remove duplicates due to symmetries
        rot_grid = np.asarray(rot_grid)
        rot_grid = cluster_poses(
            30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy()
        )
        rot_grid = np.asarray(rot_grid)

        return torch.as_tensor(rot_grid, device="cuda", dtype=torch.float)

    def generate_random_pose_hypo(
        self, K: np.ndarray, mask: np.ndarray, depth: np.ndarray
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        K : np.ndarray
            Camera intrinsic matrix of shape (3, 3)
        mask : np.ndarray
            Binary mask of shape (H, W)
        depth : np.ndarray
            Depth map of shape (H, W)

        Returns
        -------
        torch.Tensor
            Generated pose hypotheses of object, shape (N, 4, 4)
        """
        if self.rot_grid is None:
            self.rot_grid = self._make_rotation_grid()

        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth, mask, K)
        ob_in_cams[:, :3, 3] = torch.tensor(
            center, device="cuda", dtype=torch.float
        ).reshape(1, 3)
        return ob_in_cams

    def guess_translation(
        self, depth: np.ndarray, mask: np.ndarray, K: np.ndarray
    ) -> np.ndarray:
        """Guess translation based on depth and mask

        Parameters
        ----------
        depth : np.ndarray
            Depth map of shape (H, W)
        mask : np.ndarray
            Binary mask of shape (H, W)
        K : np.ndarray
            Camera intrinsic matrix of shape (3, 3)
        Returns
        -------
        np.ndarray
            Estimated translation vector of shape (3,)
        """
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            logging.info(f"mask is all zero")
            return np.zeros((3))

        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            logging.info(f"valid is empty")
            return np.zeros((3))

        zc = np.median(depth[valid])
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

        return center.reshape(3)

    def register(
        self,
        K: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        ob_mask: np.ndarray,
        iteration=5,
    ):
        """

        Parameters
        ----------
        K : np.ndarray
            A camera intrinsic matrix of shape (3, 3).
        rgb : np.ndarray
            An RGB image of shape (H, W, 3).
        depth : np.ndarray
            A depth image in meters of shape (H, W).
        ob_mask : np.ndarray
            An object mask of shape (H, W).
        iteration : int, optional
            Number of refinement iterations, by default 5

        Returns
        -------
        np.ndarray
            Estimated object pose as a 4x4 transformation matrix.
        """
        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")

        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            logging.info(f"valid too small, return")
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=K, depth=depth, mask=ob_mask)
        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device="cuda")

        xyz_map = depth2xyzmap(depth, K)
        
        poses, vis = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses.data.cpu().numpy(),
            xyz_map=xyz_map,
            glctx=self.glctx,
            iteration=iteration,
            get_vis=True,
        )

        scores, vis = self.scorer.predict(
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses.data.cpu().numpy(),
            xyz_map=xyz_map,
            mesh_tensors=self.mesh_tensors,
            mesh=self.mesh,
            glctx=self.glctx,
            get_vis=True,
        )

        ids = torch.as_tensor(scores).argsort(descending=True)
        scores = scores[ids]
        poses = poses[ids]

        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[0]
        self.best_id = ids[0]

        self.poses = poses
        self.scores = scores

        return best_pose.data.cpu().numpy()

    def track_one(self, rgb, depth, K, iteration, extra={}):
        if self.pose_last is None:
            logging.info("Please init pose by register first")
            raise RuntimeError

        depth = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")

        xyz_map = depth2xyzmap_batch(
            depth[None],
            torch.as_tensor(K, dtype=torch.float, device="cuda")[None],
            zfar=np.inf,
        )[0]

        pose, vis = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=self.pose_last.reshape(1, 4, 4).data.cpu().numpy(),
            xyz_map=xyz_map,
            glctx=self.glctx,
            iteration=iteration,
            get_vis=True,
        )

        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)
