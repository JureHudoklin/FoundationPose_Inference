# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Tuple
import logging
import os, sys
import time
import numpy as np
from omegaconf import OmegaConf
import torch
from pytorch3d.transforms import so3_exp_map

from .models.refine_network import RefineNet
from .Utils import (
    egocentric_delta_pose_to_pose,
    make_crop_data_batch_utils,
    depth_to_vis,
    make_grid_image,
    cv_draw_text,
)


class PoseRefinePredictor:
    def __init__(
        self,
    ):
        self.amp = True
        self.run_name = "2023-10-28-18-33-37"
        model_name = "model_best.pth"
        code_dir = os.path.dirname(os.path.realpath(__file__))
        ckpt_dir = f"{code_dir}/../../weights/{self.run_name}/{model_name}"

        self.cfg = OmegaConf.load(
            f"{code_dir}/../../weights/{self.run_name}/config.yml"
        )

        self.cfg["ckpt_dir"] = ckpt_dir
        self.cfg["enable_amp"] = True

        ########## Defaults, to be backward compatible
        if "use_normal" not in self.cfg:
            self.cfg["use_normal"] = False
        if "use_mask" not in self.cfg:
            self.cfg["use_mask"] = False
        if "use_BN" not in self.cfg:
            self.cfg["use_BN"] = False
        if "c_in" not in self.cfg:
            self.cfg["c_in"] = 4
        if "crop_ratio" not in self.cfg or self.cfg["crop_ratio"] is None:
            self.cfg["crop_ratio"] = 1.2
        if "n_view" not in self.cfg:
            self.cfg["n_view"] = 1
        if "zfar" not in self.cfg:
            self.cfg["zfar"] = 3

        if isinstance(self.cfg["zfar"], str) and "inf" in self.cfg["zfar"].lower():
            self.cfg["zfar"] = np.inf
        if "normal_uint8" not in self.cfg:
            self.cfg["normal_uint8"] = False
        logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

        self.model = RefineNet(cfg=self.cfg, c_in=self.cfg["c_in"]).cuda()

        logging.info(f"Using pretrained model from {ckpt_dir}")
        ckpt = torch.load(ckpt_dir)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)

        self.model.cuda().eval()
        logging.info("init done")
        self.last_trans_update = None
        self.last_rot_update = None

    @torch.inference_mode()
    def predict(
        self,
        rgb : np.ndarray,
        depth : np.ndarray,
        K : np.ndarray,
        ob_in_cams : np.ndarray,
        xyz_map : np.ndarray,
        mesh_tensors,
        mesh=None,
        glctx=None,
        iteration=5,
        get_vis=False,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        @rgb: np array (H,W,3)
        @ob_in_cams: np array (N,4,4)
        """
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        tf_to_center = np.eye(4)
        ob_centered_in_cams = ob_in_cams
        mesh_centered = mesh

        crop_ratio = self.cfg["crop_ratio"]
        bs = 512

        B_in_cams = torch.as_tensor(
            ob_centered_in_cams, device="cuda", dtype=torch.float
        )
        
        rgb_tensor = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        xyz_map_tensor = torch.as_tensor(xyz_map, device="cuda", dtype=torch.float)

        for _ in range(iteration):
            pose_data = make_crop_data_batch_utils(
                self.cfg.input_resize,
                B_in_cams,
                rgb_tensor,
                depth_tensor,
                K,
                crop_ratio=crop_ratio,
                xyz_map=xyz_map_tensor,
                cfg=self.cfg,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
            )
            B_in_cams = []
            for b in range(0, pose_data.rgbAs.shape[0], bs):
                A = torch.cat(
                    [
                        pose_data.rgbAs[b : b + bs].cuda(),
                        pose_data.xyz_mapAs[b : b + bs].cuda(),
                    ],
                    dim=1,
                ).float()
                B = torch.cat(
                    [
                        pose_data.rgbBs[b : b + bs].cuda(),
                        pose_data.xyz_mapBs[b : b + bs].cuda(),
                    ],
                    dim=1,
                ).float()
                with torch.cuda.amp.autocast(enabled=self.amp):
                    output = self.model(A, B)

                # Forward pass
                for k in output:
                    output[k] = output[k].float()
                
                # Post-process (normalize) the output
                trans_delta = output["trans"]

                rot_mat_delta = (
                    torch.tanh(output["rot"]) * self.cfg["rot_normalizer"]
                )
                rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0, 2, 1)

                trans_delta *= mesh_tensors["diameter"] / 2

                B_in_cam = egocentric_delta_pose_to_pose(
                    pose_data.poseA[b : b + bs],
                    trans_delta=trans_delta,
                    rot_mat_delta=rot_mat_delta,
                )
                B_in_cams.append(B_in_cam)

            B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams), 4, 4)

        B_in_cams_out = B_in_cams @ torch.tensor(
            tf_to_center[None], device="cuda", dtype=torch.float
        )
        torch.cuda.empty_cache()
        self.last_trans_update = trans_delta
        self.last_rot_update = rot_mat_delta
        logging.debug("Pose refinement done.")

        if get_vis:
            logging.debug("get_vis...")
            canvas = []
            padding = 2
            pose_data = make_crop_data_batch_utils(
                self.cfg.input_resize,
                torch.as_tensor(ob_centered_in_cams),
                rgb,
                depth,
                K,
                crop_ratio=crop_ratio,
                xyz_map=xyz_map_tensor,
                cfg=self.cfg,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
            )
            for id in range(0, len(B_in_cams)):
                rgbA_vis = (
                    (pose_data.rgbAs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
                )
                rgbB_vis = (
                    (pose_data.rgbBs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
                )
                row = [rgbA_vis, rgbB_vis]
                H, W = rgbA_vis.shape[:2]
                if pose_data.depthAs is not None:
                    depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H, W)
                    depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H, W)
                elif pose_data.xyz_mapAs is not None:
                    depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H, W)
                    depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H, W)
                zmin = min(depthA.min(), depthB.min())
                zmax = max(depthA.max(), depthB.max())
                depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
                depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
                row += [depthA_vis, depthB_vis]
                if pose_data.normalAs is not None:
                    pass
                row = make_grid_image(
                    row, nrow=len(row), padding=padding, pad_value=255
                )
                row = cv_draw_text(
                    row,
                    text=f"id:{id}",
                    uv_top_left=(10, 10),
                    color=(0, 255, 0),
                    fontScale=0.5,
                )
                canvas.append(row)
            canvas = make_grid_image(canvas, nrow=1, padding=padding, pad_value=255)

            pose_data = make_crop_data_batch_utils(
                self.cfg.input_resize,
                B_in_cams,
                rgb,
                depth,
                K,
                crop_ratio=crop_ratio,
                xyz_map=xyz_map_tensor,
                cfg=self.cfg,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
            )
            canvas_refined = []
            for id in range(0, len(B_in_cams)):
                rgbA_vis = (
                    (pose_data.rgbAs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
                )
                rgbB_vis = (
                    (pose_data.rgbBs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
                )
                row = [rgbA_vis, rgbB_vis]
                H, W = rgbA_vis.shape[:2]
                if pose_data.depthAs is not None:
                    depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H, W)
                    depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H, W)
                elif pose_data.xyz_mapAs is not None:
                    depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H, W)
                    depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H, W)
                zmin = min(depthA.min(), depthB.min())
                zmax = max(depthA.max(), depthB.max())
                depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
                depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
                row += [depthA_vis, depthB_vis]
                row = make_grid_image(
                    row, nrow=len(row), padding=padding, pad_value=255
                )
                canvas_refined.append(row)

            canvas_refined = make_grid_image(
                canvas_refined, nrow=1, padding=padding, pad_value=255
            )
            canvas = make_grid_image(
                [canvas, canvas_refined], nrow=2, padding=padding, pad_value=255
            )
            torch.cuda.empty_cache()
            return B_in_cams_out, canvas

        return B_in_cams_out, None
