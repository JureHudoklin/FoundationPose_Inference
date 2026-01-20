# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


from .Utils import *
from .models.score_network import *


def vis_batch_data_scores(pose_data, ids, scores, pad_margin=5):
    assert len(scores) == len(ids)
    canvas = []
    for id in ids:
        rgbA_vis = (pose_data.rgbAs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
        H, W = rgbA_vis.shape[:2]
        zmin = pose_data.depthAs[id].data.cpu().numpy().reshape(H, W).min()
        zmax = pose_data.depthAs[id].data.cpu().numpy().reshape(H, W).max()
        depthA_vis = depth_to_vis(
            pose_data.depthAs[id].data.cpu().numpy().reshape(H, W),
            zmin=zmin,
            zmax=zmax,
            inverse=False,
        )
        depthB_vis = depth_to_vis(
            pose_data.depthBs[id].data.cpu().numpy().reshape(H, W),
            zmin=zmin,
            zmax=zmax,
            inverse=False,
        )
        if pose_data.normalAs is not None:
            pass
        pad = np.ones((rgbA_vis.shape[0], pad_margin, 3)) * 255
        if pose_data.normalAs is not None:
            pass
        else:
            row = np.concatenate(
                [rgbA_vis, pad, depthA_vis, pad, rgbB_vis, pad, depthB_vis], axis=1
            )
        s = 100 / row.shape[0]
        row = cv2.resize(row, fx=s, fy=s, dsize=None)
        row = cv_draw_text(
            row,
            text=f"id:{id}, score:{scores[id]:.3f}",
            uv_top_left=(10, 10),
            color=(0, 255, 0),
            fontScale=0.5,
        )
        canvas.append(row)
        pad = np.ones((pad_margin, row.shape[1], 3)) * 255
        canvas.append(pad)
    canvas = np.concatenate(canvas, axis=0).astype(np.uint8)
    return canvas


class ScorePredictor:
    def __init__(self, amp=True):
        self.amp = amp
        self.run_name = "2024-01-11-20-02-45"

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
        if "use_BN" not in self.cfg:
            self.cfg["use_BN"] = False
        if "zfar" not in self.cfg:
            self.cfg["zfar"] = np.inf
        if "c_in" not in self.cfg:
            self.cfg["c_in"] = 4
        if "normalize_xyz" not in self.cfg:
            self.cfg["normalize_xyz"] = False
        if "crop_ratio" not in self.cfg or self.cfg["crop_ratio"] is None:
            self.cfg["crop_ratio"] = 1.2

        logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

        self.model = ScoreNetMultiPair(cfg=self.cfg, c_in=self.cfg["c_in"]).cuda()

        logging.info(f"Using pretrained model from {ckpt_dir}")
        ckpt = torch.load(ckpt_dir)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)

        self.model.cuda().eval()
        logging.info("init done")

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
        get_vis=False,
    ):
        """
        @rgb: np array (H,W,3)
        """
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        
        ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device="cuda")
        rgb_tensor = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device="cuda", dtype=torch.float)

        pose_data = make_crop_data_batch_utils(
            self.cfg.input_resize,
            ob_in_cams,
            rgb_tensor,
            depth_tensor,
            K,
            crop_ratio=self.cfg["crop_ratio"],
            xyz_map=xyz_map,
            cfg=self.cfg,
            glctx=glctx,
            mesh_tensors=mesh_tensors,
        )

        pose_data_iter = pose_data
        global_ids = torch.arange(len(ob_in_cams), device="cuda", dtype=torch.long)
        scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device="cuda")
        
        ids = []
        scores = []
        bs = 512
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
                output = self.model(A, B, L=len(A))
            scores_cur = output["score_logit"].float().reshape(-1)
            ids.append(scores_cur.argmax() + b)
            scores.append(scores_cur)
            
        scores = torch.cat(scores, dim=0).reshape(-1)
        ids = scores.argmax(dim=0, keepdim=True)

        logging.debug("Score prediction done.")
        torch.cuda.empty_cache()

        if get_vis:
            logging.debug("get_vis...")
            canvas = []
            ids = scores.argsort(descending=True)
            canvas = vis_batch_data_scores(pose_data, ids=ids, scores=scores)
            return scores, canvas

        return scores, None
