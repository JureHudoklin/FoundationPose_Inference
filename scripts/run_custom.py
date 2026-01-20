import argparse
import os
import logging
import torch
from foundation_pose.Utils import set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis, nvdiffrast_render
from foundation_pose import FoundationPose, ScorePredictor, PoseRefinePredictor
import numpy as np
import cv2
import trimesh
import nvdiffrast.torch as dr

def parse_args():
    parser = argparse.ArgumentParser(description="Run FoundationPose on custom data")
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--est_refine_iter', type=int, default=5, help='Number of refinement iterations for estimation')
    parser.add_argument('--track_refine_iter', type=int, default=2, help='Number of refinement iterations for tracking')
    
    parser.add_argument('--mesh_file', type=str, help='Path to the CAD model (e.g. .obj file)', default=f'../data/model.obj')
    parser.add_argument('--rgb_file', type=str, help='Path to RGB image', default=f'../data/rgb.png')
    parser.add_argument('--depth_file', type=str, help='Path to Depth image', default=f'../data/depth.png')
    parser.add_argument('--mask_file', type=str, help='Path to Object Mask image', default=f'../data/mask_3.png')
    parser.add_argument('--pcd_file', type=str, help='Path to Point Cloud file (e.g. .ply)', default=f'../data/scene.ply')
    parser.add_argument('--cam_K_file', type=str, help='Path to text file containing 3x3 Camera Intrinsics matrix', default=f'../data/cam_K.txt')
    parser.add_argument('--depth_scale', type=float, default=1000.0, help='Scale factor to divide depth image by to get meters (e.g. 1000 if depth is in mm)')
    parser.add_argument('--resize_factor', type=float, default=1, help='Resize factor for images. e.g. 0.5 for half size')
    
    return parser.parse_args()

def render_object_on_image(mesh, pose, K, rgb, debug_dir):
    H, W = rgb.shape[:2]
    # Convert inputs to torch tensors
    ob_in_cam = torch.as_tensor(pose, dtype=torch.float32, device="cuda").unsqueeze(0)
    
    # Render
    color, depth, _ = nvdiffrast_render(mesh=mesh, ob_in_cams=ob_in_cam, K=K, H=H, W=W)
    
    # Post-process
    color = color[0].detach().cpu().numpy() # (H, W, 4)
    depth = depth[0].detach().cpu().numpy() # (H, W)
    
    # Rendered color (assuming RGB return from nvdiffrast_render as seen in Utils)
    render_rgb = (color[..., :3] * 255).astype(np.uint8)
    
    # Mask is where depth > 0 (assuming object is in front of camera)
    mask = (depth > 0).astype(np.uint8)
    
    vis = rgb.copy()
    alpha = 0.5
    
    # Overlay on RGB
    # We only modify pixels where the object is rendered
    vis[mask==1] = (alpha * vis[mask==1] + (1-alpha) * render_rgb[mask==1]).astype(np.uint8)
    
    save_path = f'{debug_dir}/render_vis.png'
    cv2.imwrite(save_path, vis[..., ::-1])
    print(f"Saved render visualization to {save_path}")

def main():
    args = parse_args()
    
    set_logging_format()
    set_seed(0)

    # Load Mesh
    if not os.path.exists(args.mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {args.mesh_file}")
    print(f"Loading mesh from {args.mesh_file}")
    mesh = trimesh.load(args.mesh_file)
    
    # Pre-process mesh bounds
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # Initialize Debug Directory
    debug_dir = "./debug_run_custom"
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}')
    
    # Initialize Predictors and Context
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    
    # Symetry transform (rotation around Z) Split into 50 steps
    symmetry_tfs = []
    for angle in np.linspace(0, 2*np.pi, 50, endpoint=False):
        c, s = np.cos(angle), np.sin(angle)
        tf = np.eye(4)
        tf[0, 0] = c
        tf[0, 1] = -s
        tf[1, 0] = s
        tf[1, 1] = c
        symmetry_tfs.append(tf)
    
    est = FoundationPose(mesh=mesh,
                         scorer=scorer,
                         refiner=refiner,
                         symmetry_tfs=symmetry_tfs)  # Assuming no symmetry for custom object
    logging.info("Estimator initialization done")
    
    # Load Image Data
    if not os.path.exists(args.rgb_file):
        raise FileNotFoundError(f"RGB file not found: {args.rgb_file}")
    rgb = cv2.imread(args.rgb_file, cv2.IMREAD_COLOR)
    print(f"Loaded RGB image from {rgb.shape} {args.rgb_file}")
    if rgb is None:
        raise RuntimeError(f"Failed to read RGB file: {args.rgb_file}")
    rgb = rgb[..., ::-1].copy() # BGR to RGB
    
    # Load Depth Data
    if not os.path.exists(args.depth_file):
        raise FileNotFoundError(f"Depth file not found: {args.depth_file}")
    depth_raw = cv2.imread(args.depth_file, -1)
    if depth_raw is None:
        raise RuntimeError(f"Failed to read Depth file: {args.depth_file}")
    depth = depth_raw / args.depth_scale
    depth[depth < 0.001] = 0
    
    # Load Mask
    if not os.path.exists(args.mask_file):
        raise FileNotFoundError(f"Mask file not found: {args.mask_file}")
    mask_in = cv2.imread(args.mask_file, -1)
    if mask_in is None:
        raise RuntimeError(f"Failed to read Mask file: {args.mask_file}")
        
    # Handle multi-channel mask if necessary
    if len(mask_in.shape) == 3:
        # Assuming the object mask is in one of the channels, or it's a binary mask replicated
        # If user provides a color mask, this might be tricky. Let's assume non-zero is the object.
        mask = (mask_in.sum(axis=-1) > 0).astype(np.uint8)
    else:
        mask = (mask_in > 0).astype(np.uint8)
    
    # Load Camera Intrinsics
    if not os.path.exists(args.cam_K_file):
         raise FileNotFoundError(f"Camera K file not found: {args.cam_K_file}")
    K = np.loadtxt(args.cam_K_file).reshape(3,3)

    if args.resize_factor != 1.0:
        h, w = rgb.shape[:2]
        new_w, new_h = int(w * args.resize_factor), int(h * args.resize_factor)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        K[:2, :] *= args.resize_factor
        logging.info(f"Resized images to {new_w}x{new_h} with factor {args.resize_factor}. Adjusted K matrix.")

    # Register (Estimate Pose)
    logging.info("Registering...")
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    poses = [pose]
    
    print("Estimated Pose:")
    print(pose)
    
    # Save Pose
    np.savetxt(f'{debug_dir}/pose.txt', pose)
    
    # Visualization
    if True:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
        
        vis_path = f'{debug_dir}/vis.png'
        cv2.imwrite(vis_path, vis[..., ::-1])
        print(f"Saved visualization to {vis_path}")
        
        render_object_on_image(mesh, pose, K, rgb, debug_dir)
        
        if False:
            # Save transformed mesh
            m = mesh.copy()
            m.apply_transform(pose)
            m.export(f'{debug_dir}/model_tf.obj')
            print(f"Saved transformed mesh to {debug_dir}/model_tf.obj")
            
        if False:
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.creation.axis(origin_size=0.05))
            
            # Load and add Point Cloud if exists
            if args.pcd_file and os.path.exists(args.pcd_file):
                logging.info(f"Loading point cloud from {args.pcd_file}")
                pcd = trimesh.load(args.pcd_file)
                # Scale point cloud if necessary
                depth_units = args.depth_scale
                
                # The user specified that PC is in Z- and predictions in Z+.
                # We interpret this as a need to transform the PC from OpenGL-like (looking down -Z) 
                # to OpenCV-like (looking down +Z) which often involves a 180 degree rotation around X.
                # This flips Y and Z.
                T_pcd = np.eye(4)
                T_pcd[1, 1] = -1
                T_pcd[2, 2] = -1
                pcd.apply_transform(T_pcd)
                
                scene.add_geometry(pcd)
            else:
                logging.warning(f"Point cloud file not found: {args.pcd_file}")

            for i, pose in enumerate(poses):
                m = mesh.copy()
                m.apply_transform(pose)
                # Set random color
                m.visual.face_colors = trimesh.visual.random_color()
                scene.add_geometry(m)
            
            # This opens a window
            scene.show()

if __name__ == '__main__':
    main()
