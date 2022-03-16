import argparse
from omegaconf import OmegaConf
from sklearn.metrics import pairwise_distances_argmin
from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils import SaverRestore
from model import Model
from utils import SMPLModel, sample_vertex_from_mesh
import os
from tqdm import tqdm
import numpy as np
from eval_iou import KeypointDataset, name2id, nms_usip
import open3d as o3d
if os.name == 'nt':
    from sdv_src.build.Release import sdv
else:
    from sdv_src.build import sdv


def gen_mesh_with_color(pc, kp_idxs, color, kp_scale=2):
    prim_verts = None
    mesh_spheres = []
    res = 4
    
    colors = []
    for j, pt in enumerate(pc):
        if j in kp_idxs:
            continue
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=res)
        mesh_sphere.translate(pt)
        mesh_sphere.paint_uniform_color(color[j])
        mesh_spheres.append(mesh_sphere)
        if prim_verts is None:
            prim_verts = np.asarray(mesh_sphere.vertices).shape[0]
        
        colors.append(color[j])
        
    for idx in kp_idxs:
        pt = pc[idx]
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01 * kp_scale, resolution=res)
        mesh_sphere.translate(pt)
        mesh_sphere.paint_uniform_color(color[idx])
        mesh_spheres.append(mesh_sphere)
        
        colors.append(color[idx])
    
    # save to file
    vertices = []
    triangles = []
    total_vertices = 0
    for mesh in mesh_spheres:
        triangles.append(np.asarray(mesh.triangles) + total_vertices)
        vertices.append(np.asarray(mesh.vertices))
        total_vertices += np.asarray(mesh.vertices).shape[0]
        
    vertices = np.concatenate(vertices)
    triangles = np.concatenate(triangles)
        
    merged_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    
    mesh_color = np.zeros((vertices.shape[0], 3))
    colors = np.stack(colors, 0)
    for j in range(pc.shape[0]):
        mesh_color[j * prim_verts:(j + 1) * prim_verts] = colors[j]
    
    merged_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_color.reshape(-1, 3))
    return merged_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='shapenet', choices=['shapenet', 'smpl', '3dmatch'], help='Which type of evaluation (ShapeNet/SMPL/Real Scene)?')
    parser.add_argument('--kpnet_root', default='/kpnet/root', help='KeypointNet data root')
    parser.add_argument('--cat_name', default='chair', help='ShapeNet category name')
    parser.add_argument('--nms', action='store_true', help='Whether to use NMS')
    parser.add_argument('--nms_radius', type=float, default=0.2, help='NMS radius')
    parser.add_argument('--kp_num', type=int, default=-1, help='Whether to force a fixed number of keypoints (defaults: -1, no limitation)')
    args = parser.parse_args()
    
    if args.type == 'shapenet':
        model_path = 'outputs/{}/tflogs/checkpoint'.format(args.cat_name)
    elif args.type == 'smpl':
        model_path = 'outputs/{}/tflogs/checkpoint'.format('smpl')
    elif args.type == '3dmatch':
        model_path = 'outputs/{}/tflogs/checkpoint'.format('universal')
    else:
        raise Exception('unknown type')

    cfg = OmegaConf.load('config/config.yaml')
    predictor = OfflinePredictor(config=PredictConfig(model=Model(cfg),
                                                      input_names=['pc', 'pc_feature'],
                                                      output_names=['encoder/z'],
                                                      session_init=SaverRestore(model_path)))

    if args.type == 'shapenet':
        test_dataset = KeypointDataset(args.kpnet_root, name2id[args.cat_name], 'splits/test.txt')
        mesh_name, pc, feature, label = test_dataset[np.random.randint(len(test_dataset))]
        prob = predictor(pc[None], feature[None])[0][0]
    elif args.type == 'smpl':
        smpl = SMPLModel('data/model.pkl')
        pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
        beta = (np.random.rand(*smpl.beta_shape) - 0.5) * 0.06
        trans = np.zeros(smpl.trans_shape)
        smpl.set_params(beta=beta, pose=pose, trans=trans)
        pc, _, idx, u, v = sample_vertex_from_mesh(smpl.verts, smpl.faces, num_samples=cfg.num_points)
        pc_feature = np.array(sdv.compute(pc)).astype(np.float32)
        prob = predictor(pc[None], pc_feature[None])[0][0]
    elif args.type == '3dmatch':
        pcd = o3d.io.read_point_cloud('data/3DMatch/kitchen/cloud_bin_0.ply')
        pc_origin = np.array(pcd.points)
        _, trace = pcd.voxel_down_sample_and_trace(
            3e-2, pcd.get_min_bound(), pcd.get_max_bound(), False
        )
        sp_idx = trace[:, 0]
        sp_idx = sp_idx[sp_idx >= 0]
        pc = pc_origin[sp_idx]
        pc_feature = np.array(sdv.compute(pc_origin, interest_point_idxs=sp_idx)).astype(np.float32)
        prob = []
        block_size = 2048
        for i in tqdm(range((pc.shape[0] - 1) // block_size + 1)):
            prob.append(predictor(pc[i*block_size:(i+1)*block_size][None], pc_feature[i*block_size:(i+1)*block_size][None])[0][0])
        prob = np.concatenate(prob)

    if args.nms:
        kp, prob = nms_usip(pc, prob, args.nms_radius)
        kp = kp[prob > 0.5]
        prob = prob[prob > 0.5]
    else:
        kp = pc[prob > 0.5]
        prob = prob[prob > 0.5]

    if args.kp_num > 0:
        kp = kp[np.argsort(prob)[-args.kp_num:]]
    
    kp_idx = pairwise_distances_argmin(kp, pc)
    color = np.repeat(np.array([[202 / 255., 202 / 255., 202 / 255.]]), pc.shape[0], 0)
    color[kp_idx] = np.array([[1., 0, 0]])
    mesh = gen_mesh_with_color(pc, kp_idx, color)
    o3d.visualization.draw_geometries([mesh])
