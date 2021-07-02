import numpy as np
import os
from glob import glob
import json
from sklearn.metrics import pairwise_distances_argmin
from sklearn import neighbors
from sklearn.utils.graph import graph_shortest_path
import pickle
from tqdm import tqdm
from tensorpack import *
from omegaconf import OmegaConf
from dataflow import load_sdv_feature
from sdv_src.build import sdv
import tensorflow as tf
from model import Model


# adapted from usip
def nms_usip(keypoints_np, sigmas_np, NMS_radius):
    '''

    :param keypoints_np: Mx3
    :param sigmas_np: M
    :return: valid_keypoints_np, valid_sigmas_np, valid_descriptors_np
    '''
    if NMS_radius < 0.01:
        return keypoints_np, sigmas_np

    valid_keypoint_counter = 0
    valid_keypoints_np = np.zeros(keypoints_np.shape, dtype=keypoints_np.dtype)
    valid_sigmas_np = np.zeros(sigmas_np.shape, dtype=sigmas_np.dtype)

    while keypoints_np.shape[0] > 0:
        # print(sigmas_np.shape)
        # print(sigmas_np)

        max_idx = np.argmax(sigmas_np, axis=0)
        # print(min_idx)

        valid_keypoints_np[valid_keypoint_counter, :] = keypoints_np[max_idx, :]
        valid_sigmas_np[valid_keypoint_counter] = sigmas_np[max_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = np.linalg.norm(
            (valid_keypoints_np[valid_keypoint_counter:valid_keypoint_counter + 1, :] - keypoints_np), axis=1,
            keepdims=False)  # M
        mask = distance_array > NMS_radius  # M

        keypoints_np = keypoints_np[mask, ...]
        sigmas_np = sigmas_np[mask]

        # increase counter
        valid_keypoint_counter += 1

    return valid_keypoints_np[0:valid_keypoint_counter, :], \
           valid_sigmas_np[0:valid_keypoint_counter]
           

def eval_det_cls(pred, gt, geo_dists, dist_thresh=0.1):
    npos = 0
    fp_sum = 0
    fn_sum = 0
    for mesh_name in gt.keys():
        gt_kps = np.array(gt[mesh_name]).astype(np.int32)
        npos += len(gt_kps)
        pred_kps = np.array(pred[mesh_name]).astype(np.int32)
        fp = np.count_nonzero(np.all(geo_dists[mesh_name][pred_kps][:, gt_kps] > dist_thresh, axis=-1))
        fp_sum += fp
        fn = np.count_nonzero(np.all(geo_dists[mesh_name][gt_kps][:, pred_kps] > dist_thresh, axis=-1))
        fn_sum += fn

    return (npos - fn_sum) / np.maximum(npos + fp_sum, np.finfo(np.float64).eps)


def eval_iou(pred_all, gt_all, geo_dists, dist_thresh=0.05):
    iou = {}
    for classname in gt_all.keys():
        iou[classname] = eval_det_cls(pred_all[classname], gt_all[classname], geo_dists, dist_thresh)

    return iou


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    colors = np.array(data[:, -1], dtype=np.int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors


class KeypointDataset:
    def __init__(self, root, cat_id, split):
        super().__init__()
            
        annots = json.load(open(os.path.join(root, 'annotations/all.json')))
        annots = [annot for annot in annots if annot['class_id'] == cat_id]
        keypoints = dict([(annot['model_id'], [kp_info['pcd_info']['point_index'] for kp_info in annot['keypoints']]) for annot in annots])
        
        split_models = open(os.path.join(root, split)).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]
        
        self.pcds = []
        self.keypoints = []
        self.mesh_names = []
        for fn in glob(os.path.join(root, 'pcds', cat_id, '*.pcd')):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue
            self.keypoints.append(keypoints[model_id])
            self.pcds.append(naive_read_pcd(fn)[0])
            self.mesh_names.append(model_id)

        self.nclasses = 2

    def __getitem__(self, idx):
        pc = self.pcds[idx]
        label = self.keypoints[idx]
        bin_label = np.zeros((pc.shape[0],), dtype=np.int64)
        bin_label[label] = 1
        
        mesh_name = self.mesh_names[idx]
        feature = load_sdv_feature(pc, os.path.join(cfg.data.feature_cache, mesh_name + '.npy'))

        return mesh_name, pc.astype(np.float32), feature, bin_label

    def __len__(self):
        return len(self.pcds)
    

def gen_geo_dists(pc):
    graph = neighbors.kneighbors_graph(pc, 20, mode='distance', include_self=False)
    return graph_shortest_path(graph, directed=False)

name2id = {
    'airplane': '02691156',
    'chair': '03001627',
    'table': '04379243'
}

if __name__ == "__main__":
    kpnet_root = '/kpnet/root'  # MODIFY this
    
    
    f = open('iou_test.txt', 'w')
    for cat_name in ['airplane', 'chair', 'table']:
        f.write(cat_name)
        f.write('\n')
        
        tf.reset_default_graph()
        test_dataset = KeypointDataset(kpnet_root, name2id[cat_name], 'splits/test.txt')
        
        cfg = OmegaConf.load('config/config.yaml')
        cfg.cat_name = cat_name
        
        model_path = os.path.join('outputs', cat_name, 'tflogs', 'checkpoint')
        
        predictor = OfflinePredictor(config=PredictConfig(model=Model(cfg), 
                                                        input_names=['pc', 'pc_feature'], 
                                                        output_names=['encoder/z'],
                                                        session_init=SaverRestore(model_path)))
        pred_all_iou = {
            cat_name: {}
        }
        gt_all = {
            cat_name: {}
        }
        
        for i in range(len(test_dataset.mesh_names)):
            mesh_name = test_dataset.mesh_names[i]
            if mesh_name not in pred_all_iou[cat_name]:
                pred_all_iou[cat_name][mesh_name] = []

            if mesh_name not in gt_all[cat_name]:
                gt_all[cat_name][mesh_name] = []
                
        for i, data in tqdm(enumerate(test_dataset)):
            mesh_name, pc, feature, label = data
            
            z = predictor(pc[None], feature[None])[0][0]
            kp, prob = nms_usip(pc, z, 0.1)
            prediction = kp[prob > 0.5]
            
            predict_idx = pairwise_distances_argmin(prediction, pc)
            pred_all_iou[cat_name][mesh_name].extend(predict_idx)
            
            for kp in np.where(label == 1)[0]:
                gt_all[cat_name][mesh_name].append(kp)

        
        BASEDIR = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(BASEDIR, 'cache')):
            os.makedirs(os.path.join(BASEDIR, 'cache'))
        if os.path.exists(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(cat_name))):
            print('Found geodesic cache...')
            geo_dists = pickle.load(open(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(cat_name)), 'rb'))
        else:
            geo_dists = {}
            print('Generating geodesics, this may take some time...')
            for i in tqdm(range(len(test_dataset.mesh_names))):
                if test_dataset.mesh_names[i] not in geo_dists:
                    geo_dists[test_dataset.mesh_names[i]] = gen_geo_dists(test_dataset.pcds[i]).astype(np.float32)
            pickle.dump(geo_dists, open(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(cat_name)), 'wb'))
            
        for i in range(11):
            dist_thresh = 0.01 * i
            iou = eval_iou(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)

            iou_l = list(iou.values())
            s = ""
            for x in iou_l:
                s += "{}\t".format(x)
            f.write('mIoU-{}: {}\n'.format(dist_thresh, s))
            print('mIoU-{}: {}'.format(dist_thresh, s))
            
        f.write('\n')
    f.close()