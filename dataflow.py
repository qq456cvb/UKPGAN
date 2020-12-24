import numpy as np
import sys
import os
from tensorpack.callbacks.base import Callback
from tensorpack.dataflow.base import RNGDataFlow
from sdv_src.build import sdv
from utils import naive_read_pcd
from tensorpack import *
from glob import glob
import visdom
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import hydra
import matplotlib.pyplot as plt
cm = plt.get_cmap('jet')


def load_sdv_feature(pc, path, caching):
    if not caching:
        return np.array(sdv.compute(pc)).astype(np.float32)

    if not os.path.exists(path):
        feature = np.array(sdv.compute(pc)).astype(np.float32)
        np.save(path, feature)
        return feature
    else:
        return np.load(path)
    
name2id = {
    'airplane': '02691156',
    'chair': '03001627',
    'table': '04379243'
}

class MyDataFlow(RNGDataFlow):
    def __init__(self, cfg, split, training):
        self.training = training
        self.cfg = cfg
        self.pcds = []
        self.mesh_names = []
        
        split_models = open(hydra.utils.to_absolute_path(split)).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]
        
        for fn in glob(os.path.join(hydra.utils.to_absolute_path(cfg.data.pcd_root), name2id[cfg.cat_name], '*.pcd')):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue
            self.pcds.append(naive_read_pcd(fn)[0])
            self.mesh_names.append(model_id)
        
        if self.cfg.caching and not os.path.exists(hydra.utils.to_absolute_path(self.cfg.data.feature_cache)):
            os.makedirs(hydra.utils.to_absolute_path(self.cfg.data.feature_cache))

    def _setup_graph(self):
        self.predictor = self.trainer.get_predictor(['pc', 'pc_feature'], ['encoder/z', 'encoder/feature', 'recon_pc'])

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):
        pc = self.pcds[idx]
        mesh_name = self.mesh_names[idx]
        feature = load_sdv_feature(pc, os.path.join(hydra.utils.to_absolute_path(self.cfg.data.feature_cache), mesh_name + '.npy'), caching=self.cfg.caching)

        return pc, feature

    def __iter__(self):
        shuffle_list = np.arange(len(self))
        if self.training:
            self.rng.shuffle(shuffle_list)
        for idx in shuffle_list:
            pc = self.pcds[idx]
            mesh_name = self.mesh_names[idx]
            feature = load_sdv_feature(pc, os.path.join(hydra.utils.to_absolute_path(self.cfg.data.feature_cache), mesh_name + '.npy'), caching=self.cfg.caching)

            yield pc, feature

            
class VisDataFlow(MyDataFlow, Callback):
    def __init__(self, *args):
        super().__init__(*args)
        self.vis = visdom.Visdom(port=1080)
    
    def _trigger(self):
        z = []
        emb = []
        recon = []
        pcs = []
        features = []
        for _ in range(2):
            idx = np.random.randint(len(self))
            pc = self.pcds[idx]
            mesh_name = self.mesh_names[idx]
            feature = load_sdv_feature(pc, os.path.join(self.cfg.data.feature_cache, mesh_name + '.npy'), caching=self.cfg.caching)
            
            output = self.predictor(pc[None], feature[None])
            
            z.append(output[0][0])
            emb.append(output[1][0])
            recon.append(output[2][0])
            
            pcs.append(pc)
            features.append(feature)
            
        prob = z[0] > 0.5
        label = np.ones((pcs[0].shape[0],), dtype=np.int)
        if np.sum(prob.astype(np.float32)) > 0:
            label[prob] = 2
        
        self.vis.scatter(
            X=pcs[0],
            Y=label,
            win=1,
            opts=dict(
                title='Detected keypoints'
            ),
        )
        
        self.vis.scatter(
            X=recon[0],
            win=2,
            opts=dict(
                markercolor=(z[0] * 255).astype(np.int),
                title='Reconstruction'
            ),
        )
        
        rgb = np.array([cm(i * 255)[:3] for i in z[0]])
        self.vis.scatter(
            X=pcs[0],
            win=3,
            opts=dict(
                markercolor=(rgb * 255).astype(np.int),
                title='Keypoint Probability'
            ),
        )
        
        pca = PCA(n_components=3)
        scaler = MinMaxScaler()
        rgb = pca.fit_transform(emb[0])
        rgb = scaler.fit_transform(rgb)
        self.vis.scatter(
            X=pcs[0],
            win=4,
            opts=dict(
                markercolor=(rgb * 255).astype(np.int),
                title='Embedding Prediction (Model 1)'
            ),
        )
        
        rgb = pca.transform(emb[1])
        rgb = np.clip(scaler.transform(rgb), 0., 1.)
        self.vis.scatter(
            X=pcs[1],
            win=5,
            opts=dict(
                markercolor=(rgb * 255).astype(np.int),
                title='Embedding Prediction (Model 2)'
            ),
        )