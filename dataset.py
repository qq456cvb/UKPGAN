from torch.utils.data import Dataset
import hydra
from glob import glob
import os
import numpy as np
from src_shot.build import shot


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


class ShapeNetDataset(Dataset):
    name2id = {
        'airplane': '02691156',
        'chair': '03001627',
        'table': '04379243'
    }
    def __init__(self, cfg, split_fn):
        self.cfg = cfg
        self.pcds = []
        self.mesh_names = []
        
        split_models = open(hydra.utils.to_absolute_path(split_fn)).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]
        
        for fn in glob(os.path.join(hydra.utils.to_absolute_path(cfg.data.pcd_root), self.name2id[cfg.cat_name], '*.pcd')):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue
            self.pcds.append(naive_read_pcd(fn)[0])
            self.mesh_names.append(model_id)

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):
        pc = self.pcds[idx]
        pc_feature = shot.compute(pc)
        pc_feature[np.isnan(pc_feature)] = 0
        return pc.astype(np.float32), pc_feature.astype(np.float32)