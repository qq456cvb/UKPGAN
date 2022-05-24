import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import get_arch, mlp, mlp_conv, query_ball_point, rifeat



class Discriminator(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.fcs = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.fcs.append(
                nn.Sequential(
                   nn.Linear(chs[i], chs[i + 1]),
                   nn.ReLU() 
                )
            )
        
    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        return x


class Encoder(nn.Module):
    def __init__(self, radius, num_layers, ch_mid, ch_out):
        super().__init__()
        self.radius = radius
        self.layers = nn.ModuleList()
        self.layer_in = nn.Sequential(
            nn.Linear(6, ch_mid),
            nn.LeakyReLU(),
            nn.Linear(ch_mid, ch_mid),
            # nn.LeakyReLU(),
            # nn.Linear(ch_mid, ch_mid),
            # nn.LeakyReLU(),
            # nn.Linear(ch_mid, ch_mid),
            # nn.LeakyReLU(),
            # nn.Linear(ch_mid, ch_mid)
        )
        self.layer_out = nn.Linear(ch_mid, ch_out)
        self.trans = nn.ModuleList()
        for _ in range(num_layers): 
            self.layers.append(
                nn.Sequential(
                    nn.Linear(ch_mid, ch_mid),
                    nn.LeakyReLU(),
                    nn.Linear(ch_mid, ch_mid),
                    # nn.LeakyReLU(),
                    # nn.Linear(ch_mid, ch_mid),
                    # nn.LeakyReLU(),
                    # nn.Linear(ch_mid, ch_mid),
                    # nn.LeakyReLU(),
                    # nn.Linear(ch_mid, ch_mid)
                )
            )
            self.trans.append(nn.Linear(2 * ch_mid, ch_mid))
    
    def forward(self, pc):
        nbr_idx = query_ball_point(self.radius, 128, pc, pc)  # B x N x K
        
        nbr_pts = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2, nbr_idx[..., None].expand(*nbr_idx.shape, pc.shape[-1]))
        nbr_feats = rifeat(nbr_pts, pc[:, :, None])
        feat = torch.mean(self.layer_in(nbr_feats), -2)
        for i, layer in enumerate(self.layers):
            nbr_feats = torch.gather(feat.unsqueeze(-3).expand(*feat.shape[:-1], *feat.shape[-2:]), -2, nbr_idx[..., None].expand(*nbr_idx.shape, feat.shape[-1]))
            nbr_feats = layer(nbr_feats)
            feat_max = nbr_feats.mean(-2)
            feat = self.trans[i](torch.cat([feat, feat_max], -1))
        feat = self.layer_out(feat)
        return feat


class TopNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nfeat = cfg.topnet.nfeat
        self.code_nfts = cfg.topnet.code_nfts
        self.nin = cfg.topnet.nfeat + cfg.topnet.code_nfts
        self.nout = cfg.topnet.nfeat
        self.tarch = get_arch(cfg.topnet.nlevels, cfg.num_points)
        self.npoints = cfg.num_points
        level0 = nn.Sequential(
            mlp(self.code_nfts, [256, 64, self.nfeat * int(self.tarch[0])], bn=True),
            nn.Tanh()
        )
        self.levels = nn.ModuleList([level0])
        nin = self.nin
        for i in range(1, len(self.tarch)):
            if i == len(self.tarch) - 1:
                nout = 3
                bn = False
            else:
                nout = self.nout
                bn = True
                
            level = nn.Sequential(
                self.create_level(i, nin, nout, bn),
                nn.Tanh()
            )
            self.levels.append(level)
            # nin = nout * int(self.tarch[i]) + self.code_nfts
            # print(nin, nout * int(self.tarch[i]))
    
    def create_level(self, level, input_channels, output_channels, bn):
        return mlp_conv(input_channels, [input_channels, int(input_channels / 2),
                                    int(input_channels / 4), int(input_channels / 8),
                                    output_channels * int(self.tarch[level])], bn)
    
    def forward(self, code : torch.Tensor):
        nlevels = len(self.tarch)
        level0 = self.levels[0](code).reshape(-1, self.nfeat, int(self.tarch[0]))
        outs = [level0, ]
        for i in range(1, nlevels):
            if i == len(self.tarch) - 1:
                nout = 3
            else:
                nout = self.nout
            inp = outs[-1]
            y = torch.cat([inp, code[:, :, None].expand(-1, -1, inp.shape[2])], 1)
            outs.append(self.levels[i](y).reshape(y.shape[0], nout, -1))
            
        reconstruction = outs[-1].transpose(-1, -2)
        return reconstruction
    
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.encoder = Encoder(cfg.encoder.radius, cfg.encoder.num_layers, cfg.encoder.ch_mid, cfg.encoder.emb_dim + 1)
        
        self.beta_dist = torch.distributions.Beta(concentration1=cfg.beta.concentration1, concentration0=cfg.beta.concentration0)
        self.discriminator = Discriminator(cfg.discriminator.chs)
        self.fc_topnet = nn.Linear(cfg.encoder.emb_dim * 2, cfg.topnet.code_nfts)
        self.topnet = TopNet(cfg)
        self.cfg = cfg
    
    def forward(self, pc, pc_feature):
        pc_sym = torch.stack([-pc[..., 0], pc[..., 1], pc[..., 2]], -1)
        dist2sym = torch.cdist(pc, pc_sym)
        nearest_idx = torch.argmin(dist2sym, -1)  # B x N
        
        # import pdb; pdb.set_trace()
        embedding = self.encoder(pc)
        z = torch.sigmoid(embedding[..., -1])
        emb = F.normalize(embedding[..., :-1], dim=-1)
        # import pdb; pdb.set_trace()
        if self.training:
            real_z = self.beta_dist.sample(z.shape).to(z)
            fake_val = self.discriminator(z)
            real_val = self.discriminator(real_z)
            loss_gan = torch.mean(fake_val - real_val)
            
            z_interp = z + torch.rand_like(z) * (real_z - z)
            gradient_f = torch.autograd.grad(torch.sum(self.discriminator(z_interp)), z_interp)[0]
            loss_gp = torch.mean(torch.maximum(torch.norm(gradient_f, dim=-1) - 1, torch.zeros((*gradient_f.shape[:-1],)).to(gradient_f)) ** 2)
            
            code = torch.cat([torch.max(torch.relu(emb) * z[..., None], 1)[0], torch.max(torch.relu(-emb) * z[..., None], 1)[0]], -1)
            code = self.fc_topnet(code)
            
            # topnet
            rec = self.topnet(code)
            dist = torch.cdist(pc, rec)
            loss_recon = self.cfg.recon_factor * torch.mean(torch.min(dist, -1)[0] + torch.min(dist, -2)[0])
            
            # symmetric loss
            embbeding_sym = torch.gather(embedding, 1, nearest_idx[..., None].expand_as(embedding))
            loss_sym = self.cfg.symmetry_factor * torch.mean(torch.sum(torch.abs(embbeding_sym - embedding), -1))
            
            return z, emb, loss_gan, loss_gp, loss_recon, loss_sym
        else:
            return z, emb