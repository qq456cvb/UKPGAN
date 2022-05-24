import pytorch_lightning as pl
import hydra
import torch
from dataset import ShapeNetDataset
from torch.utils.data import DataLoader

from model import Model


class UKPGAN(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Model(cfg)
        self.automatic_optimization = False

    def train_dataloader(self):
        return DataLoader(
            ShapeNetDataset(self.cfg, self.cfg.data.train_txt),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            ShapeNetDataset(self.cfg, self.cfg.data.val_txt),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.num_workers
        )

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        z, emb, loss_gan, loss_gp, loss_recon, loss_sym = self.model(batch)
        
        opt_d.zero_grad()
        self.manual_backward(loss_gan + loss_gp, retain_graph=True)
        opt_d.step()
        
        opt_g.zero_grad()
        self.manual_backward(-torch.mean(self.model.discriminator(z)) + loss_recon + loss_sym, retain_graph=False)
        opt_g.step()
        
        self.log('train/loss_gan', loss_gan.item(), prog_bar=True)
        self.log('train/loss_gp', loss_gp.item(), prog_bar=True)
        self.log('train/loss_recon', loss_recon.item(), prog_bar=True)
        self.log('train/loss_sym', loss_sym.item(), prog_bar=True)
        
    def validation_epoch_end(self, validation_step_outputs):
        pass
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam([*self.model.encoder.parameters(), *self.model.topnet.parameters(), *self.model.fc_topnet.parameters()], lr=self.cfg.learning_rate, weight_decay=1e-4)
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-4)
        return [opt_g, opt_d]
    
    
@hydra.main(config_path='config', config_name='config')
def main(cfg):
    accel = 'ddp'
    pl_module = UKPGAN(cfg)
    trainer = pl.Trainer(max_epochs=cfg.max_epoch, gpus=torch.cuda.device_count(), accelerator=accel)
    trainer.fit(pl_module)

if __name__ == '__main__':
    main()