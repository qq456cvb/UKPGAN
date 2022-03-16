import tensorpack
from tensorpack.callbacks.monitor import ScalarPrinter
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.callbacks.summary import SimpleMovingAverage
from tensorpack.callbacks.trigger import PeriodicTrigger
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.parallel import PrefetchData
from tensorpack.train.config import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SimpleTrainer
from tensorpack.utils import logger
from model import Model
from dataflow import SMPLDataFlow, VisDataFlow, ShapeNetDataFlow, VisSMPLDataFlow
from multiprocessing import cpu_count
import hydra
from shutil import copyfile
import tensorflow as tf
import subprocess


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    print(cfg)
    
    tf.reset_default_graph()
    
    logger.set_logger_dir('tflogs', action='d')

    copyfile(hydra.utils.to_absolute_path('model.py'), 'model.py')
    copyfile(hydra.utils.to_absolute_path('dataflow.py'), 'dataflow.py')
    
    if cfg.cat_name == 'smpl':
        train_df = SMPLDataFlow(cfg, True, 1000)
        val_df = VisSMPLDataFlow(cfg, True, 1000, port=1080)
    else:
        train_df = ShapeNetDataFlow(cfg, cfg.data.train_txt, True)
        val_df = VisDataFlow(cfg, cfg.data.val_txt, False, port=1080)
    
    config = TrainConfig(
        model=Model(cfg),
        dataflow=BatchData(PrefetchData(train_df, cpu_count() // 2, cpu_count() // 2), cfg.batch_size),
        callbacks=[
            ModelSaver(),
            SimpleMovingAverage(['recon_loss', 'GAN/loss_d', 'GAN/loss_g', 'GAN/gp_loss', 'symmetry_loss'], 100),
            PeriodicTrigger(val_df, every_k_steps=30)
        ],
        monitors=tensorpack.train.DEFAULT_MONITORS() + [ScalarPrinter(enable_step=True, enable_epoch=False)],
        max_epoch=10
    )
    launch_train_with_config(config, SimpleTrainer())


if __name__ == '__main__':
    main()
    
