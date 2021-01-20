from tensorpack import *
import tensorflow as tf
import numpy as np
from tensorpack.models.fc import FullyConnected
from tensorpack.models.linearwrap import LinearWrap
from tensorpack.models.regularize import regularize_cost
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils import varreplace
from tensorpack.tfutils import varreplace, summary, get_current_tower_context, optimizer, gradproc
from utils import *
from matplotlib import cm
cmap = cm.get_cmap('jet')


def SmoothNet(x, cfg):
    batch_size = tf.shape(x)[0]
    edge = int(np.cbrt(cfg.input_dim))

    x = tf.reshape(x, [-1, edge, edge, edge, 1])
    # NETWORK 2: logits local, features local
    # 3D smooth net
    # Join the 3DSmoothNet structure with the desired output dimension
    net_structure = [1, 32, 32, 64, 64, 128, 128]
    outputDim = 128
    channels = [*net_structure, outputDim]

    # In the third layer stride is 2
    stride = np.ones(len(channels))
    stride[2] = 2

    # Apply dropout in the 6th layer
    dropout_flag = np.zeros(len(channels))
    dropout_flag[5] = 1

    layer_index = 0

    # Loop over the desired layers
    for layer in np.arange(0, len(channels) - 2):
        scope_name = "3DIM_cnn" + str(layer_index + 1)
        with tf.variable_scope(scope_name):
            x = conv_block(x, [channels[layer], channels[layer + 1]],
                           dropout_flag[layer], 0.7, layer_index,
                           stride_input=stride[layer])

        layer_index += 1

    with tf.variable_scope('3DIM_cnn7'):
        x = out_block(x, [channels[-2], channels[-1]],
                      layer_index)

    x = tf.reshape(x, [-1, x.get_shape()[-1]])

    x = (LinearWrap(x)
          .FullyConnected('fc1', 512, activation=tf.nn.relu)
          .FullyConnected('fc2', 256, activation=tf.nn.relu)
          .FullyConnected('fc3', 129, activation=None))()

    x = tf.reshape(x, tf.stack([batch_size, cfg.num_points, 129]))
    return x
    
    
class Model(ModelDesc):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def inputs(self):
        """
        Define all the inputs (with type, shape, name) that the graph will need.
        """
        return [tf.placeholder(tf.float32, [None, self.cfg.num_points, 3], name='pc'),
                tf.placeholder(tf.float32, [None, self.cfg.num_points, self.cfg.input_dim], name='pc_feature')]

    @auto_reuse_variable_scope
    def discriminator(self, logits):
        return (LinearWrap(logits).FullyConnected('fc1', 512, activation=tf.nn.relu)
                .FullyConnected('fc2', 256, activation=tf.nn.relu)
                .FullyConnected('fc3', 128, activation=tf.nn.relu)
                .FullyConnected('fc4', 64, activation=tf.nn.relu)
                .FullyConnected('fc5', 1, activation=None))()

    def build_graph(self, pc, pc_feature):
        pc_symmetry = tf.stack([-pc[..., 0], pc[..., 1], pc[..., 2]], -1)  # -x
        dist2sym = tf.reduce_sum((pc[:, :, None] - pc_symmetry[:, None]) ** 2, -1)
        nearest_idx = tf.argmin(dist2sym, -1, output_type=tf.int32)
        
        # smoothnet encoder, only local features are used
        embedding = SmoothNet(pc_feature, self.cfg)
        with tf.variable_scope('encoder'):
            z = tf.sigmoid(embedding[:, :, -1], name='z')
            output_x = tf.nn.l2_normalize(embedding[:, :, :-1], axis=-1, name='feature')

        gp_loss = 0.
        loss_d = 0.
        loss_g = 0.
        if get_current_tower_context().is_training:
            beta_dist = tf.distributions.Beta(concentration1=self.cfg.beta.concentration1, concentration0=self.cfg.beta.concentration0)
                
            with tf.variable_scope('GAN'):
                real_z = beta_dist.sample(tf.shape(z))
                fake_val = self.discriminator(tf.stop_gradient(z))
                real_val = self.discriminator(real_z)
                loss_d = tf.reduce_mean(fake_val - real_val, name='loss_d')
                with varreplace.freeze_variables(stop_gradient=True):
                    loss_g = tf.reduce_mean(-self.discriminator(z), name='loss_g')

                z_interp = z + tf.random_uniform((tf.shape(fake_val)[0], 1)) * (real_z - z)
                gradient_f = tf.gradients(self.discriminator(z_interp), [z_interp])[0]
                gp_loss = tf.reduce_mean(tf.maximum(tf.norm(gradient_f, axis=-1) - 1, 0) ** 2, name='gp_loss')
        code = tf.concat([tf.reduce_max(tf.nn.relu(output_x) * z[..., None], 1), tf.reduce_max(tf.nn.relu(-output_x) * z[..., None], 1)], axis=-1, name='code')
        code = FullyConnected('fc_global', code, self.cfg.topnet.code_nfts, activation=None)
        
        # topnet decoder
        tarch = get_arch(self.cfg.topnet.nlevels, self.cfg.num_points)

        def create_level(level, input_channels, output_channels, inputs, bn):
            with tf.variable_scope('level_%d' % level, reuse=tf.AUTO_REUSE):
                features = mlp_conv(inputs, [input_channels, int(input_channels / 2),
                                             int(input_channels / 4), int(input_channels / 8),
                                             output_channels * int(tarch[level])],
                                    get_current_tower_context().is_training, bn)
                features = tf.reshape(features, [tf.shape(features)[0], -1, output_channels])
            return features

        Nin = self.cfg.topnet.nfeat + self.cfg.topnet.code_nfts
        Nout = self.cfg.topnet.nfeat
        bn = True
        N0 = int(tarch[0])
        nlevels = len(tarch)
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            level0 = mlp(code, [256, 64, self.cfg.topnet.nfeat * N0], get_current_tower_context().is_training, bn=True)
            level0 = tf.tanh(level0, name='tanh_0')
            level0 = tf.reshape(level0, [-1, N0, self.cfg.topnet.nfeat])
            outs = [level0, ]
            for i in range(1, nlevels):
                if i == nlevels - 1:
                    Nout = 3
                    bn = False
                inp = outs[-1]
                y = tf.expand_dims(code, 1)
                y = tf.tile(y, [1, tf.shape(inp)[1], 1])
                y = tf.concat([inp, y], 2)
                outs.append(tf.tanh(create_level(i, Nin, Nout, y, bn), name='tanh_%d' % (i)))

        reconstruction = tf.reshape(outs[-1], [-1, self.cfg.num_points, 3], name='recon_pc')
        loss_recon = chamfer(reconstruction, pc)

        loss_recon = tf.identity(self.cfg.recon_factor * tf.reduce_mean(loss_recon), name='recon_loss')

        batch_size = tf.shape(output_x)[0]
        batch_idx = tf.tile(tf.range(batch_size)[:, None], [1, tf.shape(nearest_idx)[1]])
        feature_sym = tf.gather_nd(embedding, tf.stack([batch_idx, nearest_idx], -1))
        
        loss_sym = tf.identity(self.cfg.symmetry_factor * tf.reduce_mean(tf.reduce_sum(tf.abs(feature_sym - embedding), -1)), 'symmetry_loss')
        
        wd_cost = tf.multiply(1e-4, regularize_cost('.*(_W|kernel)', tf.nn.l2_loss), name='regularize_loss')
        loss_gan = loss_d + loss_g + gp_loss
        total_cost = tf.add_n([loss_recon, wd_cost, loss_gan, loss_sym], name='total_cost')
        summary.add_moving_summary(loss_recon, loss_sym)
        summary.add_param_summary(['.*(_W|kernel)', ['histogram', 'rms']])
        return total_cost

    def optimizer(self):
        opt = tf.train.AdamOptimizer(self.cfg.learning_rate)
        return optimizer.apply_grad_processors(
            opt, [
            gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3)),
            gradproc.SummaryGradient()])
