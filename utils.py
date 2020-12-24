import tensorflow as tf
import math
import numpy as np
import ops
from tensorpack.models.regularize import Dropout


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


def get_arch(nlevels, npts):
    tree_arch = {}
    tree_arch[2] = [32, 64]
    tree_arch[4] = [4, 8, 8, 8]
    tree_arch[6] = [2, 4, 4, 4, 4, 4]
    tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]

    logmult = int(math.log2(npts / 2048))
    assert 2048 * (2 ** logmult) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch == np.min(arch))[0][-1]
        arch[last_min_pos] *= 2
        logmult -= 1
    return arch


def conv_block(input, channels, dropout_flag, dropout_rate, laxer_idx, stride_input=1, k_size=3,
               padding_type='SAME'):
    # Traditional 3D conv layer followed by batch norm and relu activation

    i_size = input.get_shape().as_list()[-2] / stride_input

    weights = ops.weight([k_size, k_size, k_size, channels[0], channels[1]],
                         layer_name='wcnn' + str(laxer_idx + 1), reuse=tf.get_variable_scope().reuse)

    bias = ops.bias([i_size, i_size, i_size, channels[1]], layer_name='bcnn' + str(laxer_idx + 1),
                    reuse=tf.get_variable_scope().reuse)

    conv_output = tf.add(
        ops.conv3d(input, weights, stride=[stride_input, stride_input, stride_input], padding=padding_type), bias)
    conv_output = ops.batch_norm(conv_output)
    conv_output = ops.relu(conv_output)

    if dropout_flag:
        conv_output = Dropout(conv_output, keep_prob=dropout_rate)

    return conv_output


def out_block(input, channels, laxer_idx, stride_input=1, k_size=8, padding_type='VALID'):
    # Last conv layer, flatten the output
    weights = ops.weight([k_size, k_size, k_size, channels[0], channels[1]],
                         layer_name='wcnn' + str(laxer_idx + 1))

    bias = ops.bias([1, 1, 1, channels[1]], layer_name='bcnn' + str(laxer_idx + 1))

    conv_output = tf.add(
        ops.conv3d(input, weights, stride=[stride_input, stride_input, stride_input], padding=padding_type), bias)
    conv_output = ops.batch_norm(conv_output)
    conv_output = tf.contrib.layers.flatten(conv_output)

    return conv_output


def mlp(features, layer_dims, phase, bn=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            activation_fn=None,
            normalizer_fn=None,
            scope='fc_%d' % i)
        if bn:
            with tf.variable_scope('fc_bn_%d' % (i), reuse=tf.AUTO_REUSE):
                features = tf.layers.batch_normalization(features, training=phase)
        features = tf.nn.relu(features, 'fc_relu_%d' % i)

    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, phase, bn=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            activation=None,
            name='conv_%d' % i)
        if bn:
            with tf.variable_scope('conv_bn_%d' % (i), reuse=tf.AUTO_REUSE):
                inputs = tf.layers.batch_normalization(inputs, training=phase)
        inputs = tf.nn.relu(inputs, 'conv_relu_%d' % i)
    outputs = tf.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation=None,
        name='conv_%d' % (len(layer_dims) - 1))
    return outputs


# pcd1,2 : B x N x 3
def chamfer(pcd1, pcd2):
    dist = tf.norm(pcd1[:, :, None] - pcd2[:, None], axis=-1)  # B x M x N
    return tf.reduce_mean(tf.reduce_min(dist, -1)) + tf.reduce_mean(tf.reduce_min(dist, -2))
