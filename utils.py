import tensorflow as tf
import math
import numpy as np
import ops
from tensorpack.models.regularize import Dropout


import imp
import numpy as np
import pickle
import open3d as o3d
import os

BASE_DIR = os.path.dirname(__file__)

class SMPLModel():
  def __init__(self, model_path):
    """
    SMPL model.

    Parameter:
    ---------
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
    with open(model_path, 'rb') as f:
      params = pickle.load(f)

      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      self.v_template = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']

    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    }
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None

    self.update()

  def set_params(self, pose=None, beta=None, trans=None):
    """
    Set pose, shape, and/or translation parameters of SMPL model. Verices of the
    model will be updated and returned.

    Parameters:
    ---------
    pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
    relative to parent joint. For root joint it's global orientation.
    Represented in a axis-angle format.

    beta: Parameter for model shape. A vector of shape [10]. Coefficients for
    PCA component. Only 10 components were released by MPI.

    trans: Global translation of shape [3].

    Return:
    ------
    Updated vertices.

    """
    if pose is not None:
      self.pose = pose
    if beta is not None:
      self.beta = beta
    if trans is not None:
      self.trans = trans
    self.update()
    return self.verts

  def update(self):
    """
    Called automatically when parameters are updated.

    """
    # how beta affect body shape
    v_shaped = self.shapedirs.dot(self.beta) + self.v_template
    # joints location
    self.J = self.J_regressor.dot(v_shaped)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3)
    )
    lrotmin = (self.R[1:] - I_cube).ravel()
    # how pose affect body shape in zero pose
    v_posed = v_shaped + self.posedirs.dot(lrotmin)
    # world transformation of each joint
    G = np.empty((self.kintree_table.shape[1], 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.kintree_table.shape[1]):
      G[i] = G[self.parent[i]].dot(
        self.with_zeros(
          np.hstack(
            [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
          )
        )
      )
    G = G - self.pack(
      np.matmul(
        G,
        np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
        )
      )
    # transformation of each vertex
    T = np.tensordot(self.weights, G, axes=[[1], [0]])
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
    v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    self.verts = v + self.trans.reshape([1, 3])

  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(np.float64).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

  def save_to_obj(self, path):
    """
    Save the SMPL model into .obj file.

    Parameter:
    ---------
    path: Path to save.

    """
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    
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


def sample_vertex_from_mesh(vertex, facet, colors=None, rnd_idxs=None, u=None, v=None, num_samples=2048):
    # mean = np.mean(vertex, axis=0, keepdims=True)
    # norm = np.max(np.linalg.norm(vertex - mean, axis=1))

    vertex = vertex * 1e2
    triangles = np.take(vertex, facet, axis=0)
    vx, vy, vz = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    if colors is not None:
        trianlges_color = np.take(colors, facet, axis=0)
        cx, cy, cz = trianlges_color[:, 0, :], trianlges_color[:, 1, :], trianlges_color[:, 2, :]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(vy - vx, vz - vx), axis=1)
    probs = triangle_areas / np.sum(triangle_areas)

    if rnd_idxs is None:
        rnd_idxs = np.random.choice(np.arange(probs.shape[0]), size=num_samples, p=probs)
    vx, vy, vz = vx[rnd_idxs], vy[rnd_idxs], vz[rnd_idxs]
    
    if colors is not None:
        cx, cy, cz = cx[rnd_idxs], cy[rnd_idxs], cz[rnd_idxs]
    if u is None:
        u = np.random.rand(vx.shape[0], 1)
    if v is None:
        v = np.random.rand(vx.shape[0], 1)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - (u + v)
    pts = (vx * u + vy * v + vz * w) / 1e2
    if colors is not None:
        c = (cx * u + cy * v + cz * w)
    else:
        c = None
    # pts = pts - mean
    # pts = pts / norm
    return pts, c, rnd_idxs, u, v