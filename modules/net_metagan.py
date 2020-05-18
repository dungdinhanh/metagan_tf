from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from functools import partial
from modules.ops import *
import numpy as np
from tensorflow.contrib.layers.python import layers as tf_layers

'''
************************************************************************
* The small Meta-GAN architecture for MNIST (28 x 28 x 1)
************************************************************************
'''

def encoder_dcgan_mnist(img, x_shape, z_dim=100, dim=64, \
                             kernel_size=5, stride=2, \
                             name = 'encoder', \
                             reuse=True, training=True):
                                 
    bn = partial(batch_norm, is_training=training)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, \
                                                biases_initializer=None)
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = relu(conv(y, dim, kernel_size, stride))        #14 x 14 x dim
        y = conv_bn_relu(y, dim * 2, kernel_size, stride)  #7  x 7  x dim x 2
        y = conv_bn_relu(y, dim * 4, kernel_size, stride)  #4  x 4  x dim x 4
        logit = fc(y, z_dim)
        return logit
        
def generator_dcgan_mnist(z, x_shape, dim=64, kernel_size=5, stride=2, \
                          name = 'generator', \
                          reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, \
                            activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, \
                                                biases_initializer=None)
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]  
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 4)                 
        y = tf.reshape(y, [-1, 4, 4, dim * 4])             #4 x 4 x dim x 4
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride) #8 x 8 x dim x 2           
        y = y[:,:7,:7,:]                                   #7 x 7 x dim x 2
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride) #14 x 14 x dim
        y = dconv(y, x_shape[2], kernel_size, stride)      #28 x 28 x 3
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)

def discriminator_dcgan_mnist(img, x_shape, dim=64, \
                             kernel_size=5, stride=2, \
                             name='discriminator', \
                             reuse=True, training=True):
                                 
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
                           activation_fn=lrelu, biases_initializer=None)
    
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))       #14 x 14 x dim
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride) #7  x 7  x dim x 2
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride) #4  x 4  x dim x 4
        feature = y
        logit = fc(y, 1)
        return tf.nn.sigmoid(logit), logit


def normalize(inp, activation, reuse, scope, norm='batch_norm', is_training=True):
    if norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope, is_training=is_training)
    elif norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

def construct_weights(dim=64, kernel_size=5, channel_size=1, psi=[10, 10], aux=True):
    weights={}
    dtype = tf.float32
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
    fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
    k=kernel_size
    channels=channel_size
    aux_num = 0
    if aux:
        aux_num = np.sum(psi)

    weights['conv1'] = tf.get_variable('conv1', [k, k, channels, dim], initializer=conv_initializer, dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([dim]))

    weights['conv2'] = tf.get_variable('conv2', [k, k, dim, dim * 2], initializer=conv_initializer, dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([dim*2]))

    weights['conv3'] = tf.get_variable('conv3', [k, k, dim*2, dim * 4], initializer=conv_initializer, dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([dim*4]))

    weights['w4'] = tf.get_variable('w4', [4 * 4 * dim * 4, 1], initializer=fc_initializer)
    weights['b4'] = tf.Variable(tf.zeros(1))

    if aux:
        weights['conv5'] = tf.get_variable('conv5', [4, 4, dim * 4, dim * 4], initializer=conv_initializer, dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros(dim * 4))

        weights['w6'] = tf.get_variable('w6', [dim * 4, dim*2], initializer=fc_initializer)
        weights['b6'] = tf.Variable(tf.zeros(dim*2))

        weights['w7'] = tf.get_variable('w7', [dim*2, aux_num], initializer=fc_initializer)
        weights['b7'] = tf.Variable(tf.zeros(aux_num))

    return weights


def meta_discriminator_dcgan_mnist(img, x_shape, weights,
                                   stride=2,
                                   name='discriminator',
                                   reuse=True, training=True, aux=True):
    y = tf.reshape(img, [-1, x_shape[0], x_shape[1], x_shape[2]])
    n_stride = stride
    stride, no_stride = [1, n_stride, n_stride, 1], [1, 1, 1, 1]
    with tf.variable_scope(name, reuse=reuse):
        y = tf.nn.conv2d(y, weights['conv1'], stride, 'SAME') + weights['b1'] # 14 x 14 x dim
        y = tf.nn.leaky_relu(y)

        y = tf.nn.conv2d(y, weights['conv2'], stride, 'SAME') + weights['b2'] # 7 x 7 x dim*2
        y = normalize(y, tf.nn.leaky_relu, reuse, 'bn1', is_training=training) # in case error look at this

        y = tf.nn.conv2d(y, weights['conv3'], stride, 'SAME') + weights['b3'] # 4 x 4 x dim *4
        y = normalize(y, tf.nn.leaky_relu, reuse, 'bn2', is_training=training)

        logit = tf_layers.flatten(y)
        logit1 = tf.matmul(logit, weights['w4']) + weights['b4']

        if aux:
            y1 = tf.nn.conv2d(y, weights['conv5'], no_stride, 'VALID') + weights['b5']
            y1 = normalize(y1, tf.nn.leaky_relu, reuse, 'bn3', is_training=training)
            y1 = tf.nn.leaky_relu(y1)


            y1 = tf_layers.flatten(y1)
            y1 = tf.matmul(y1, weights['w6']) + weights['b6']
            y1 = tf.nn.relu(y1)

            y1 = tf.matmul(y1, weights['w7']) + weights['b7']
            logit2 = tf.nn.softmax(y1, dim=1)
            return logit1, logit2
        return logit1, None

# def meta_discriminator_dcgan_mnist(img, x_shape, dim=64, \
#                               kernel_size=5, stride=2, \
#                               name='discriminator', psi=[10, 10],\
#                               reuse=True, training=True, aux=True):
#     bn = partial(batch_norm, is_training=training)
#     conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
#                             activation_fn=lrelu, biases_initializer=None)
#     aux_num = 0
#     if aux:
#         aux_num = np.sum(psi)
#     y = tf.reshape(img, [-1, x_shape[0], x_shape[1], x_shape[2]])
#
#     with tf.variable_scope(name, reuse=reuse):
#         y = lrelu(conv(y, dim, kernel_size, stride))  # 14 x 14 x dim
#         y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  # 7  x 7  x dim x 2
#         y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  # 4  x 4  x dim x 4
#         feature = y
#         logit = fc(y, 1)
#         if aux:
#             # y1 = tf.nn.conv2d(y, dim*4, 4, 1, 0)
#             y1 = conv_bn_lrelu(y, dim*4, 4, 1, padding='VALID')
#             y1 = tf.nn.leaky_relu(y1)
#             logit2 = fc(y1, dim * 2)
#             logit2 = tf.nn.relu(logit2)
#             logit2 = fc(logit2, int(aux_num))
#             logit2 = tf.nn.softmax(logit2, dim=1) #dim or axis ?
#             return logit, logit2
#             pass
#         return logit, None


def mask_softmask(x, mask, dim=1):
    logits = tf.math.exp(x - tf.reduce_max(x, 0)) * mask / tf.reduce_sum(tf.math.exp(x - tf.reduce_max(x, 0)) * mask, axis=dim, keepdims=True)
    return logits

def label_gen_dcgan_mnist(img, x_shape, dim=64, \
                          kernel_size=5, stride=2, \
                          name='labelgen', psi=[10], \
                          reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
                            activation_fn=lrelu, biases_initializer=None)
    aux_num = np.sum(psi)
    y = tf.reshape(img, [-1, x_shape[0], x_shape[1], x_shape[2]])

    with tf.variable_scope(name, reuse=reuse):  # not check yet
        y = lrelu(conv(y, dim, kernel_size, stride))  # 14 x 14 x dim
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  # 7  x 7  x dim x 2
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  # 4  x 4  x dim x 4
        y1 = conv_bn_lrelu(y, dim * 4, 4, 1, padding='VALID')
        y1 = tf.nn.leaky_relu(y1)
        logit2 = fc(y1, dim * 2)
        logit2 = tf.nn.relu(logit2)
        logit2 = fc(logit2, int(aux_num))
        logit2 = tf.nn.softmax(logit2, dim=1)  # dim or axis ?
        return logit2

# def label_gen_dcgan_mnist(img, x_shape, prim_y, dim=64, \
#                           kernel_size=5, stride=2, \
#                           name='labelgen', psi=[10, 10],\
#                           reuse=True, training=True):
#     bn = partial(batch_norm, is_training=training)
#     conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
#                             activation_fn=lrelu, biases_initializer=None)
#     aux_num = np.sum(psi)
#     nb = prim_y.shape[0]
#     aux_indices = np.arange(nb, dtype=np.int64)
#     index = np.zeros((nb, len(psi), aux_num)) + 1e-8
#     for i in range(len(psi)):
#         index[:, i, int(np.sum(psi[:i])):np.sum(psi[:i+1])] = 1
#     # index = tf.convert_to_tensor(index)
#     # list_y = tf.convert_to_tensor(prim_y)
#     list_y = prim_y
#     # list_y = tf.dtypes.cast(prim_y, tf.int64)
#     # list_y = tf.reshape(list_y, [list_y.get_shape()[0],])
#     mask = index[aux_indices, list_y]
#     mask = tf.constant(mask, dtype=tf.float32)
#
#     y = tf.reshape(img, [-1, x_shape[0], x_shape[1], x_shape[2]])
#
#     with tf.variable_scope(name, reuse=reuse): #not check yet
#         y = lrelu(conv(y, dim, kernel_size, stride))  # 14 x 14 x dim
#         y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  # 7  x 7  x dim x 2
#         y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  # 4  x 4  x dim x 4
#         y1 = conv_bn_lrelu(y, dim * 4, 4, 1, padding='VALID')
#         y1 = tf.nn.leaky_relu(y1)
#         logit2 = fc(y1, dim * 2)
#         logit2 = tf.nn.relu(logit2)
#         logit2 = fc(logit2, int(aux_num))
#         logit2 = tf.nn.softmax(logit2, dim=1) #dim or axis ?
#         return mask_softmask(logit2, mask, dim=1)


'''
************************************************************************
* The small Meta-GAN architecture for CIFAR-10 and CIFAR-100 (32 x 32 x 3)
************************************************************************
'''


def encoder_dcgan_cifar(img, x_shape, z_dim=128, dim=64, kernel_size=5, \
                        stride=2, name='encoder', \
                        reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
                            activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img, [-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))  # 16 x 16 x dim
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  # 8 x 8 x dim x 2
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  # 4 x 4 x dim x 4
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)  # 2 x 2 x dim x 8
        logit = fc(y, z_dim)
        return logit


def generator_dcgan_cifar(z, x_shape, dim=64, kernel_size=5, stride=2, \
                          name='generator', \
                          reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, \
                            activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, \
                         activation_fn=relu, biases_initializer=None)

    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 2 * 2 * dim * 8)
        y = tf.reshape(y, [-1, 2, 2, dim * 8])  # 2 x 2 x dim x 8
        y = dconv_bn_relu(y, dim * 4, kernel_size, stride)  # 4 x 4 x dim x 4
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride)  # 8 x 8 x dim x 2
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride)  # 16 x 16 x dim
        y = dconv(y, x_shape[2], kernel_size, stride)  # 32 x 32 x 3
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)


def construct_weights_discriminator_cifar(dim=64, kernel_size=5, channel_size=3, psi=[10]):
    weights={}
    dtype = tf.float32
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
    fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
    k=kernel_size
    channels=channel_size
    aux_num = int(np.sum(psi))

    weights['conv1'] = tf.get_variable('conv1', [k, k, channels, dim], initializer=conv_initializer, dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([dim]))

    weights['conv2'] = tf.get_variable('conv2', [k, k, dim, dim * 2], initializer=conv_initializer, dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([dim*2]))

    weights['conv3'] = tf.get_variable('conv3', [k, k, dim*2, dim * 4], initializer=conv_initializer, dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([dim*4]))

    weights['conv4'] = tf.get_variable('conv4', [k, k, dim*4, dim * 8], initializer=conv_initializer, dtype=dtype)
    weights['b4'] = tf.Variable(tf.zeros([dim*8]))

    weights['w5'] = tf.get_variable('w5', [2 * 2 * dim * 8, 1], initializer=fc_initializer)
    weights['b5'] = tf.Variable(tf.zeros(1))

    # Weights for auxiliary task
    weights['conv6'] = tf.get_variable('conv6', [2, 2, dim * 8, dim * 8], initializer=conv_initializer, dtype=dtype)
    weights['b6'] = tf.Variable(tf.zeros(dim * 8))

    weights['w7'] = tf.get_variable('w7', [dim * 8, dim*4], initializer=fc_initializer)
    weights['b7'] = tf.Variable(tf.zeros(dim*4))

    weights['w8'] = tf.get_variable('w8', [dim*4, aux_num], initializer=fc_initializer)
    weights['b8'] = tf.Variable(tf.zeros(aux_num))

    return weights


def meta_discriminator_dcgan_cifar(img, x_shape, weights, \
                              stride=2, \
                              name='discriminator', \
                              reuse=True, training=True):
    # Remember to set ss_task = 0
    # bn = partial(batch_norm, is_training=training)
    # conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
    #                         activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img, [-1, x_shape[0], x_shape[1], x_shape[2]])
    n_stride = stride
    stride, no_stride = [1, n_stride, n_stride, 1], [1, 1, 1, 1]
    with tf.variable_scope(name, reuse=reuse):
        y = tf.nn.conv2d(y, weights['conv1'], stride, 'SAME') + weights['b1']  # 16 x 16 x dim
        y = tf.nn.leaky_relu(y)

        y = tf.nn.conv2d(y, weights['conv2'], stride, 'SAME') + weights['b2']  # 8 x 8 x dim x 2
        y = normalize(y, tf.nn.leaky_relu, reuse, 'bn1', is_training=training)

        y = tf.nn.conv2d(y, weights['conv3'], stride, 'SAME') + weights['b3']  # 4 x 4 x dim x 4
        y = normalize(y, tf.nn.leaky_relu, reuse, 'bn2', is_training=training)

        y = tf.nn.conv2d(y, weights['conv4'], stride, 'SAME') + weights['b4']  # 2 x 2 x dim x 8
        y = normalize(y, tf.nn.leaky_relu, reuse, 'bn3', is_training=training)

        logit = tf_layers.flatten(y)
        logit1 = tf.matmul(logit, weights['w5']) + weights['b5']

        # For recog task
        y1 = tf.nn.conv2d(y, weights['conv6'], no_stride, 'VALID') + weights['b6']
        y1 = normalize(y1, tf.nn.leaky_relu, reuse, 'bn4', is_training=training)

        y1 = tf_layers.flatten(y1)
        y1 = tf.matmul(y1, weights['w7']) + weights['b7']
        y1 = tf.nn.relu(y1)

        y1 = tf.matmul(y1, weights['w8']) + weights['b8']
        logit2 = tf.nn.softmax(y1, dim=1)

        return tf.nn.sigmoid(logit1), logit1, logit2, logit

        # if ss_task == 1:
        #     k = 4
        # elif ss_task == 2:
        #     k = 5
        # else:
        #     k = -1
        # if ss_task > 0:
        #     print('[net_metagan.py -- meta_discriminator_dcgan_cifar] SS task = %d with k = %d classes' % (ss_task, k))
        #     cls = fc(y, k)
        #     return tf.nn.sigmoid(logit1), logit1, \
        #             logit, cls
        # y = lrelu(conv(y, dim, kernel_size, 2))  # 16 x 16 x dim
        # y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  # 8 x 8 x dim x 2
        # y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  # 4 x 4 x dim x 4
        # y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)  # 2 x 2 x dim x 8
        # feature = y

# Please test label generator
def label_gen_cifar(img, x_shape, dim=64, \
                          kernel_size=5, stride=2, \
                          name='labelgen', psi=[10], \
                          reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
                            activation_fn=lrelu, biases_initializer=None)
    aux_num = int(np.sum(psi))
    y = tf.reshape(img, [-1, x_shape[0], x_shape[1], x_shape[2]])

    with tf.variable_scope(name, reuse=reuse):  # not check yet
        y = lrelu(conv(y, dim, kernel_size, stride))  # 16 x 16 x dim
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  # 8  x 8  x dim x 2
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  # 4  x 4  x dim x 4
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)  # 2  x 2  x dim x 8
        y1 = conv_bn_lrelu(y, dim * 8, 2, 1, padding='VALID')
        # y1 = tf.nn.leaky_relu(y1)
        logit2 = fc(y1, dim * 4)
        logit2 = tf.nn.relu(logit2)
        logit2 = fc(logit2, dim * 2)
        logit2 = tf.nn.relu(logit2)
        logit2 = fc(logit2, aux_num)
        logit2 = tf.nn.softmax(logit2, dim=1)  # dim or axis ?
        return logit2

# def label_gen_cifar(img, x_shape, prim_y, dim=64, \
#                           kernel_size=5, stride=2, \
#                           name='labelgen', psi=[10, 10],\
#                           reuse=True, training=True):
#     bn = partial(batch_norm, is_training=training)
#     conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
#                             activation_fn=lrelu, biases_initializer=None)
#     aux_num = np.sum(psi)
#     nb = prim_y.shape[0]
#     aux_indices = np.arange(nb, dtype=np.int64)
#     index = np.zeros((nb, len(psi), aux_num)) + 1e-8
#     for i in range(len(psi)):
#         index[:, i, int(np.sum(psi[:i])):np.sum(psi[:i+1])] = 1
#     # index = tf.convert_to_tensor(index)
#     # list_y = tf.convert_to_tensor(prim_y)
#     list_y = prim_y
#     # list_y = tf.dtypes.cast(prim_y, tf.int64)
#     # list_y = tf.reshape(list_y, [list_y.get_shape()[0],])
#     mask = index[aux_indices, list_y]
#     mask = tf.constant(mask, dtype=tf.float32)
#
#     y = tf.reshape(img, [-1, x_shape[0], x_shape[1], x_shape[2]])
#
#     with tf.variable_scope(name, reuse=reuse): #not check yet
#         y = lrelu(conv(y, dim, kernel_size, stride))  # 14 x 14 x dim
#         y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  # 7  x 7  x dim x 2
#         y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  # 4  x 4  x dim x 4
#         y1 = conv_bn_lrelu(y, dim * 4, 4, 1, padding='VALID')
#         y1 = tf.nn.leaky_relu(y1)
#         logit2 = fc(y1, dim * 2)
#         logit2 = tf.nn.relu(logit2)
#         logit2 = fc(logit2, int(aux_num))
#         logit2 = tf.nn.softmax(logit2, dim=1) #dim or axis ?
#         return mask_softmask(logit2, mask, dim=1)



