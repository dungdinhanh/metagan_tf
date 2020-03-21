from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from functools import partial
from modules.ops import *

'''
************************************************************************
* The small DC-GAN architecture for MNIST (28 x 28 x 1)
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
