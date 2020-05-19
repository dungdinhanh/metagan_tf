'''
************************************************************************
Implementation of SS/MS-DistGAN model by the authors of the paper:
"Self-supervised GAN: Analysis and Improvement with Multi-class Minimax
Game", NeurIPS 2019.
************************************************************************
'''

import os
import numpy as np
import tensorflow as tf
import time
import warnings
from modules.fiutils import mkdirs


DISCRIMINATOR = 'discriminator'
DISCRIMINATOR_AUX = 'discriminator_aux'
GENERATOR = 'generator'
LABEL_GEN = 'labelgen'

warnings.filterwarnings("ignore")

from modules.imutils import *
from modules.mdutils import *
from modules.vsutils import *
from modules.net_metagan import *
from modules.net_sngan import *
from modules.net_resnet import *

from support.mnist_classifier import classify


class MetaGAN(object):

    def __init__(self, model='metagan', \
                 is_train=1, \
                 aux_task=2, \
                 augment="rotation", \
                 lambda_gp=1.0, \
                 lr=2e-4, beta1=0.5, beta2=0.9, \
                 noise_dim=128, \
                 nnet_type='dcgan', \
                 loss_type='log', \
                 df_dim=64, gf_dim=64, ef_dim=64, \
                 dataset=None, batch_size=64, \
                 nb_test_real=10000, \
                 nb_test_fake=5000, \
                 real_dir=None, \
                 n_steps=300000, \
                 decay_step=10000, decay_rate=1.0, \
                 log_interval=10, \
                 out_dir='./output/', \
                 verbose=True,
                 psi=[10],
                 lamb_ent=0.2):
        """
        Initializing MS-Dist-GAN model
        """
        self.verbose = verbose

        print('\n[metagan.py -- __init__] Intializing ... ')
        # dataset
        self.dataset = dataset
        self.db_name = self.dataset.db_name()
        print('[metagan.py -- __init__] db_name = %s' % (self.db_name))

        # training parameters
        self.model = model
        self.is_train = is_train
        self.lr = lr
        self.lr_sd = 0.01
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.n_steps = n_steps
        self.batch_size = self.dataset.mb_size()
        self.psi = psi
        self.lamb_ent = lamb_ent

        if self.verbose == True:
            print('[metagan.py -- __init__] ' \
                  + 'model = %s, ' % (self.model) \
                  + 'lr    = %s, ' % (self.lr) \
                  + 'beta1 = %f, ' % (self.beta1) \
                  + 'beta2 = %f, ' % (self.beta2) \
                  + 'decay_step = %d, ' % (self.decay_step) \
                  + 'decay_rate = %f' % (self.decay_rate))

            print('[metagan.py -- __init__] ' \
                  + 'n_steps = %d, ' % (self.n_steps) \
                  + 'batch_size = %d' % (self.batch_size))

        # architecture
        self.nnet_type = nnet_type
        self.loss_type = loss_type
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        if self.verbose == True:
            print('[metagan.py -- __init__] ' \
                  + 'nnet_type = %s, ' % (self.nnet_type) \
                  + 'loss_type = %s' % (self.loss_type))
            print('[metagan.py -- __init__] ' \
                  + 'gf_dim = %d, ' % (self.gf_dim) \
                  + 'df_dim = %d' % (self.df_dim))

        # dimensions
        self.data_dim = dataset.data_dim()
        self.data_shape = dataset.data_shape()
        self.noise_dim = noise_dim

        if self.verbose == True:
            print('[metagan.py -- __init__] ' \
                  + 'data_dim  = %d, ' % (self.data_dim) \
                  + 'noise_dim = %d' % (self.noise_dim))
            print('[metagan.py -- __init__] ' \
                  + 'data_shape = {}'.format(self.data_shape))

        # pamraeters
        self.lambda_gp = lambda_gp

        if self.verbose == True:
            print('[metagan.py -- __init__] ' \
                  + 'lambda_gp = %f' % (self.lambda_gp))

        self.nb_test_real = nb_test_real
        self.nb_test_fake = nb_test_fake

        if (real_dir is not None) and real_dir:
            self.real_dir = real_dir
            self.use_existing_real = True
        else:
            self.real_dir = out_dir + '/real/'  # real dir
            self.use_existing_real = False
            mkdirs(self.real_dir)

        if self.verbose == True:
            print('[metagan.py -- __init__] Computing FID with: ' \
                  + 'nb_test_real = %d, ' % (self.nb_test_real) \
                  + 'nb_test_fake = %d, ' % (self.nb_test_fake))
            print("[metagan.py -- __init__] real_dir: " + self.real_dir)

        # others
        self.out_dir = out_dir
        self.ckpt_dir = out_dir + '/model/'  # save pre-trained model
        self.log_file = out_dir + '.txt'  # save log files
        self.log_interval = log_interval

        if self.verbose == True:
            print('[metagan.py -- __init__] ' \
                  + 'out_dir = {}'.format(self.out_dir))
            print('[metagan.py -- __init__] ' \
                  + 'ckpt_dir = {}'.format(self.ckpt_dir))
            print('[metagan.py -- __init__] ' \
                  + 'log_interval = {}'.format(self.log_interval))
            print('[metagan.py -- __init__] ' \
                  + 'verbose = {}'.format(self.verbose))

        print('[metagan.py -- __init__] Done.')

        self.create_model()



    def sample_z(self, N):
        return np.random.uniform(-1.0, 1.0, size=[N, self.noise_dim])

    def create_discriminator(self):
        if self.nnet_type == 'metagan' and self.db_name == 'mnist':
            return discriminator_dcgan_mnist
        # elif self.nnet_type == 'dcgan' and self.db_name == 'mnist-1k':
        #     return discriminator_dcgan_stacked_mnist
        # elif self.nnet_type == 'dcgan' and self.db_name == 'celeba':
        #     return discriminator_dcgan_celeba
        elif self.nnet_type == 'metagan' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
            return meta_discriminator_dcgan_cifar
        else:
            print('[metagan_general.py -- create_discriminator] The dataset %s are not supported by the network %s' % (
                self.db_name, self.nnet_type))

    def create_generator(self):
        if self.nnet_type == 'metagan' and self.db_name == 'mnist':
            return generator_dcgan_mnist
        # elif self.nnet_type == 'dcgan' and self.db_name == 'mnist-1k':
        #     return generator_dcgan_stacked_mnist
        # elif self.nnet_type == 'dcgan' and self.db_name == 'celeba':
        #     return generator_dcgan_celeba
        elif self.nnet_type == 'metagan' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
            return generator_dcgan_cifar
        # elif self.nnet_type == 'sngan' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
        #     return generator_sngan_cifar
            # elif self.nnet_type == 'sngan' and self.db_name == 'stl10':
        #     return generator_sngan_stl10
        # elif self.nnet_type == 'resnet' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
        #     return generator_resnet_cifar
        # elif self.nnet_type == 'resnet' and self.db_name == 'stl10':
        #     return generator_resnet_stl10
        else:
            print('[metagan_general.py -- create_generator] The dataset %s are not supported by the network %s' % (
            self.db_name, self.nnet_type))

    def create_label_generator(self):
        if self.nnet_type == "metagan" and self.db_name in ['mnist']:
            return label_gen_dcgan_mnist
        elif self.nnet_type == 'metagan' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
            return label_gen_cifar
        else:
            print('[metagan.py -- create_label_generator] The dataset %s are not supported by the network %s' % (
            self.db_name, self.nnet_type))

    def create_encoder(self):
        if self.nnet_type == 'metagan' and self.db_name == 'mnist':
            return encoder_dcgan_mnist
        # elif self.nnet_type == 'dcgan' and self.db_name == 'mnist-1k':
        #     return encoder_dcgan_stacked_mnist
        # elif self.nnet_type == 'dcgan' and self.db_name == 'celeba':
        #     return encoder_dcgan_celeba
        elif self.nnet_type == 'metagan' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
            return encoder_dcgan_cifar
        # elif self.nnet_type == 'sngan' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
        #     return encoder_dcgan_cifar
        # elif self.nnet_type == 'sngan' and self.db_name == 'stl10':
        #     return encoder_sngan_stl10
        # elif self.nnet_type == 'resnet' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
        #     return encoder_resnet_cifar
        # elif self.nnet_type == 'resnet' and self.db_name == 'stl10':
        #     return encoder_resnet_stl10
        else:
            print('[msdistgan.py -- create_encoder] The dataset %s are not supported by the network %s' % (
            self.db_name, self.nnet_type));

    def create_optimizer(self, loss, var_list, learning_rate, beta1, beta2):
        """Create the optimizer operation.

        :param loss: The loss to minimize.
        :param var_list: The variables to update.
        :param learning_rate: The learning rate.
        :param beta1: First moment hyperparameter of ADAM.
        :param beta2: Second moment hyperparameter of ADAM.
        :return: Optimizer operation.
        """
        return tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(loss, var_list=var_list)


    def create_model(self, aux=True):
        # aux = False
        self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim])
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dim])
        self.zn = tf.placeholder(tf.float32, shape=[None, self.noise_dim])  # to generate flexible number of images
        self.Y = tf.placeholder(tf.float32, shape=[self.batch_size, np.sum(self.psi)])
        self.l_rsd = tf.placeholder(tf.float32, shape=[1,])
        self.iteration = tf.placeholder(tf.int32, shape=None)

        # create generator
        with tf.variable_scope('generator'):
            self.G = self.create_generator()
            self.X_f = self.G(self.z, self.data_shape, dim=self.gf_dim, reuse=False)  # to generate fake samples
            self.X_fn = self.G(self.zn, self.data_shape, dim=self.gf_dim,
                               reuse=True)  # to generate flexible number of fake images

        with tf.variable_scope('labelgen'):
            self.L = self.create_label_generator()
            self.label_real_d = self.L(self.X, self.data_shape,
                                       dim=self.df_dim,
                                       reuse=False, psi=self.psi)
            self.label_fake_d = self.L(self.X_f, self.data_shape,
                                       dim=self.df_dim,
                                       reuse=True, psi=self.psi)


        # create discriminator
        with tf.variable_scope('discriminator'):
            self.weights = construct_weights_discriminator_cifar(self.df_dim)
            self.D = self.create_discriminator()
            self.d_real_prim_logits, self.d_real_aux_logits = self.D(self.X, self.data_shape, self.weights, reuse=False)
            self.d_fake_prim_logits, self.d_fake_aux_logits = self.D(self.X_f, self.data_shape, self.weights,
                                                                     reuse=True)

            self.get_d_fake_prim_sig = tf.nn.sigmoid(self.d_fake_prim_logits)
            self.get_d_fake_aux_sig = self.d_fake_aux_logits

            self.get_d_real_prim_sig = tf.nn.sigmoid(self.d_real_prim_logits)
            self.get_d_real_aux_sig = self.d_real_aux_logits
            if self.lambda_gp > 0.0:
                epsilon = tf.random_uniform(shape=[tf.shape(self.X)[0], 1], minval=0., maxval=1.)
                interpolation = epsilon * self.X + (1 - epsilon) * self.X_f
                _, d_inter, _ = self.D(interpolation, self.data_shape, self.weights, reuse=True)
                gradients = tf.gradients([d_inter], [interpolation])[0]
                slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
                self.penalty = tf.reduce_mean((slopes - 1) ** 2)
            # For 2nd derivatives
            self.d_real_prim_logits_l, self.d_real_aux_logits_l = self.D(self.X, self.data_shape, self.weights,
                                                                         reuse=True, name=DISCRIMINATOR)
            self.d_fake_prim_logits_l, self.d_fake_aux_logits_l = self.D(self.X_f, self.data_shape, self.weights,
                                                                         reuse=True, name=DISCRIMINATOR)

            self.d_real_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_prim_logits_l,
                                                                                   labels=tf.ones_like(
                                                                                       self.d_real_prim_logits)))

            self.d_real_aux_l = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_aux_logits_l,
                                                        labels=self.label_real_d))


            if self.lambda_gp > 0.0:
                self.d_cost_gan_l = self.d_real_l + self.d_real_aux_l + self.lambda_gp * self.penalty
            else:
                self.d_cost_gan_l = self.d_real_l + self.d_real_aux_l
            grads = tf.gradients(self.d_cost_gan_l, list(self.weights.values()))
            gradients = dict(zip(self.weights.keys(), grads))
            fast_weights = dict(
                zip(self.weights.keys(),
                    [self.weights[key] - self.l_rsd[0] * gradients[key] for key in self.weights.keys()]))
            self.d_real_prim_logits_l2, self.d_real_aux_logits_l2 = self.D(self.X, self.data_shape, fast_weights,
                                                                           reuse=True, name=DISCRIMINATOR)
            self.d_fake_prim_logits_l2, self.d_fake_aux_logits_l2 = self.D(self.X_f, self.data_shape, fast_weights,
                                                                           reuse=True, name=DISCRIMINATOR)


        # Original losses with log function
        if self.loss_type == 'log':
            # Discriminator Loss
            self.d_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_prim_logits,
                                                                                 labels=tf.ones_like(
                                                                                     self.d_real_prim_logits)))
            self.d_real_aux = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_aux_logits, labels=self.label_real_d))

            self.d_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_prim_logits,
                                                                                 labels=tf.zeros_like(
                                                                                     self.d_fake_prim_logits)))
            self.d_fake_aux = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_aux_logits, labels=self.label_fake_d))

            if self.lambda_gp > 0.0:
                self.d_cost_gan = self.d_real + self.d_real_aux + self.d_fake + self.d_fake_aux + self.lambda_gp * self.penalty
            else:
                self.d_cost_gan = self.d_real + self.d_real_aux + self.d_fake + self.d_fake_aux

            # Generator loss
            self.g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_prim_logits,
                                                                                 labels=tf.ones_like(
                                                                                     self.d_fake_prim_logits)))
            self.g_cost_gan = self.g_cost + self.d_fake_aux

            # Label Generator loss:
            self.l_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_prim_logits_l2,
                                                                                 labels=tf.ones_like(
                                                                                     self.d_real_prim_logits)))
            self.label_real_d_mean = tf.reduce_mean(self.label_real_d, axis=0)
            self.cross_entropy_loss_real = tf.reduce_sum(self.label_real_d_mean * tf.log(self.label_real_d_mean +
                                                                                         1e-20))

            self.l_cost = self.l_real + self.lamb_ent * self.cross_entropy_loss_real
            # self.l_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.label_real_d, labels=self.Y))
            # self.l_cost = self.l_real




        else:
            print('\n[metagan.py -- create_model] %s is not supported.' % (self.loss_type))

        self.d_cost = self.d_cost_gan
        self.g_cost = self.g_cost_gan

        # Create optimizers
        self.vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.vars_l = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=LABEL_GEN)
        print('[metagan.py -- create_model] ********** parameters of Generator **********')
        print(self.vars_g)
        print('[metagan.py -- create_model] ********** parameters of Discriminator **********')
        print(self.vars_d)

        self.vars_g_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        self.vars_d_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        self.vars_l_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=LABEL_GEN)

        if self.is_train == 1:

            # Setup for weight decay
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_step, self.decay_rate,
                                                            staircase=True)

            self.opt_g = self.create_optimizer(self.g_cost, self.vars_g, self.learning_rate, self.beta1, self.beta2)
            self.opt_d = self.create_optimizer(self.d_cost, self.vars_d, self.learning_rate, self.beta1, self.beta2)
            self.opt_l = self.create_optimizer(self.l_cost, self.vars_l, self.learning_rate, self.beta1, self.beta2)

        self.init = tf.global_variables_initializer()

    def train(self):
        """
        Training the model
        """
        # aux = False
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        self.log_file_lb = self.out_dir + "_lb.txt"
        fid = open(self.log_file, "w")
        fid_lb = open(self.log_file_lb, "w")

        # saver = tf.train.Saver(var_list = self.vars_g_save + self.vars_d_save, max_to_keep=1)
        saver = tf.train.Saver(var_list=self.vars_g_save + self.vars_d_save + self.vars_l_save, max_to_keep=20)
        step1 = 0

        log_file_classification_real = self.out_dir + "_positive.csv"
        log_file_classification_fake = self.out_dir + "_negative.csv"
        f_real = open(log_file_classification_real, "w")
        f_fake = open(log_file_classification_fake, "w")

        with tf.Session(config=run_config) as sess:

            start = time.time()
            sess.run(self.init)

            print("Training generator and discriminator")
            for step in range(self.n_steps + 1):

                # train discriminator
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                sess.run([self.opt_d], feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})

                # train generator
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                sess.run([self.opt_g], feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})

                if step % self.log_interval == 0:
                    if self.verbose:
                        # compute losses for printing out
                        elapsed = int(time.time() - start)

                        loss_d, loss_g = sess.run([self.d_cost, self.g_cost],
                                                  feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                        output_str = '[metagan.py -- train] ' \
                                     + 'step: %d, ' % (step) \
                                     + 'D loss: %f, ' % (loss_d) \
                                     + 'G loss: %f, ' % (loss_g) \
                                     + 'time: %d s' % (elapsed)

                        print(output_str)
                        fid.write(str(output_str) + '\n')
                        fid.flush()




                if step % (self.log_interval * 1000) == 0:
                    # save real images
                    im_save_path = os.path.join(self.out_dir, 'image_%d_real.jpg' % (step))
                    imsave_batch(mb_X, self.data_shape, im_save_path)

                    # save generated images
                    im_save_path = os.path.join(self.out_dir, 'image_%d_fake.jpg' % (step))
                    mb_X_f = sess.run(self.X_f, feed_dict={self.z: mb_z})
                    imsave_batch(mb_X_f, self.data_shape, im_save_path)

                if step % (self.log_interval * 1000) == 0:
                    def generate_and_save_real_images(real_dir, dataset, nb_test_real, batch_size, data_shape):
                        count = 0
                        for v in range(nb_test_real // batch_size + 1):
                            mb_X = dataset.next_batch()
                            im_real_save = np.reshape(mb_X, (-1, data_shape[0], data_shape[1], data_shape[2]))
                            for ii in range(np.shape(mb_X)[0]):
                                if count < nb_test_real:
                                    real_path = real_dir + '/image_%05d.jpg' % (
                                        np.min([v * batch_size + ii, nb_test_real]))
                                    imwrite(im_real_save[ii, :, :, :], real_path)
                                    count = count + 1

                    if step == 0:
                        # generate real samples to compute FID
                        if self.use_existing_real == False:
                            generate_and_save_real_images(self.real_dir, self.dataset, self.nb_test_real,
                                                          self.batch_size, self.data_shape)

                    # generate fake samples to compute FID
                    fake_dir = self.out_dir + '/fake_%d/' % (step)
                    mkdirs(fake_dir)
                    class_fake_dir = fake_dir + "classification/"
                    mkdirs(class_fake_dir)
                    if step >= 0:
                        count = 0
                        for v in range(self.nb_test_fake // self.batch_size + 1):
                            mb_z = self.sample_z(np.shape(mb_X)[0])
                            im_fake_save = sess.run(self.X_f, feed_dict={self.z: mb_z})
                            real_fake, label_guess = sess.run([self.get_d_fake_prim_sig, self.get_d_fake_aux_sig],
                                                              feed_dict={self.X_f: im_fake_save})
                            real_fake = np.asarray(real_fake)
                            label_guess = np.asarray(label_guess)
                            im_fake_save = np.reshape(im_fake_save,
                                                      (-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
                            for ii in range(np.shape(mb_z)[0]):
                                if count < self.nb_test_fake:
                                    chosen_labels = label_guess[ii]
                                    image_label = np.argmax(chosen_labels)
                                    real_percent = float(real_fake[ii])
                                    label_folder = class_fake_dir + "/class_%d" % (int(image_label))
                                    mkdirs(label_folder)
                                    fake_path = label_folder+ '/image_%05d_confidence%f_real%f.jpg' % (
                                        np.min([v * self.batch_size + ii, self.nb_test_fake]),
                                        float(chosen_labels[image_label]), real_percent)
                                    fake_path2 = fake_dir + '/image_%05d.jpg' % (
                                        np.min([v * self.batch_size + ii, self.nb_test_fake]))
                                    log_string_fake = self.get_log_string_csv(
                                        np.min([v * self.batch_size + ii, self.nb_test_fake]), image_label,
                                        real_percent * 100, chosen_labels)
                                    f_fake.write(log_string_fake)
                                    f_fake.flush()
                                    imwrite(im_fake_save[ii, :, :, :], fake_path)
                                    imwrite(im_fake_save[ii, :, :, :], fake_path2)
                                    count = count + 1


                if step % 170 == 0:

                    print("Training Label generator")
                    for i in range(170):
                        mb_X, mb_l = self.dataset.next_batch_with_labels()
                        mb_z = self.sample_z(np.shape(mb_X)[0])
                        mb_Y = np.zeros([np.shape(mb_l)[0], self.psi[0]], dtype=np.float32)
                        mb_Y[np.arange(np.shape(mb_l)[0]), mb_l[:,0]] =1

                        sess.run([self.opt_l], feed_dict={self.X: mb_X, self.Y: mb_Y,self.z: mb_z, self.iteration: step})

                        if step1 % self.log_interval == 0:
                            if self.verbose:
                                # compute losses for printing out
                                elapsed = int(time.time() - start)

                                # loss_d, loss_g = sess.run([self.d_cost, self.g_cost],
                                #                           feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                                loss_l = sess.run(self.l_cost,
                                                  feed_dict={self.X: mb_X, self.l_rsd: [self.lr_sd], self.iteration: step})
                                output_str = '[metagan.py -- train label generator] ' \
                                             + 'step: %d, ' % (step1) \
                                             + 'L loss: %f, ' % (loss_l) \
                                             + 'time: %d s' % (elapsed)

                                print(output_str)
                                fid_lb.write(str(output_str) + '\n')
                                fid_lb.flush()
                        step1 += 1
                    self.lr_sd = self.lr_sd * 0.5

                if step % 25000 == 0:
                    if not os.path.exists(self.ckpt_dir + '%d/' % (step)):
                        os.makedirs(self.ckpt_dir + '%d/' % (step))
                    save_path = saver.save(sess, '%s%d/epoch_%d.ckpt' % (self.ckpt_dir, step, step), global_step=step)
                    print('[metagan.py -- train D and G] the trained model is saved at: % s' % save_path)
                    # print('[metagan.py -- train D and G] the trained model is saved at: % s' % save_path)

    @staticmethod
    def get_log_string_csv(image_id, image_label, real_percent, np_array, real_label=None):
        if real_label is not None:
            log_string = "%05d, %d, %d, %f," % (image_id, image_label, real_label, real_percent)
        else:
            log_string = "%05d, %d, %f," % (image_id, image_label, real_percent)
        for element in np_array:
            log_string += "%f, " % element
        log_string += "\n"
        return log_string

    # def create_model(self):
    #
    #     self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim])
    #     self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dim])
    #     self.zn = tf.placeholder(tf.float32, shape=[None, self.noise_dim])  # to generate flexible number of images
    #
    #     self.iteration = tf.placeholder(tf.int32, shape=None)
    #
    #     # argument real samples for SS and MS task
    #     if self.ss_task == 1:  # SS task
    #         self.Xarg, self.larg, self.ridx = tf_argument_image_rotation(self.X, self.data_shape)
    #     elif self.ss_task == 2:  # MS task
    #         self.Xarg, self.larg, self.ridx = tf_argument_image_rotation_plus_fake(self.X, self.data_shape)
    #
    #     # create encoder
    #     with tf.variable_scope('encoder'):
    #         self.E = self.create_encoder()
    #         self.z_e = self.E(self.X, self.data_shape, self.noise_dim, dim=self.ef_dim, reuse=False)
    #
    #     # create generator
    #     with tf.variable_scope('generator'):
    #         self.G = self.create_generator()
    #         self.X_f = self.G(self.z, self.data_shape, dim=self.gf_dim, reuse=False)  # to generate fake samples
    #         self.X_r = self.G(self.z_e, self.data_shape, dim=self.gf_dim,
    #                           reuse=True)  # to generate reconstruction samples
    #         self.X_fn = self.G(self.zn, self.data_shape, dim=self.gf_dim,
    #                            reuse=True)  # to generate flexible number of fake images
    #
    #     # create discriminator
    #     with tf.variable_scope('discriminator'):
    #         self.D = self.create_discriminator()
    #         # D loss for SS/MS tasks
    #
    #         self.d_real_sigmoid, self.d_real_logit, self.f_real = self.D(self.X, self.data_shape, dim=self.df_dim,
    #                                                                      reuse=False)
    #         self.d_fake_sigmoid, self.d_fake_logit, self.f_fake = self.D(self.X_f, self.data_shape, dim=self.df_dim,
    #                                                                      reuse=True)
    #         self.d_recon_sigmoid, self.d_recon_logit, self.f_recon = self.D(self.X_r, self.data_shape,
    #                                                                         dim=self.df_dim, ss_task=self.ss_task,
    #                                                                         reuse=True)
    #
    #         # compute gradient penalty for discriminator loss
    #         epsilon = tf.random_uniform(shape=[tf.shape(self.X)[0], 1], minval=0., maxval=1.)
    #         interpolation = epsilon * self.X + (1 - epsilon) * self.X_f
    #         if self.ss_task == 1 or self.ss_task == 2:
    #             _, d_inter, _, _ = self.D(interpolation, self.data_shape, dim=self.df_dim, ss_task=self.ss_task,
    #                                       reuse=True)
    #         else:
    #             _, d_inter, _ = self.D(interpolation, self.data_shape, dim=self.df_dim, ss_task=self.ss_task,
    #                                    reuse=True)
    #         gradients = tf.gradients([d_inter], [interpolation])[0]
    #         slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
    #         self.penalty = tf.reduce_mean((slopes - 1) ** 2)
    #
    #         # compute SS loss
    #         if self.ss_task == 1:
    #
    #             # predict real/fake classes of argumented samples with classifier
    #             _, _, _, self.real_cls = self.D(self.Xarg, self.data_shape, dim=self.df_dim, ss_task=self.ss_task,
    #                                             reuse=True)
    #             _, _, _, self.fake_cls = self.D(self.Xarg_f, self.data_shape, dim=self.df_dim, ss_task=self.ss_task,
    #                                             reuse=True)
    #
    #             # SS loss for discriminator learning
    #             self.d_acc = tf.reduce_mean(
    #                 tf.nn.softmax_cross_entropy_with_logits(logits=self.real_cls, labels=self.larg))
    #
    #             # SS loss for generator learning
    #             self.g_real_acc = self.d_acc
    #             self.g_fake_acc = tf.reduce_mean(
    #                 tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_cls, labels=self.larg_f))
    #
    #             self.g_acc = tf.abs(self.g_fake_acc - self.g_real_acc, name='abs')
    #
    #         # compute MS loss
    #         elif self.ss_task == 2:
    #             # predict real/fake classes of argumented samples with classifier
    #             _, _, _, self.real_cls = self.D(self.Xarg, self.data_shape, dim=self.df_dim, ss_task=self.ss_task,
    #                                             reuse=True)
    #             _, _, _, self.fake_cls = self.D(self.Xarg_f, self.data_shape, dim=self.df_dim, ss_task=self.ss_task,
    #                                             reuse=True)
    #             _, _, _, self.mixe_cls = self.D(self.Xarg_mix, self.data_shape, dim=self.df_dim, ss_task=self.ss_task,
    #                                             reuse=True)
    #
    #             # SS loss for discriminator learning
    #             self.d_acc = tf.reduce_mean(
    #                 tf.nn.softmax_cross_entropy_with_logits(logits=self.mixe_cls, labels=self.larg_mix))
    #
    #             # SS loss for generator learning
    #             self.g_real_acc = tf.reduce_mean(
    #                 tf.nn.softmax_cross_entropy_with_logits(logits=self.real_cls, labels=self.larg))
    #             self.g_fake_acc = tf.reduce_mean(
    #                 tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_cls, labels=self.larg_f))
    #
    #             self.g_acc = tf.abs(self.g_fake_acc - self.g_real_acc, name='abs')
    #
    #     # reconstruction loss with data-latent distance (Dist-GAN)
    #     self.ae_loss = tf.reduce_mean(tf.square(self.f_real - self.f_recon))
    #     self.md_x = tf.reduce_mean(self.f_recon - self.f_fake)
    #     self.md_z = tf.reduce_mean(self.z_e - self.z) * self.lambda_w
    #     self.ae_reg = tf.square(self.md_x - self.md_z)
    #
    #     # Decay the weight of reconstruction for ResNet architecture
    #     t = tf.cast(self.iteration, tf.float32) / self.n_steps
    #     # mu = 0 if t <= N/2, mu in [0,0.05]
    #     # if N/2 < t and t < 3N/2 and mu = 0.05 if t > 3N/2
    #     self.mu = tf.maximum(tf.minimum((t * 0.1 - 0.05) * 2, 0.05), 0.0)
    #     w_real = 0.95 + self.mu
    #     w_recon = 0.05 - self.mu
    #     w_fake = 1.0
    #
    #     # Discriminator loss with log function
    #     if self.loss_type == 'log':
    #         # Loss
    #         self.d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logit,
    #                                                                              labels=tf.ones_like(
    #                                                                                  self.d_real_sigmoid)))
    #         self.d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit,
    #                                                                              labels=tf.zeros_like(
    #                                                                                  self.d_fake_sigmoid)))
    #         self.d_recon = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_recon_logit,
    #                                                                               labels=tf.ones_like(
    #                                                                                   self.d_recon_sigmoid)))
    #         self.d_cost_gan = 0.95 * self.d_real + 0.05 * self.d_recon + self.d_fake + self.lambda_p * self.penalty
    #
    #     # Discriminator loss with hinge loss function
    #     elif self.loss_type == 'hinge':
    #         if self.nnet_type == 'dcgan':
    #             self.d_cost_gan = -(w_real * tf.reduce_mean(tf.minimum(0., -1 + self.d_real_logit)) + \
    #                                 w_recon * tf.reduce_mean(tf.minimum(0., -1 + self.d_recon_logit)) + \
    #                                 tf.reduce_mean(
    #                                     tf.minimum(0., -1 - self.d_fake_logit)) + self.lambda_p * self.penalty)
    #         else:
    #             self.d_cost_gan = -(w_real * tf.reduce_mean(tf.minimum(0., -1 + self.d_real_sigmoid)) + \
    #                                 w_recon * tf.reduce_mean(tf.minimum(0., -1 + self.d_recon_sigmoid)) + \
    #                                 tf.reduce_mean(
    #                                     tf.minimum(0., -1 - self.d_fake_sigmoid)) + self.lambda_p * self.penalty)
    #
    #             # Reconstruction loss with data-latent distance (original from Dist-GAN)
    #     self.r_cost = self.ae_loss + self.lambda_r * self.ae_reg
    #
    #     # Generator loss by matching D scores (original from Dist-GAN)
    #     self.g_cost_gan = tf.abs(tf.reduce_mean(self.d_real_sigmoid - self.d_fake_sigmoid))
    #
    #     # Combine GAN task and SS task
    #     if self.ss_task > 0:
    #         self.d_cost = self.d_cost_gan + self.lambda_d * self.d_acc
    #         self.g_cost = self.g_cost_gan + self.lambda_g * self.g_acc
    #     else:
    #         self.d_cost = self.d_cost_gan
    #         self.g_cost = self.g_cost_gan
    #
    #     # Create optimizers
    #     if self.nnet_type == 'resnet':
    #
    #         self.vars_e = [var for var in tf.trainable_variables() if 'encoder' in var.name]
    #         self.vars_g = [var for var in tf.trainable_variables() if 'generator' in var.name]
    #         self.vars_d = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
    #
    #         print('[msdistgan.py -- create_model] ********** parameters of Encoder **********')
    #         print(self.vars_e)
    #         print('[msdistgan.py -- create_model] ********** parameters of Generator **********')
    #         print(self.vars_g)
    #         print('[msdistgan.py -- create_model] ********** parameters of Discriminator **********')
    #         print(self.vars_d)
    #
    #         self.vars_g_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    #         self.vars_d_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    #         self.vars_e_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
    #
    #         if self.is_train == 1:
    #             self.decay_rate = tf.maximum(0.,
    #                                          tf.minimum(1. - (tf.cast(self.iteration, tf.float32) / self.n_steps), 0.5))
    #
    #             self.opt_rec = tf.train.AdamOptimizer(learning_rate=self.lr * self.decay_rate, beta1=self.beta1,
    #                                                   beta2=self.beta2)
    #             self.opt_gen = tf.train.AdamOptimizer(learning_rate=self.lr * self.decay_rate, beta1=self.beta1,
    #                                                   beta2=self.beta2)
    #             self.opt_dis = tf.train.AdamOptimizer(learning_rate=self.lr * self.decay_rate, beta1=self.beta1,
    #                                                   beta2=self.beta2)
    #
    #             self.gen_gv = self.opt_gen.compute_gradients(self.g_cost, var_list=self.vars_g)
    #             self.dis_gv = self.opt_dis.compute_gradients(self.d_cost, var_list=self.vars_d)
    #             self.rec_gv = self.opt_rec.compute_gradients(self.r_cost, var_list=self.vars_e)
    #
    #             self.opt_r = self.opt_rec.apply_gradients(self.rec_gv)
    #             self.opt_g = self.opt_gen.apply_gradients(self.gen_gv)
    #             self.opt_d = self.opt_dis.apply_gradients(self.dis_gv)
    #
    #     else:
    #
    #         # Create optimizers
    #         self.vars_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    #         self.vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    #         self.vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    #
    #         print('[msdistgan.py -- create_model] ********** parameters of Encoder **********')
    #         print(self.vars_e)
    #         print('[msdistgan.py -- create_model] ********** parameters of Generator **********')
    #         print(self.vars_g)
    #         print('[msdistgan.py -- create_model] ********** parameters of Discriminator **********')
    #         print(self.vars_d)
    #
    #         self.vars_e_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
    #         self.vars_g_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    #         self.vars_d_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    #
    #         if self.is_train == 1:
    #
    #             # Setup for weight decay
    #             self.global_step = tf.Variable(0, trainable=False)
    #             self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_step,
    #                                                             self.decay_rate, staircase=True)
    #
    #             if self.db_name in ['mnist', 'mnist-1k']:
    #                 self.opt_r = self.create_optimizer(self.r_cost, self.vars_e + self.vars_g, self.learning_rate,
    #                                                    self.beta1, self.beta2)
    #             else:
    #                 self.opt_r = self.create_optimizer(self.r_cost, self.vars_e, self.learning_rate, self.beta1,
    #                                                    self.beta2)
    #             self.opt_g = self.create_optimizer(self.g_cost, self.vars_g, self.learning_rate, self.beta1, self.beta2)
    #             self.opt_d = self.create_optimizer(self.d_cost, self.vars_d, self.learning_rate, self.beta1, self.beta2)
    #
    #     self.init = tf.global_variables_initializer()

    # def train(self):
    #     """
    #     Training the model
    #     """
    #     run_config = tf.ConfigProto()
    #     run_config.gpu_options.allow_growth = True
    #
    #     fid = open(self.log_file, "w")
    #
    #     saver = tf.train.Saver(var_list=self.vars_e_save + self.vars_g_save + self.vars_d_save, max_to_keep=1)
    #
    #     with tf.Session(config=run_config) as sess:
    #
    #         start = time.time()
    #         sess.run(self.init)
    #
    #         for step in range(self.n_steps + 1):
    #
    #             # train auto-encoder
    #             mb_X = self.dataset.next_batch()
    #             mb_z = self.sample_z(np.shape(mb_X)[0])
    #             if step == 0:
    #                 # check f_feature size of discriminator
    #                 f_real = sess.run(self.f_real, feed_dict={self.X: mb_X, self.z: mb_z})
    #                 print(
    #                     '[msdistgan.py -- train] ***** SET CORRECT FEATURE SIZE ***** : feature_dim = {} if this value is different from the value set in main function'.format(
    #                         np.shape(f_real)[1]))
    #             X, X_f, X_r, X_reg = sess.run([self.X, self.X_f, self.X_r, self.ae_reg],
    #                                           feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
    #             sess.run([self.opt_r], feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
    #
    #             # train discriminator
    #             mb_X = self.dataset.next_batch()
    #             mb_z = self.sample_z(np.shape(mb_X)[0])
    #             sess.run([self.opt_d], feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
    #
    #             # train generator
    #             mb_X = self.dataset.next_batch()
    #             mb_z = self.sample_z(np.shape(mb_X)[0])
    #             sess.run([self.opt_g], feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
    #
    #             # compute losses to print
    #             if self.ss_task > 0:
    #                 loss_d, loss_d_gan, loss_d_acc, loss_g, loss_g_gan, loss_g_acc, loss_r = \
    #                     sess.run([self.d_cost, self.d_cost_gan, self.d_acc, self.g_cost, self.g_cost_gan, self.g_acc,
    #                               self.r_cost], feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
    #             else:
    #                 loss_d, loss_g, loss_r = sess.run([self.d_cost, self.g_cost, self.r_cost],
    #                                                   feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
    #
    #             if step % self.log_interval == 0:
    #                 if self.verbose:
    #                     elapsed = int(time.time() - start)
    #                     if self.ss_task > 0:
    #                         output_str = '[msdistgan.py -- train] step: {:4d}, D loss: {:8.4f}, D loss (gan): {:8.4f}, D loss (acc): {:8.4f} G loss: {:8.4f}, G loss (gan): {:8.4f}, G loss (acc): {:8.4f}, R loss: {:8.4f}, time: {:3d} s'.format(
    #                             step, loss_d, loss_d_gan, loss_d_acc, loss_g, loss_g_gan, loss_g_acc, loss_r, elapsed)
    #                     else:
    #                         output_str = '[msdistgan.py -- train] step: {:4d}, D loss: {:8.4f}, D loss (gan): {:8.4f}, D loss (acc): {:8.4f} G loss: {:8.4f}, G loss (gan): {:8.4f}, G loss (acc): {:8.4f}, R loss: {:8.4f}, time: {:3d} s'.format(
    #                             step, loss_d, loss_d, 0, loss_g, loss_g, 0, loss_r, elapsed)
    #                     print(output_str)
    #                     fid.write(str(output_str) + '\n')
    #                     fid.flush()
    #
    #             if step % (self.log_interval * 1000) == 0:
    #                 # save real images
    #                 im_save_path = os.path.join(self.out_dir, 'image_%d_real.jpg' % (step))
    #                 imsave_batch(mb_X, self.data_shape, im_save_path)
    #
    #                 # save generated images
    #                 im_save_path = os.path.join(self.out_dir, 'image_%d_fake.jpg' % (step))
    #                 mb_X_f = sess.run(self.X_f, feed_dict={self.z: mb_z})
    #                 imsave_batch(mb_X_f, self.data_shape, im_save_path)
    #
    #                 if self.ss_task > 0:
    #                     # save argumented images
    #                     Xarg = sess.run(self.Xarg, feed_dict={self.X: mb_X, self.z: mb_z})
    #                     im_save_path = os.path.join(self.out_dir, 'image_%d_real_argu.jpg' % (step))
    #                     imsave_batch(Xarg, self.data_shape, im_save_path)
    #
    #                     if self.ss_task == 2:
    #                         # save mix argumented images
    #                         Xarg_mix = sess.run(self.Xarg_mix, feed_dict={self.X: mb_X, self.z: mb_z})
    #                         im_save_path = os.path.join(self.out_dir, 'image_%d_mixe_argu.jpg' % (step))
    #                         imsave_batch(Xarg_mix, self.data_shape, im_save_path)
    #
    #             if step % (self.log_interval * 1000) == 0:
    #
    #                 if step == 0:
    #                     real_dir = self.out_dir + '/real/'
    #                     if not os.path.exists(real_dir):
    #                         os.makedirs(real_dir)
    #
    #                 fake_dir = self.out_dir + '/fake_%d/' % (step)
    #                 if not os.path.exists(fake_dir):
    #                     os.makedirs(fake_dir)
    #
    #                 # generate real samples to compute FID
    #                 if step == 0:
    #                     for v in range(self.nb_test_real // self.batch_size + 1):
    #                         mb_X = self.dataset.next_batch()
    #                         im_real_save = np.reshape(mb_X,
    #                                                   (-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
    #                         for ii in range(np.shape(mb_X)[0]):
    #                             real_path = real_dir + '/image_%05d.jpg' % (
    #                                 np.min([v * self.batch_size + ii, self.nb_test_real]))
    #                             imwrite(im_real_save[ii, :, :, :], real_path)
    #                 # generate fake samples to compute FID
    #                 elif step > 0:
    #                     for v in range(self.nb_test_fake // self.batch_size + 1):
    #                         mb_z = self.sample_z(np.shape(mb_X)[0])
    #                         im_fake_save = sess.run(self.X_f, feed_dict={self.z: mb_z})
    #                         im_fake_save = np.reshape(im_fake_save,
    #                                                   (-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
    #                         for ii in range(np.shape(mb_z)[0]):
    #                             fake_path = fake_dir + '/image_%05d.jpg' % (
    #                                 np.min([v * self.batch_size + ii, self.nb_test_fake]))
    #                             imwrite(im_fake_save[ii, :, :, :], fake_path)
    #
    #             if step > 0 and step % int(self.n_steps / 2) == 0:
    #                 if not os.path.exists(self.ckpt_dir + '%d/' % (step)):
    #                     os.makedirs(self.ckpt_dir + '%d/' % (step))
    #                 save_path = saver.save(sess, '%s%d/epoch_%d.ckpt' % (self.ckpt_dir, step, step))
    #                 print('[msdistgan.py -- train] the trained model is saved at: % s' % save_path)
