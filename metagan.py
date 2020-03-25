import os
import numpy as np
import tensorflow as tf
import time

from modules.imutils import *
from modules.mdutils import *
from modules.fiutils import mkdirs
from modules.net_metagan  import  *

from support.mnist_classifier import classify
import glob
DISCRIMINATOR = 'discriminator'
DISCRIMINATOR_AUX = 'discriminator_aux'
GENERATOR = 'generator'
LABEL_GEN = 'labelgen'


class MetaGan(object):

    def __init__(self, model='metagan', \
                 is_train = 1,          \
                 aux_task = 2,          \
                 augment  = "rotation", \
                 lambda_gp = 1.0,       \
                 lr=2e-4, beta1 = 0.5, beta2 = 0.9,     \
                 noise_dim = 128,                       \
                 nnet_type='dcgan',                     \
                 loss_type='log',                       \
                 df_dim = 64, gf_dim = 64, ef_dim = 64, \
                 dataset = None, batch_size = 64,       \
                 nb_test_real = 10000,                  \
                 nb_test_fake = 5000,                   \
                 real_dir     = None,                   \
                 n_steps      = 300000,                 \
                 decay_step   = 10000, decay_rate = 1.0,\
                 log_interval = 10,                     \
                 out_dir = './output/',                 \
                 verbose = True,
                 psi=[10, 10]):
        """
        Initializing MS-Dist-GAN model
        """
        self.verbose      = verbose

        print('\n[metagan.py -- __init__] Intializing ... ')
        # dataset
        self.dataset   = dataset
        self.db_name   = self.dataset.db_name()
        print('[metagan.py -- __init__] db_name = %s' % (self.db_name))

        # training parameters
        self.model      = model
        self.is_train   = is_train
        self.lr         = lr
        self.beta1      = beta1
        self.beta2      = beta2
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.n_steps    = n_steps
        self.batch_size = self.dataset.mb_size()
        self.psi = psi

        if self.verbose == True:
            print('[metagan.py -- __init__] ' \
                              +  'model = %s, ' % (self.model)\
                              +  'lr    = %s, ' % (self.lr)\
                              +  'beta1 = %f, ' % (self.beta1)\
                              +  'beta2 = %f, ' % (self.beta2)\
                              +  'decay_step = %d, ' % (self.decay_step)\
                              +  'decay_rate = %f' % (self.decay_rate))
                               
            print('[metagan.py -- __init__] ' \
                              + 'n_steps = %d, ' % (self.n_steps)\
                              + 'batch_size = %d' % (self.batch_size))

        # architecture
        self.nnet_type = nnet_type
        self.loss_type = loss_type
        self.ef_dim    = ef_dim
        self.gf_dim    = gf_dim
        self.df_dim    = df_dim
        
        if self.verbose == True:
            print('[metagan.py -- __init__] ' \
                              + 'nnet_type = %s, ' % (self.nnet_type) \
                              + 'loss_type = %s'   % (self.loss_type))
            print('[metagan.py -- __init__] ' \
                              + 'gf_dim = %d, ' % (self.gf_dim) \
                              + 'df_dim = %d'   % (self.df_dim))
        
        # dimensions
        self.data_dim   = dataset.data_dim()
        self.data_shape = dataset.data_shape()
        self.noise_dim  = noise_dim
        
        if self.verbose == True:
            print('[metagan.py -- __init__] ' \
                           + 'data_dim  = %d, ' % (self.data_dim) \
                           + 'noise_dim = %d'   % (self.noise_dim))
            print('[metagan.py -- __init__] ' \
                           + 'data_shape = {}'.format(self.data_shape))

        # pamraeters
        self.lambda_gp  = lambda_gp
                        
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
                           + 'nb_test_real = %d, ' % (self.nb_test_real)  \
                           + 'nb_test_fake = %d, ' % (self.nb_test_fake))
            print("[metagan.py -- __init__] real_dir: " + self.real_dir)
            
        # others
        self.out_dir      = out_dir
        self.ckpt_dir     = out_dir + '/model/' # save pre-trained model
        self.log_file     = out_dir + '.txt'    # save log files
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
        return np.random.uniform(-1.0,1.0,size=[N, self.noise_dim])

    def create_discriminator(self):
        if self.nnet_type == "%s"%self.model and self.db_name in ['mnist']:
            return discriminator_dcgan_mnist        
        else:
            print('[metagan.py -- create_discriminator] The dataset %s are not supported by the network %s' %(self.db_name, self.nnet_type));

    def create_meta_discriminator(self):
        if self.nnet_type == 'metagan' and self.db_name in ['mnist']:
            return meta_discriminator_dcgan_mnist
        else:
            print('[metagan.py -- create_meta_discriminator] The dataset %s are not supported by the network %s' %(self.db_name, self.nnet_type))
            
    def create_generator(self):
        if self.nnet_type == 'metagan' and self.db_name in ['mnist']:
            return generator_dcgan_mnist            
        else:
            print('[metagan.py -- create_generator] The dataset %s are not supported by the network %s' %(self.db_name, self.nnet_type));

    def create_label_generator(self):
        if self.nnet_type == "metagan" and self.db_name in ['mnist']:
            return label_gen_dcgan_mnist
        else:
            print('[metagan.py -- create_label_generator] The dataset %s are not supported by the network %s' %(self.db_name, self.nnet_type));
                      
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

    @staticmethod
    def make_copy_params_op(v1_list, v2_list):
        """
        Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
        The ordering of the variables in the lists must be identical.
        """
        v1_list = list(sorted(v1_list, key=lambda v: v.name))
        v2_list = list(sorted(v2_list, key=lambda v: v.name))

        update_ops = []
        for v1, v2 in zip(v1_list, v2_list):
            op = v2.assign(v1)
            update_ops.append(op)
        return update_ops

    def create_model(self, aux=True):
        # aux = False
        self.X   = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim])
        self.z   = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dim])
        self.zn  = tf.placeholder(tf.float32, shape=[None, self.noise_dim]) # to generate flexible number of images
        
        self.iteration = tf.placeholder(tf.int32, shape=None)
                   
        # create generator
        with tf.variable_scope('generator'):
            self.G    = self.create_generator()
            self.X_f  = self.G(self.z,   self.data_shape, dim = self.gf_dim, reuse=False)   # to generate fake samples
            self.X_fn = self.G(self.zn,  self.data_shape, dim = self.gf_dim, reuse=True)    # to generate flexible number of fake images

        # with tf.variable_scope('labelgen'):
        #     self.L = self.create_label_generator()
        #     self.label_real_d = self.L(self.X, self.data_shape, tf.ones_like(self.d_real_prim), dim=self.df_dim, reuse=False, psi=self.psi)
        #     self.label_fake_d = self.L(self.X_f, self.data_shape, tf.zeros_like(self.d_real_prim), dim=self.df_dim, reuse=False, psi=self.psi)
        #     self.label_real_g = self.L(self.X_f, self.data_shape, tf.ones_like(self.d_real_prim), dim=self.df_dim, reuse=False, psi=self.psi)
        # create label generator
        if aux:
            with tf.variable_scope('labelgen'):
                self.L = self.create_label_generator()
                self.label_real_d = self.L(self.X, self.data_shape, np.ones(self.batch_size, dtype=np.int64), dim=self.df_dim,
                                           reuse=False, psi=self.psi)
                self.label_fake_d = self.L(self.X_f, self.data_shape, np.zeros(self.batch_size, dtype=np.int64), dim=self.df_dim,
                                           reuse=True, psi=self.psi)
                self.label_real_g = self.L(self.X_f, self.data_shape, np.ones(self.batch_size, dtype=np.int64), dim=self.df_dim,
                                           reuse=True, psi=self.psi)


        # create discriminator
        with tf.variable_scope('discriminator'):
            self.weights = construct_weights(self.df_dim, aux=aux)
            self.D = self.create_meta_discriminator()
            self.d_real_prim_logits,  self.d_real_aux_logits = self.D(self.X, self.data_shape, self.weights, reuse=False, aux=aux)
            self.d_fake_prim_logits,  self.d_fake_aux_logits = self.D(self.X_f, self.data_shape, self.weights, reuse=True, aux=aux)


            self.get_d_fake_prim_sig = tf.nn.sigmoid(self.d_fake_prim_logits)
            self.get_d_fake_aux_sig = self.d_fake_aux_logits

            self.get_d_real_prim_sig = tf.nn.sigmoid(self.d_real_prim_logits)
            self.get_d_real_aux_sig = self.d_real_aux_logits
            # self.get_d_real = (self.d_real_prim_logits, self.d_real_aux_logits)
            # only compute gradient penalty for discriminator loss when: lambda_gp > 0 to speed up the program
            if self.lambda_gp > 0.0:
                epsilon = tf.random_uniform(shape=[tf.shape(self.X)[0],1], minval=0., maxval=1.)
                interpolation = epsilon * self.X + (1 - epsilon) * self.X_f
                _,d_inter,_ = self.D(interpolation, self.data_shape, self.weights,reuse=True)
                gradients = tf.gradients([d_inter], [interpolation])[0]
                slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
                self.penalty = tf.reduce_mean((slopes - 1) ** 2)

            if aux:
                # auxiliary disriminator
                self.d_real_prim_logits_l, self.d_real_aux_logits_l = self.D(self.X ,self.data_shape, self.weights, reuse=True, name=DISCRIMINATOR,
                                                           aux=aux)
                self.d_fake_prim_logits_l, self.d_fake_aux_logits_l = self.D(self.X_f, self.data_shape, self.weights, reuse=True, name=DISCRIMINATOR,
                                                           aux=aux)

                self.d_real_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_prim_logits_l,
                                                                                       labels=tf.ones_like(
                                                                                           self.d_real_prim_logits)))

                self.d_real_aux_l = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_aux_logits_l,
                                                            labels=self.label_real_d))


                self.d_fake_l = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_prim_logits_l, labels=tf.zeros_like(
                        self.d_fake_prim_logits)))
                self.d_fake_aux_l = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_aux_logits_l, labels=self.label_fake_d))
                if self.lambda_gp > 0.0:
                    self.d_cost_gan_l = self.d_real_l + self.d_real_aux_l + self.d_fake_l + self.d_fake_aux_l + self.lambda_gp * self.penalty
                else:
                    self.d_cost_gan_l = self.d_real_l + self.d_real_aux_l + self.d_fake_l + self.d_fake_aux_l
                grads = tf.gradients(self.d_cost_gan_l, list(self.weights.values()))
                gradients = dict(zip(self.weights.keys(), grads))
                fast_weights = dict(
                    zip(self.weights.keys(), [self.weights[key] - self.lr * gradients[key] for key in self.weights.keys()]))
                self.d_real_prim_logits_l2, self.d_real_aux_logits_l2 = self.D(self.X, self.data_shape, fast_weights,
                                                                             reuse=True, name=DISCRIMINATOR,
                                                                             aux=aux)
                self.d_fake_prim_logits_l2, self.d_fake_aux_logits_l2 = self.D(self.X_f, self.data_shape, fast_weights,
                                                                             reuse=True, name=DISCRIMINATOR,
                                                                             aux=aux)

                # self.d_real_prim_logits_l, self.d_real_aux_logits_l = self.D(self.X ,self.data_shape, self.weights, reuse=False, name=DISCRIMINATOR,
                #                                            aux=aux)
                # self.d_fake_prim_logits_l, self.d_fake_aux_logits_l = self.D(self.X_f, self.data_shape, self.weights, reuse=True, name=DISCRIMINATOR,
                #                                            aux=aux)
                #
                # self.d_real_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_prim_logits_l,
                #                                                                        labels=tf.ones_like(
                #                                                                            self.d_real_prim_logits)))
                #
                # self.d_real_aux_l = tf.reduce_mean(
                #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_aux_logits_l,
                #                                             labels=self.label_real_d))
                #
                # self.d_fake_l = tf.reduce_mean(
                #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_prim_logits_l, labels=tf.zeros_like(
                #         self.d_fake_prim_logits)))
                # self.d_fake_aux_l = tf.reduce_mean(
                #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_aux_logits_l, labels=self.label_fake_d))
                # if self.lambda_gp > 0.0:
                #     self.d_cost_gan_l = self.d_real_l + self.d_real_aux_l + self.d_fake_l + self.d_fake_aux_l + self.lambda_gp * self.penalty
                # else:
                #     self.d_cost_gan_l = self.d_real_l + self.d_real_aux_l + self.d_fake_l + self.d_fake_aux_l
                # grads = tf.gradients(self.d_cost_gan_l, list(self.weights.values()))
                # gradients = dict(zip(self.weights.keys(), grads))
                # fast_weights = dict(
                #     zip(self.weights.keys(), [self.weights[key] - self.learning_rate * gradients[key] for key in self.weights.keys()]))
                # self.d_real_prim_logits_l2, self.d_real_aux_logits_l2 = self.D(self.X, self.data_shape, fast_weights,
                #                                                              reuse=False, name=DISCRIMINATOR,
                #                                                              aux=aux)
                # self.d_fake_prim_logits_l2, self.d_fake_aux_logits_l2 = self.D(self.X_f, self.data_shape, fast_weights,
                #                                                              reuse=True, name=DISCRIMINATOR,
                #                                                              aux=aux)



                
        # Original losses with log function
        if self.loss_type == 'log':
            # Discriminator Loss
            self.d_real   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_prim_logits, labels=tf.ones_like(self.d_real_prim_logits)))
            if aux:
                self.d_real_aux  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_aux_logits, labels=self.label_real_d))
            else:
                self.d_real_aux = tf.zeros_like(self.d_real)

            self.d_fake   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_prim_logits, labels=tf.zeros_like(self.d_fake_prim_logits)))
            if aux:
                self.d_fake_aux   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_aux_logits, labels=self.label_fake_d))
            else:
                self.d_fake_aux = tf.zeros_like(self.d_fake)

            if self.lambda_gp > 0.0:
                self.d_cost_gan  = self.d_real + self.d_real_aux + self.d_fake + self.d_fake_aux + self.lambda_gp * self.penalty
            else:
                self.d_cost_gan  = self.d_real + self.d_real_aux + self.d_fake + self.d_fake_aux
                    
            # Generator loss
            self.g_cost  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_prim_logits, labels=tf.ones_like(self.d_fake_prim_logits)))
            if aux:
                self.g_cost_aux  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_aux_logits, labels=self.label_real_g))
            else:
                self.g_cost_aux = tf.zeros_like(self.g_cost)
            # self.g_cost_gan  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_prim, labels=tf.ones_like(self.d_fake_prim)))
            self.g_cost_gan  = self.g_cost + self.g_cost_aux

            # Label Generator loss:
            if aux:
                self.l_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_prim_logits_l2,
                                                                                       labels=tf.ones_like(self.d_real_prim_logits)))
                self.label_real_d_mean = tf.reduce_mean(self.label_real_d, axis=0)
                self.cross_entropy_loss_real = self.label_real_d_mean * tf.log(self.label_real_d_mean + 1e-20)


                self.l_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_prim_logits_l2, labels=tf.zeros_like(
                        self.d_fake_prim_logits)))
                self.label_fake_d_mean = tf.reduce_mean(self.label_fake_d, axis=0)
                self.cross_entropy_loss_fake = self.label_fake_d_mean * tf.log(self.label_fake_d_mean + 1e-20)


                self.l_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_fake_prim_logits_l2, labels=tf.ones_like(
                        self.d_fake_prim_logits)))
                self.label_real_g_mean = tf.reduce_mean(self.label_real_g, axis=0)
                self.cross_entropy_loss_gen = self.label_real_g_mean * tf.log(self.label_real_g_mean + 1e-20)


                self.l_cost = self.l_real + self.l_fake + self.l_gen + self.cross_entropy_loss_real + \
                              self.cross_entropy_loss_fake + self.cross_entropy_loss_gen





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
            self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_step, self.decay_rate, staircase=True)

            self.opt_g = self.create_optimizer(self.g_cost, self.vars_g, self.learning_rate, self.beta1, self.beta2)
            self.opt_d = self.create_optimizer(self.d_cost, self.vars_d, self.learning_rate, self.beta1, self.beta2)
            if aux:
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
        fid = open(self.log_file,"w")
        fid_lb = open(self.log_file_lb, "w")
        
        # saver = tf.train.Saver(var_list = self.vars_g_save + self.vars_d_save, max_to_keep=1)
        saver = tf.train.Saver(var_list=self.vars_g_save + self.vars_d_save + self.vars_l_save, max_to_keep=20)
        step1=0

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
                sess.run([self.opt_d],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                
                # train generator
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                sess.run([self.opt_g],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                
                if step % self.log_interval == 0:
                    if self.verbose:
                       # compute losses for printing out
                       elapsed = int(time.time() - start)
                       
                       loss_d, loss_g = sess.run([self.d_cost, self.g_cost], feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                       output_str = '[metagan.py -- train] '\
                                        + 'step: %d, '         % (step)   \
                                        + 'D loss: %f, '       % (loss_d) \
                                        + 'G loss: %f, '       % (loss_g) \
                                        + 'time: %d s'         % (elapsed)                       


                       print(output_str)
                       fid.write(str(output_str)+'\n')
                       fid.flush()

                if step % (self.log_interval*1000) == 0:
                    # save real images
                    im_save_path = os.path.join(self.out_dir,'image_%d_real.jpg' % (step))
                    imsave_batch(mb_X, self.data_shape, im_save_path)
                    
                    # save generated images
                    im_save_path = os.path.join(self.out_dir,'image_%d_fake.jpg' % (step))
                    mb_X_f = sess.run(self.X_f,feed_dict={self.z: mb_z})
                    imsave_batch(mb_X_f, self.data_shape, im_save_path)
                                                                    
                if step % (self.log_interval*1000) == 0:
                    
                    def generate_and_save_real_images(real_dir, dataset, nb_test_real, batch_size, data_shape):
                        count = 0
                        for v in range(nb_test_real // batch_size + 1):
                            mb_X = dataset.next_batch()
                            im_real_save = np.reshape(mb_X,(-1, data_shape[0], data_shape[1], data_shape[2]))
                            for ii in range(np.shape(mb_X)[0]):
                                if count < nb_test_real:
                                    real_path = real_dir + '/image_%05d.jpg' % (np.min([v*batch_size + ii, nb_test_real]))
                                    imwrite(im_real_save[ii,:,:,:], real_path)                          
                                    count = count + 1
                                         
                    if step == 0:
                        #generate real samples to compute FID
                        if self.use_existing_real == False:
                           generate_and_save_real_images(self.real_dir, self.dataset, self.nb_test_real, self.batch_size, self.data_shape)
                    
                    #generate fake samples to compute FID        
                    fake_dir = self.out_dir + '/fake_%d/'%(step)
                    mkdirs(fake_dir)
                    fake_dir_pos = fake_dir + "positive/"
                    fake_dir_neg = fake_dir + "negative/"
                    mkdirs(fake_dir_neg)
                    mkdirs(fake_dir_pos)
                    if step >= 0:
                        count = 0
                        for v in range(self.nb_test_fake // self.batch_size + 1):
                            mb_z = self.sample_z(np.shape(mb_X)[0])
                            im_fake_save = sess.run(self.X_f,feed_dict={self.z: mb_z})
                            real_fake, label_guess = sess.run([self.get_d_fake_prim_sig, self.get_d_fake_aux_sig], feed_dict={self.X_f: im_fake_save})
                            real_fake = np.asarray(real_fake)
                            label_guess = np.asarray(label_guess)
                            im_fake_save = np.reshape(im_fake_save,(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
                            for ii in range(np.shape(mb_z)[0]):
                                if count < self.nb_test_fake:
                                    sig_real = real_fake[ii]
                                    # if sig_real < 0.5:
                                    #     is_real = False
                                    # else:
                                    #     is_real = True
                                    # if is_real:
                                    #     chosen_labels = label_guess[ii, 1 * self.psi[1]: self.psi[1] + self.psi[1]]
                                    # else:
                                    #     chosen_labels = label_guess[ii, 0 * self.psi[0]: 0 * self.psi[0] + self.psi[0]]
                                    chosen_labels_real = label_guess[ii, 1 * self.psi[1]: self.psi[1] + self.psi[1]]
                                    chosen_labels_fake = label_guess[ii, 0 * self.psi[0]: 0 * self.psi[0] + self.psi[0]]
                                    image_label_real = np.argmax(chosen_labels_real)
                                    image_label_fake = np.argmax(chosen_labels_fake)
                                    # image_label = np.argmax(chosen_labels)
                                    # if is_real:
                                    #     label_folder = fake_dir_pos + "/class_%d/"%(int(image_label))
                                    # else:
                                    #     label_folder = fake_dir_neg + "/class_%d/"%(int(image_label))
                                    label_folder_pos = fake_dir_pos + "/class_%d" % (int(image_label_real))
                                    label_folder_neg = fake_dir_neg + "/class_%d" % (int(image_label_fake))
                                    mkdirs(label_folder_pos)
                                    mkdirs(label_folder_neg)
                                    # mkdirs(label_folder)
                                    # fake_path = label_folder + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
                                    fake_path_pos = label_folder_pos + '/image_%05d_confidence%f.jpg' % (
                                        np.min([v * self.batch_size + ii, self.nb_test_fake]),
                                        float(chosen_labels_real[image_label_real]))
                                    fake_path_neg = label_folder_neg + '/image_%05d_confidence%f.jpg' % (
                                        np.min([v * self.batch_size + ii, self.nb_test_fake]),
                                        float(chosen_labels_fake[image_label_fake]))
                                    fake_path2 = fake_dir + '/image_%05d.jpg' % (
                                        np.min([v * self.batch_size + ii, self.nb_test_fake]))
                                    log_string_real = self.get_log_string_csv(
                                        np.min([v * self.batch_size + ii, self.nb_test_fake]), chosen_labels_real)
                                    log_string_fake = self.get_log_string_csv(
                                        np.min([v * self.batch_size + ii, self.nb_test_fake]), chosen_labels_real)
                                    f_real.write(log_string_real)
                                    f_real.flush()

                                    f_fake.write(log_string_fake)
                                    f_fake.flush()

                                    # fake_path_pos = label_folder_pos + '/image_%05d.jpg' % (
                                    #     np.min([v * self.batch_size + ii, self.nb_test_fake]))
                                    # fake_path_neg = label_folder_neg + '/image_%05d.jpg' % (
                                    #     np.min([v * self.batch_size + ii, self.nb_test_fake]))
                                    # fake_path2 = fake_dir + '/image_%05d.jpg' % (
                                    #     np.min([v * self.batch_size + ii, self.nb_test_fake]))
                                    # imwrite(im_fake_save[ii,:,:,:], fake_path)
                                    imwrite(im_fake_save[ii, :, :, :], fake_path_pos)
                                    imwrite(im_fake_save[ii, :, :, :], fake_path_neg)
                                    imwrite(im_fake_save[ii, :, :, :], fake_path2)
                                    count = count + 1




                if step%170==0:
                    print("Training Label generator")
                    for i in range(170):
                        mb_X = self.dataset.next_batch()
                        mb_z = self.sample_z(np.shape(mb_X)[0])
                        sess.run([self.opt_l], feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})

                        if step1 % self.log_interval == 0:
                            if self.verbose:
                                # compute losses for printing out
                                elapsed = int(time.time() - start)

                                # loss_d, loss_g = sess.run([self.d_cost, self.g_cost],
                                #                           feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                                loss_l = sess.run(self.l_cost,
                                                  feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                                output_str = '[metagan.py -- train label generator] ' \
                                             + 'step: %d, ' % (step1) \
                                             + 'L loss: %f, ' % (loss_l) \
                                             + 'time: %d s' % (elapsed)

                                print(output_str)
                                fid_lb.write(str(output_str) + '\n')
                                fid_lb.flush()
                        step1 += 1

                if step > 0 and step % 25000 == 0:
                    if not os.path.exists(self.ckpt_dir +'%d/'%(step)):
                        os.makedirs(self.ckpt_dir +'%d/'%(step))
                    save_path = saver.save(sess, '%s%d/epoch_%d.ckpt' % (self.ckpt_dir, step,step), global_step=step)
                    print('[metagan.py -- train D and G] the trained model is saved at: % s' % save_path)
                    # print('[metagan.py -- train D and G] the trained model is saved at: % s' % save_path)

    # def checkpoint_train(self):
    #     """
    #     Training the model
    #     """
    #     # aux = False
    #     run_config = tf.ConfigProto()
    #     run_config.gpu_options.allow_growth = True
    #
    #     self.log_file_lb = self.log_file.split(".")[0] + "_lb.txt"
    #
    #     saver = tf.train.Saver(var_list=self.vars_g_save + self.vars_d_save + self.vars_l_save, max_to_keep=1)
    #
    #     with tf.Session(config=run_config) as sess:
    #         sess.run(self.init)
    #         list_folders = glob.glob(os.path.join(self.ckpt_dir, "*"))
    #         for folder in list_folders:
    #             ckpt_name = os.path.join(folder, "checkpoint.ckpt")
    #             print(folder)
    #             iter = int(folder.split("/")[-1])
    #             saver.restore(sess, save_path=ckpt_name)
    #
    #             for v in range(self.nb_test_fake // self.batch_size + 1):
    #                 mb_z = self.sample_z(np.shape(self.batch_size)[0])
    #                 im_fake_save = sess.run(self.X_f, feed_dict={self.z: mb_z})
    #                 real_fake, label_guess = sess.run([self.get_d_fake_prim_sig, self.get_d_fake_aux_sig],
    #                                                   feed_dict={self.X_f: im_fake_save})
    #                 real_fake = np.asarray(real_fake)
    #                 label_guess = np.asarray(label_guess)
    #                 im_fake_save = np.reshape(im_fake_save,
    #                                           (-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
    #                 fake_dir = self.out_dir + '/fake_%d/' % (iter)
    #                 mkdirs(fake_dir)
    #                 fake_dir_pos = fake_dir + "positive/"
    #                 fake_dir_neg = fake_dir + "negative/"
    #                 mkdirs(fake_dir_neg)
    #                 mkdirs(fake_dir_pos)
    #
    #                 for ii in range(np.shape(mb_z)[0]):
    #                     if count < self.nb_test_fake:
    #                         chosen_labels_real = label_guess[ii, 1 * self.psi[1]: self.psi[1] + self.psi[1]]
    #                         chosen_labels_fake = label_guess[ii, 0 * self.psi[0]: 0 * self.psi[0] + self.psi[0]]
    #                         image_label_real = np.argmax(chosen_labels_real)
    #                         image_label_fake = np.argmax(chosen_labels_fake)
    #                         # image_label = np.argmax(chosen_labels)
    #                         # if is_real:
    #                         #     label_folder = fake_dir_pos + "/class_%d/"%(int(image_label))
    #                         # else:
    #                         #     label_folder = fake_dir_neg + "/class_%d/"%(int(image_label))
    #                         label_folder_pos = fake_dir_pos + "/class_%d" % (int(image_label_real))
    #                         label_folder_neg = fake_dir_neg + "/class_%d" % (int(image_label_fake))
    #                         mkdirs(label_folder_pos)
    #                         mkdirs(label_folder_neg)
    #                         # mkdirs(label_folder)
    #                         # fake_path = label_folder + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
    #                         fake_path_pos = label_folder_pos + '/image_%05d.jpg' % (
    #                             np.min([v * self.batch_size + ii, self.nb_test_fake]))
    #                         fake_path_neg = label_folder_neg + '/image_%05d.jpg' % (
    #                             np.min([v * self.batch_size + ii, self.nb_test_fake]))
    #                         fake_path2 = fake_dir + '/image_%05d.jpg' % (
    #                             np.min([v * self.batch_size + ii, self.nb_test_fake]))
    #                         # imwrite(im_fake_save[ii,:,:,:], fake_path)
    #                         imwrite(im_fake_save[ii, :, :, :], fake_path_pos)
    #                         imwrite(im_fake_save[ii, :, :, :], fake_path_neg)
    #                         imwrite(im_fake_save[ii, :, :, :], fake_path2)
    #                         count = count + 1

    @staticmethod
    def get_log_string_csv(image_id, np_array):
        log_string = "%05d, "%image_id
        for element in np_array:
            log_string += "%f, "%element
        log_string += "\n"
        return log_string

    def checkpoint_train(self):
        """
        Training the model
        """
        # aux = False
        # Generating real
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        real_test_dir = self.out_dir + '/real_test_discriminator/'
        mkdirs(real_test_dir)
        real_dir = os.path.join(real_test_dir, "real")
        mkdirs(real_dir)
        self.log_file_classification_real = os.path.join(real_dir, "positive_classification.csv")
        self.log_file_classification_fake = os.path.join(real_dir, "negative_classification.csv")
        self.log_file_classification_label = os.path.join(real_dir, "label_generator.csv")
        f_real = open(self.log_file_classification_real, "w")
        f_fake = open(self.log_file_classification_fake, "w")
        f_label = open(self.log_file_classification_label, "w")
        im_fake_dir = os.path.join(real_test_dir, "fake")
        mkdirs(im_fake_dir)
        self.log_file_classification_real_imf = os.path.join(im_fake_dir, "positive_classification.csv")
        self.log_file_classification_fake_imf = os.path.join(im_fake_dir, "negative_classification.csv")
        self.log_file_classification_label_imf = os.path.join(im_fake_dir, "label_generator.csv")
        f_real_imf = open(self.log_file_classification_real_imf, "w")
        f_fake_imf = open(self.log_file_classification_fake_imf, "w")
        f_label_imf = open(self.log_file_classification_label_imf, "w")
        # self.d_real_prim_logits_l, self.d_real_aux_logits_l = self.D(self.X, self.data_shape, self.weights, reuse=True,
        #                                                              name=DISCRIMINATOR,
        #                                                              aux=aux)
        # self.d_fake_prim_logits_l, self.d_fake_aux_logits_l = self.D(self.X_f, self.data_shape, self.weights,
        #                                                              reuse=True, name=DISCRIMINATOR,
        #                                                              aux=aux)

        # self.label_real_d = self.L(self.X, self.data_shape, np.ones(self.batch_size, dtype=np.int64), dim=self.df_dim,
        #                            reuse=False, psi=self.psi)
        # self.label_fake_d = self.L(self.X_f, self.data_shape, np.zeros(self.batch_size, dtype=np.int64),
        #                            dim=self.df_dim,
        #                            reuse=True, psi=self.psi)
        # self.label_real_g = self.L(self.X_f, self.data_shape, np.ones(self.batch_size, dtype=np.int64), dim=self.df_dim,
        #                            reuse=True, psi=self.psi)

        saver = tf.train.Saver(var_list=self.vars_g_save + self.vars_d_save + self.vars_l_save, max_to_keep=1)

        with tf.Session(config=run_config) as sess:
            sess.run(self.init)
            folder = os.path.join(self.ckpt_dir, "300000")
            # for folder in list_folders:
            ckpt_name = os.path.join(folder, "epoch_300000.ckpt")
            print(folder)
            iter = int(folder.split("/")[-1])
            saver.restore(sess, save_path=ckpt_name)
            count = 0
            for v in range(self.nb_test_fake // self.batch_size + 1):
                mb_X, mb_l = self.dataset.next_batch_with_labels()
                # mb_z = self.sample_z(np.shape(mb_X)[0])

                im_real_save = mb_X
                real_fake, label_guess = sess.run([self.get_d_real_prim_sig, self.get_d_real_aux_sig],
                                                  feed_dict={self.X: im_real_save})
                label_real_d = sess.run([self.label_real_d], feed_dict={self.X:im_real_save})
                real_fake = np.asarray(real_fake)
                label_guess = np.asarray(label_guess)
                label_real_d = np.asarray(label_real_d)
                im_real_save = np.reshape(im_real_save,
                                          (-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
                # fake_dir = self.out_dir + '/real_test_discriminator_%d/' % (iter)
                # mkdirs(fake_dir)
                # fake_dir_pos = fake_dir + "positive/"
                # fake_dir_neg = fake_dir + "negative/"
                # mkdirs(fake_dir_neg)
                # mkdirs(fake_dir_pos)


                for ii in range(np.shape(mb_X)[0]):
                    if count < self.nb_test_fake:
                        chosen_labels_real = label_guess[ii, 1 * self.psi[1]: self.psi[1] + self.psi[1]]
                        chosen_labels_fake = label_guess[ii, 0 * self.psi[0]: 0 * self.psi[0] + self.psi[0]]
                        label_gen = label_real_d[ii]
                        image_label_real = np.argmax(chosen_labels_real)
                        image_label_fake = np.argmax(chosen_labels_fake)

                        real_label = mb_l[ii]


                        fake_dir = os.path.join(real_dir, "class_%d"%int(real_label))
                        mkdirs(fake_dir)
                        fake_dir_pos = os.path.join(fake_dir, "positive")
                        fake_dir_neg = os.path.join(fake_dir, "negative")
                        mkdirs(fake_dir_neg)
                        mkdirs(fake_dir_pos)

                        # image_label = np.argmax(chosen_labels)
                        # if is_real:
                        #     label_folder = fake_dir_pos + "/class_%d/"%(int(image_label))
                        # else:
                        #     label_folder = fake_dir_neg + "/class_%d/"%(int(image_label))
                        label_folder_pos = fake_dir_pos + "/class_%d" % (int(image_label_real))
                        label_folder_neg = fake_dir_neg + "/class_%d" % (int(image_label_fake))
                        mkdirs(label_folder_pos)
                        mkdirs(label_folder_neg)
                        # mkdirs(label_folder)
                        # fake_path = label_folder + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
                        fake_path_pos = label_folder_pos + '/image_%05d_confidence%f.jpg' % (
                            np.min([v * self.batch_size + ii, self.nb_test_fake]), float(chosen_labels_real[image_label_real]))
                        fake_path_neg = label_folder_neg + '/image_%05d_confidence%f.jpg' % (
                            np.min([v * self.batch_size + ii, self.nb_test_fake]), float(chosen_labels_fake[image_label_fake]))
                        fake_path2 = fake_dir + '/image_%05d.jpg' % (
                            np.min([v * self.batch_size + ii, self.nb_test_fake]))
                        log_string_real = self.get_log_string_csv(np.min([v * self.batch_size + ii, self.nb_test_fake]), chosen_labels_real)
                        log_string_fake = self.get_log_string_csv(np.min([v * self.batch_size + ii, self.nb_test_fake]), chosen_labels_real)
                        log_string_label = self.get_log_string_csv(np.min([v * self.batch_size + ii, self.nb_test_fake]), label_gen)
                        f_real.write(log_string_real)
                        f_real.flush()

                        f_fake.write(log_string_fake)
                        f_fake.flush()

                        f_label.write(log_string_label)
                        f_label.flush()
                        # imwrite(im_fake_save[ii,:,:,:], fake_path)
                        imwrite(im_real_save[ii, :, :, :], fake_path_pos)
                        imwrite(im_real_save[ii, :, :, :], fake_path_neg)
                        # imwrite(im_real_save[ii, :, :, :], fake_path2)
                        count = count + 1

            print("Generating fake")
            for v in range(self.nb_test_fake // self.batch_size + 1):
                # mb_X, mb_l = self.dataset.next_batch_with_labels()
                mb_z = self.sample_z(self.batch_size)
                im_fake_save = sess.run(self.X_f, feed_dict={self.z: mb_z})
                real_fake, label_guess = sess.run([self.get_d_fake_prim_sig, self.get_d_fake_aux_sig],
                                                  feed_dict={self.X_f: im_fake_save})
                label_fake_d = sess.run([self.label_real_d], feed_dict={self.X: im_fake_save})
                real_fake = np.asarray(real_fake)
                label_guess = np.asarray(label_guess)
                im_real_save = np.reshape(im_fake_save,
                                          (-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
                label_fake_d = np.asarray(label_fake_d)


                # fake_dir = self.out_dir + '/real_test_discriminator_%d/' % (iter)
                # mkdirs(fake_dir)
                # fake_dir_pos = fake_dir + "positive/"
                # fake_dir_neg = fake_dir + "negative/"
                # mkdirs(fake_dir_neg)
                # mkdirs(fake_dir_pos)


                for ii in range(np.shape(im_fake_save)[0]):
                    if count < self.nb_test_fake:
                        chosen_labels_real = label_guess[ii, 1 * self.psi[1]: self.psi[1] + self.psi[1]]
                        chosen_labels_fake = label_guess[ii, 0 * self.psi[0]: 0 * self.psi[0] + self.psi[0]]
                        label_gen = label_fake_d[ii]
                        image_label_real = np.argmax(chosen_labels_real)
                        image_label_fake = np.argmax(chosen_labels_fake)


                        fake_dir_pos = os.path.join(im_fake_dir, "positive")
                        fake_dir_neg = os.path.join(im_fake_dir, "negative")
                        mkdirs(fake_dir_neg)
                        mkdirs(fake_dir_pos)

                        # image_label = np.argmax(chosen_labels)
                        # if is_real:
                        #     label_folder = fake_dir_pos + "/class_%d/"%(int(image_label))
                        # else:
                        #     label_folder = fake_dir_neg + "/class_%d/"%(int(image_label))
                        label_folder_pos = fake_dir_pos + "/class_%d" % (int(image_label_real))
                        label_folder_neg = fake_dir_neg + "/class_%d" % (int(image_label_fake))
                        mkdirs(label_folder_pos)
                        mkdirs(label_folder_neg)
                        # mkdirs(label_folder)
                        # fake_path = label_folder + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
                        fake_path_pos = label_folder_pos + '/image_%05d_confidence%f.jpg' % (
                            np.min([v * self.batch_size + ii, self.nb_test_fake]), float(chosen_labels_real[image_label_real]))
                        fake_path_neg = label_folder_neg + '/image_%05d_confidence%f.jpg' % (
                            np.min([v * self.batch_size + ii, self.nb_test_fake]), float(chosen_labels_fake[image_label_fake]))
                        fake_path2 = im_fake_dir + '/image_%05d.jpg' % (
                            np.min([v * self.batch_size + ii, self.nb_test_fake]))
                        log_string_real = self.get_log_string_csv(np.min([v * self.batch_size + ii, self.nb_test_fake]), chosen_labels_real)
                        log_string_fake = self.get_log_string_csv(np.min([v * self.batch_size + ii, self.nb_test_fake]), chosen_labels_real)
                        log_string_label = self.get_log_string_csv(np.min([v * self.batch_size + ii, self.nb_test_fake]), label_gen)
                        f_real_imf.write(log_string_real)
                        f_real_imf.flush()

                        f_fake_imf.write(log_string_fake)
                        f_fake_imf.flush()

                        f_label_imf.write(log_string_label)
                        f_label_imf.flush()
                        # imwrite(im_fake_save[ii,:,:,:], fake_path)
                        imwrite(im_real_save[ii, :, :, :], fake_path_pos)
                        imwrite(im_real_save[ii, :, :, :], fake_path_neg)
                        # imwrite(im_real_save[ii, :, :, :], fake_path2)
                        count = count + 1





# sig_real = real_fake[ii]
# if sig_real < 0.5:
#     is_real = False
# else:
#     is_real =True
# if is_real:
#     chosen_labels = label_guess[ii, 1*self.psi[1]: self.psi[1] + self.psi[1]]
# else:
#     chosen_labels = label_guess[ii, 0 * self.psi[0]: 0*self.psi[0] + self.psi[0]]
# image_label = np.argmax(chosen_labels)
# # if is_real:
# #     label_folder = fake_dir_pos + "/class_%d/"%(int(image_label))
# # else:
# #     label_folder = fake_dir_neg + "/class_%d/"%(int(image_label))
# label_folder_pos = fake_dir_pos + "/class_%d"%(int(image_label))
# label_folder_neg = fake_dir_neg + "/class_%d"%(int(image_label))
# mkdirs(label_folder_pos)
# mkdirs(label_folder_neg)
# # mkdirs(label_folder)
# # fake_path = label_folder + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
# fake_path_pos = label_folder_pos + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
# fake_path_neg = label_folder_neg + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
# fake_path2 = fake_dir + '/image_%05d.jpg' % (
#     np.min([v * self.batch_size + ii, self.nb_test_fake]))
# # imwrite(im_fake_save[ii,:,:,:], fake_path)
# imwrite(im_fake_save[ii,:,:,:], fake_path_pos)
# imwrite(im_fake_save[ii,:,:,:], fake_path_neg)
# imwrite(im_fake_save[ii,:,:,:], fake_path2)
# count = count + 1