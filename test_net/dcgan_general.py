import os
import numpy as np
import tensorflow as tf
import time

from modules.imutils import *
from modules.mdutils import *
from modules.fiutils import mkdirs
from modules.net_dcgan  import  *

from support.mnist_classifier import classify

class DCGAN(object):

    def __init__(self, model='dcgan', \
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
                 verbose = True):
        """
        Initializing MS-Dist-GAN model
        """
        self.verbose      = verbose
        
        print('\n[dcgan.py -- __init__] Intializing ... ')
        # dataset
        self.dataset   = dataset
        self.db_name   = self.dataset.db_name()
        print('[dcgan.py -- __init__] db_name = %s' % (self.db_name))

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
        
        if self.verbose == True:
            print('[dcgan.py -- __init__] ' \
                              +  'model = %s, ' % (self.model)\
                              +  'lr    = %s, ' % (self.lr)\
                              +  'beta1 = %f, ' % (self.beta1)\
                              +  'beta2 = %f, ' % (self.beta2)\
                              +  'decay_step = %d, ' % (self.decay_step)\
                              +  'decay_rate = %f' % (self.decay_rate))
                               
            print('[dcgan.py -- __init__] ' \
                              + 'n_steps = %d, ' % (self.n_steps)\
                              + 'batch_size = %d' % (self.batch_size))

        # architecture
        self.nnet_type = nnet_type
        self.loss_type = loss_type
        self.ef_dim    = ef_dim
        self.gf_dim    = gf_dim
        self.df_dim    = df_dim
        
        if self.verbose == True:
            print('[dcgan.py -- __init__] ' \
                              + 'nnet_type = %s, ' % (self.nnet_type) \
                              + 'loss_type = %s'   % (self.loss_type))
            print('[dcgan.py -- __init__] ' \
                              + 'gf_dim = %d, ' % (self.gf_dim) \
                              + 'df_dim = %d'   % (self.df_dim))
        
        # dimensions
        self.data_dim   = dataset.data_dim()
        self.data_shape = dataset.data_shape()
        self.noise_dim  = noise_dim
        
        if self.verbose == True:
            print('[dcgan.py -- __init__] ' \
                           + 'data_dim  = %d, ' % (self.data_dim) \
                           + 'noise_dim = %d'   % (self.noise_dim))
            print('[dcgan.py -- __init__] ' \
                           + 'data_shape = {}'.format(self.data_shape))

        # pamraeters
        self.lambda_gp  = lambda_gp
                        
        if self.verbose == True:
            print('[dcgan.py -- __init__] ' \
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
            print('[dcgan.py -- __init__] Computing FID with: ' \
                           + 'nb_test_real = %d, ' % (self.nb_test_real)  \
                           + 'nb_test_fake = %d, ' % (self.nb_test_fake))
            print("[dcgan.py -- __init__] real_dir: " + self.real_dir)
            
        # others
        self.out_dir      = out_dir
        self.ckpt_dir     = out_dir + '/model/' # save pre-trained model
        self.log_file     = out_dir + '.txt'    # save log files
        self.log_interval = log_interval
                
        if self.verbose == True:
            print('[dcgan.py -- __init__] ' \
                           + 'out_dir = {}'.format(self.out_dir))
            print('[dcgan.py -- __init__] ' \
                           + 'ckpt_dir = {}'.format(self.ckpt_dir))
            print('[dcgan.py -- __init__] ' \
                        + 'log_interval = {}'.format(self.log_interval))
            print('[dcgan.py -- __init__] ' \
                        + 'verbose = {}'.format(self.verbose))
        
        print('[dcgan.py -- __init__] Done.')

        self.create_model()

    def sample_z(self, N):
        return np.random.uniform(-1.0,1.0,size=[N, self.noise_dim])

    def create_discriminator(self):
        if self.nnet_type == 'dcgan' and self.db_name in ['mnist']:
            return discriminator_dcgan_mnist
        elif self.nnet_type == 'dcgan' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
            return discriminator_dcgan_cifar
        else:
            print('[dcgan.py -- create_discriminator] The dataset %s are not supported by the network %s' %(self.db_name, self.nnet_type));
            
    def create_generator(self):
        if self.nnet_type == 'dcgan' and self.db_name in ['mnist']:
            return generator_dcgan_mnist
        elif self.nnet_type == 'dcgan' and self.db_name in ['cifar10', 'cifar100', 'imagenet_32']:
            return generator_dcgan_cifar
        else:
            print('[dcgan.py -- create_generator] The dataset %s are not supported by the network %s' %(self.db_name, self.nnet_type));
                      
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

    def create_model(self):

        self.X   = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim])
        self.z   = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dim])
        self.zn  = tf.placeholder(tf.float32, shape=[None, self.noise_dim]) # to generate flexible number of images
        
        self.iteration = tf.placeholder(tf.int32, shape=None)
                   
        # create generator
        with tf.variable_scope('generator'):
            self.G    = self.create_generator()
            self.X_f  = self.G(self.z,   self.data_shape, dim = self.gf_dim, reuse=False)   # to generate fake samples
            self.X_fn = self.G(self.zn,  self.data_shape, dim = self.gf_dim, reuse=True)    # to generate flexible number of fake images
                       
        # create discriminator
        with tf.variable_scope('discriminator'):
            self.D   = self.create_discriminator()
            self.d_real_sigmoid,  self.d_real_logit,_ = self.D(self.X,   self.data_shape, dim = self.df_dim, reuse=False)
            self.d_fake_sigmoid,  self.d_fake_logit,_ = self.D(self.X_f, self.data_shape, dim = self.df_dim, reuse=True)
                                
            # only compute gradient penalty for discriminator loss when: lambda_gp > 0 to speed up the program
            if self.lambda_gp > 0.0:
                epsilon = tf.random_uniform(shape=[tf.shape(self.X)[0],1], minval=0., maxval=1.)
                interpolation = epsilon * self.X + (1 - epsilon) * self.X_f
                _,d_inter,_ = self.D(interpolation, self.data_shape, dim = self.df_dim, ss_task = self.aux_task, reuse=True)
                gradients = tf.gradients([d_inter], [interpolation])[0]
                slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
                self.penalty = tf.reduce_mean((slopes - 1) ** 2)
                
        # Original losses with log function
        if self.loss_type == 'log':
            # Discriminator Loss
            self.d_real   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logit, labels=tf.ones_like(self.d_real_sigmoid)))
            self.d_fake   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit, labels=tf.zeros_like(self.d_fake_sigmoid)))
            if self.lambda_gp > 0.0:
                self.d_cost_gan  = self.d_real + self.d_fake + self.lambda_gp * self.penalty
            else:
                self.d_cost_gan  = self.d_real + self.d_fake
                    
            # Generator loss
            self.g_cost_gan  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit, labels=tf.ones_like(self.d_fake_sigmoid)))
        else:
            print('\n[dcgan.py -- create_model] %s is not supported.' % (self.loss_type))
                                            
        self.d_cost = self.d_cost_gan    
        self.g_cost = self.g_cost_gan
            
        # Create optimizers
        self.vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                                
        print('[dcgan.py -- create_model] ********** parameters of Generator **********')
        print(self.vars_g)
        print('[dcgan.py -- create_model] ********** parameters of Discriminator **********')
        print(self.vars_d)
        
        self.vars_g_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        self.vars_d_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        
        if self.is_train == 1:
            
            # Setup for weight decay
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_step, self.decay_rate, staircase=True)

            self.opt_g = self.create_optimizer(self.g_cost, self.vars_g, self.learning_rate, self.beta1, self.beta2)
            self.opt_d = self.create_optimizer(self.d_cost, self.vars_d, self.learning_rate, self.beta1, self.beta2)
        
        self.init = tf.global_variables_initializer()

    def train(self):
        """
        Training the model
        """
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        
        fid = open(self.log_file,"w")
        
        saver = tf.train.Saver(var_list = self.vars_g_save + self.vars_d_save, max_to_keep=1)
       
        with tf.Session(config=run_config) as sess:
            
            start = time.time()
            sess.run(self.init)
                       
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
                       output_str = '[dcgan.py -- train] '\
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
                    if step >= 0:
                        count = 0
                        for v in range(self.nb_test_fake // self.batch_size + 1):
                            mb_z = self.sample_z(np.shape(mb_X)[0])
                            im_fake_save = sess.run(self.X_f,feed_dict={self.z: mb_z})
                            im_fake_save = np.reshape(im_fake_save,(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
                            for ii in range(np.shape(mb_z)[0]):
                                if count < self.nb_test_fake:
                                    fake_path = fake_dir + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
                                    imwrite(im_fake_save[ii,:,:,:], fake_path)
                                    count = count + 1

                if step > 0 and step % int(self.n_steps/2) == 0:
                    if not os.path.exists(self.ckpt_dir +'%d/'%(step)):
                        os.makedirs(self.ckpt_dir +'%d/'%(step))
                    save_path = saver.save(sess, '%s%d/epoch_%d.ckpt' % (self.ckpt_dir, step,step))
                    print('[dcgan.py -- train] the trained model is saved at: % s' % save_path)
