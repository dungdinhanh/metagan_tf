import os, sys
import numpy as np
import tensorflow as tf
import os.path

from modules.mdutils import *
from modules.imutils import *
from modules.fiutils import *

from support.classify_mnist_v2 import classify
from support.fid_score import fid

def generate_fake_samples(model, out_dir = None, ckpt_dir = None, ext = 'jpg', n_steps = 300000):
    
    #tf.reset_default_graph()
        
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    if out_dir is None:
       out_dir = model.out_dir + '/fake_samples/'
    if ckpt_dir is None:
       ckpt_dir = model.ckpt_dir + '/%d/'%(n_steps)
    
    print('[eval.py -- generate_fake_samples] generating samples at %s = ' % (out_dir))
                     
    mkdirs(out_dir)
    
    with tf.Session(config=run_config) as sess:
        flag = load_checkpoint(ckpt_dir, sess)
        if flag == True:            
            index=0
            for v in range(400):
                mb_z = model.sample_z(model.batch_size)
                im_fake_save = sess.run(model.X_f,feed_dict={model.z: mb_z})
                im_fake_save = np.reshape(im_fake_save,(-1, model.data_shape[0], model.data_shape[1], model.data_shape[2]))

                for ii in range(np.shape(mb_z)[0]):
                    fake_path = out_dir + '/image_%05d.' % (index) + ext
                    imwrite(im_fake_save[ii,:,:,:], fake_path)
                    index=index+1
    return out_dir

def generate_real_samples_fid(model, out_dir = None, ext = 'jpg', n_steps = 300000):
    
    print('[eval.py -- generate_real_samples_fid] generating %d real samples ...' % (model.nb_test_real))
    
    if out_dir is None:
       out_dir = model.out_dir + '/real_%d/' % (n_steps + 10000)
    
    data     = model.dataset.generate_nsamples(model.nb_test_real)
    real_dir = out_dir
            
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        im_save = np.reshape(data,(-1, model.data_shape[0], model.data_shape[1], model.data_shape[2]))
        for i in range(model.nb_test_real):
            real_path = real_dir + '/image_%05d.' % (i) + ext
            imwrite(im_save[i,:,:,:], real_path)
            print('[eval.py -- generate_real] generating real sample: %05d' %(i))
    else:
        print('[eval.py -- generate_real] warning: %s existed.' % (real_dir))
        
    return real_dir
        
def generate_fake_samples_fid(model, out_dir = None, ckpt_dir = None, ext = 'jpg', n_steps = 300000):
        
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    print('[eval.py -- generate_fake_samples_fid] generating %d fake samples ...' % (model.nb_test_fake))
    
    if out_dir is None:
       out_dir = model.out_dir + '/fake_%d/' % (n_steps + 10000)
    if ckpt_dir is None:
       ckpt_dir = model.ckpt_dir + '/%d/'%(n_steps)
       
    mkdirs(out_dir)
    
    print('[eval.py -- generate_fake_samples_fid] into: %s' % (out_dir))   
            
    with tf.Session(config=run_config) as sess:
        flag = load_checkpoint(ckpt_dir, sess)
        if flag == True:             
            for v in range(model.nb_test_fake // model.batch_size + 1):
                mb_z = model.sample_z(model.batch_size)
                im_fake_save = sess.run(model.X_f,feed_dict={model.z: mb_z})
                im_fake_save = np.reshape(im_fake_save,(-1, model.data_shape[0], model.data_shape[1], model.data_shape[2]))
                for ii in range(np.shape(mb_z)[0]):
                    fake_path = out_dir + '/image_%05d.' % (np.min([v*model.batch_size + ii, model.nb_test_fake])) + ext
                    imwrite(im_fake_save[ii,:,:,:], fake_path)
    return out_dir
                        
def compute_mode_kl(fake_source, is_train = 0, ext='jpg'):
    
    tf.reset_default_graph()
    
    model_path = "./support/pretrained-model/model.ckpt"
    modes = 1000
    classifier = classify()
    
    if is_train == 1:
        classifier.Train(save_path=model_path)
    else:
        _, Curr_Labels, _ , Curr_Labels2, _, Curr_Labels3 = classifier.Evaluate_Labels(source=fake_source, model_path=model_path, ext=ext)
        all_label= Curr_Labels*100+Curr_Labels2*10+Curr_Labels3
        hist, _ = np.histogram(all_label, modes)

        numModes= sum(hist>0)
        print('[eval.py -- compute_mode_kl] #modes = ', numModes)
        p=hist/float(np.sum(hist))

        ########## compute KL divergence
        KL=0
        q=1.0/1000
        for j in range(modes):
            if p[j]>0:          
                KL= KL+p[j]*math.log(p[j]/q)
        
        print('[eval.py -- compute_mode_kl] #KL = ', KL)
        '''
        plt.figure(1)
        plt.bar(range(modes),p)
        plt.title('histogram')
        plt.show()
        '''
        return numModes, KL

def compute_fid_score(dbname = 'cifar10', \
                      input_dir  = '../../gan/output/', \
                      model = 'cifar10_wgangp_dcgan_wdis_lp_10_300000', \
                      gth_path = None, gen_path = None,\
                      nb_train = 10000, nb_test = 10000,\
                      start = 10000, niters = 300000, step = 10000, \
                      re_est_gth = False,\
                      gpu = "0"):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
         
    mu_gth_file    = os.path.join(input_dir, model, 'mu_gth_' + dbname + '_%d.npy' % (nb_train))
    sigma_gth_file = os.path.join(input_dir, model,'sigma_gth_' + dbname + '_%d.npy' % (nb_train))

    print('[eval.py -- compute_fid_score] computing FID score ...')

    inception_path = fid.check_or_download_inception('/tmp') # download inception network

    logfile = os.path.join(input_dir, model, model + '_fid_%d_%d.txt'%(start,niters))
    print('[eval.py -- compute_fid_score] FID file: %s' % (logfile))
    
    fid_log = open(logfile, 'w')
    
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    if os.path.isfile(mu_gth_file) and os.path.isfile(sigma_gth_file) and re_est_gth:
        fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
        mu_gth = np.load(mu_gth_file)
        sigma_gth = np.load(sigma_gth_file)
        print('[eval.py -- compute_fid_score] gth_path = %s' % (gth_path))
        with tf.Session(config=run_config) as sess:
            sess.run(tf.global_variables_initializer())
            if gen_path is None:
                for i in range(start,niters+1,step):
                    gen_path = os.path.join(input_dir, model, dbname, 'fake_%d'%i) # set path to some generated images
                    print('[eval.py -- compute_fid_score] gen_path = %s'%(gen_path))
                    mu_gen, sigma_gen = fid._handle_path(gen_path, sess)
                    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_gth, sigma_gth)
                    strout = "step: %d - FID: %s" % (i, fid_value)
                    print(strout)
                    fid_log.write(strout + '\n')
                    fid_log.flush()
            else:
                mu_gen, sigma_gen = fid._handle_path(gen_path, sess)
                fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_gth, sigma_gth)
                strout = "step: %d - FID: %s" % (i, fid_value)
                
    else:
        fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('gth_path FID = %s' % (gth_path))
            print('gen_path FID = %s' % (gen_path))
            if gth_path is None:
               gth_path = os.path.join(input_dir, model, dbname, 'real') # set path to some ground truth images
            print('[eval.py -- compute_fid_score] gth_path = %s' % (gth_path))
            mu_gth, sigma_gth = fid._handle_path(gth_path, sess)
            if gen_path is None:
                for i in range(start,niters+1,step):
                    gen_path = os.path.join(input_dir, model, dbname, 'fake_%d'%i) # set path to some generated images
                    print('[eval.py -- compute_fid_score] gen_path = %s'%(gen_path))
                    mu_gen, sigma_gen = fid._handle_path(gen_path, sess)
                    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_gth, sigma_gth)
                    strout = "step: %d - FID: %s" % (i, fid_value)
                    print(strout)
                    fid_log.write(strout + '\n')
                    fid_log.flush()
            else:
                mu_gen, sigma_gen = fid._handle_path(gen_path, sess)
                fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_gth, sigma_gth)
                strout = "step: %d - FID: %s" % (i, fid_value)
        #np.save(mu_gth_file, mu_gth)
        #np.save(sigma_gth_file, sigma_gth)

    return fid_value

def compute_fid_score_mnist(dbname, input_dir, \
                            model, gth_path = None, gen_path = None, \
                            batch_size=64, \
                            nb_train = 10000, nb_test = 10000, \
                            start = 0, niters = 300000, step = 10000, \
                            ext='jpg', gpu = "0"):
    
    tf.reset_default_graph()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        
    print('[eval.py -- compute_fid_score_mnist] computing FID score ...')
    
    logfile = os.path.join(input_dir, model, model + '_fid_%d_%d.txt'%(start,niters))
    print('[eval.py -- compute_fid_score_mnist] FID file: %s' % (logfile))
    
    fid_log = open(logfile, 'w')
    
    if (gth_path is None) or (not gth_path):
       gth_path = os.path.join(input_dir, model, dbname, 'real/') # set path to some ground truth images
	   
           
    model_path = "./support/pretrained-model/model.ckpt"
    classifier = classify()
    classifier.Build_model()
    
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
            
    # compute statistic for real samples
    real_act = classifier.Compute_Activations(sess, source=gth_path, model_path=model_path, ext=ext, restore=True)
    
    m1, s1   = fid.calculate_activation_statistics2(real_act.reshape(nb_train,-1))
    
    for i in range(start,niters+1,step):
        if i == 0:
            continue

        gen_path = os.path.join(input_dir, model, dbname, 'fake_%d/'%i) # set path to some ground truth images
        
        # compute statistic for fake samples
        print('[eval.py -- compute_fid_score_mnist] gen_path = %s'%(gen_path))
        
        fake_act = classifier.Compute_Activations(sess, source=gen_path, model_path=model_path, ext=ext)
        
        m2, s2 = fid.calculate_activation_statistics2(fake_act.reshape(nb_test,-1))
    
        fid_value = fid.calculate_frechet_distance(m1, s1, m2, s2)
        
        strout = "step: %d - FID: %s" % (i, fid_value)
        print(strout)
        fid_log.write(strout + '\n')
        fid_log.flush()
    
    return fid_value
#
#
# def compute_fid_score_mnist2(dbname, input_dir, \
#                             model, gth_path=None, gen_path=None, \
#                             batch_size=64, \
#                             nb_train=10000, nb_test=10000, \
#                             start=0, niters=300000, step=10000, \
#                             ext='jpg', gpu="0"):
#     tf.reset_default_graph()
#
#     os.environ['CUDA_VISIBLE_DEVICES'] = gpu
#
#     print('[eval.py -- compute_fid_score_mnist] computing FID score ...')
#
#     logfile = os.path.join(input_dir, model, model + '_fid_%d_%d.txt' % (start, niters))
#     print('[eval.py -- compute_fid_score_mnist] FID file: %s' % (logfile))
#
#     fid_log = open(logfile, 'w')
#
#     if (gth_path is None) or (not gth_path):
#         gth_path = os.path.join(input_dir, model, dbname, 'real/')  # set path to some ground truth images
#
#     model_path = "./support/pretrained-model/model.ckpt"
#     classifier = classify()
#     classifier.Build_model()
#
#     run_config = tf.ConfigProto()
#     run_config.gpu_options.allow_growth = True
#     sess = tf.Session(config=run_config)
#
#     # compute statistic for real samples
#     real_act = classifier.Compute_Activations(sess, source=gth_path, model_path=model_path, ext=ext, restore=True)
#
#     m1, s1 = fid.calculate_activation_statistics2(real_act.reshape(nb_train, -1))
#
#     for i in range(start, niters + 1, step):
#         if i == 0:
#             continue
#
#         gen_path = os.path.join(input_dir, model, dbname, 'fake_%d/' % i)  # set path to some ground truth images
#
#         # compute statistic for fake samples
#         print('[eval.py -- compute_fid_score_mnist] gen_path = %s' % (gen_path))
#
#         fake_act = classifier.Compute_Activations(sess, source=gen_path, model_path=model_path, ext=ext)
#
#         m2, s2 = fid.calculate_activation_statistics2(fake_act.reshape(nb_test, -1))
#
#         fid_value = fid.calculate_frechet_distance(m1, s1, m2, s2)
#
#         strout = "step: %d - FID: %s" % (i, fid_value)
#         print(strout)
#         fid_log.write(strout + '\n')
#         fid_log.flush()
#
#     return fid_value
