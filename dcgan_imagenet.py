import os, sys
import numpy as np
import argparse
from   test_net.dcgan_general import DCGAN
from   modules.dataset import Dataset
from   modules.eval import compute_fid_score
from   modules.fiutils import mkdirs

if __name__ == '__main__':
   
    '''
    ********************************************************************
    * Command-line arguments
    ********************************************************************
    ''' 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',       type=int,   default=0,               help='The ID of the specified GPU')
    parser.add_argument('--is_train',     type=int,   default=1,               help='1: Training, 2: Computing FID')
    parser.add_argument('--out_dir',      type=str,   default="./output/",     help='The ouput folder')
    parser.add_argument('--data_source',  type=str,   default="./data/imagenet32/", help='The place of storing MNIST dataset.')
    parser.add_argument('--nb_test_real', type=int,   default=10000,           help='Number of real samples to compute FID')
    parser.add_argument('--nb_test_fake', type=int,   default=10000,           help='Number of fake samples to compute FID')
    parser.add_argument('--n_steps',      type=int,   default=400000,          help='The number of training iterations')
    parser.add_argument('--noise_dim',    type=int,   default=100,             help='The dimension of latent noise')
    parser.add_argument('--batch_size',   type=int,   default=64,              help='Mini-batch size')
    parser.add_argument('--lambda_gp',    type=float, default=0.0,             help='The gradient penalty term')
    parser.add_argument('--percent',      type=float, default=100,             help='The percentage (%) of original dataset, i.e. default = 25%.')
    parser.add_argument('--real_dir',     type=str,   default="",              help='If the real samples are existing to compute FID, do not need to create new real ones')
    parser.add_argument('--db_name', type=str, default='imagenet_32', help='Database options: cifar10 or cifar100 or imagenet_32')
    parser.add_argument('--model', type=str, default="dcgan", help="model name")
    parser.add_argument('--colab', type=int, default=0, help="train on colab or not")
    parser.add_argument('--load', type=int, default=0, help="load 1 or re-train 0")
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
        
    '''
    ********************************************************************
    * Database and outputs
    ********************************************************************
    '''
    db_name       = opt.db_name
    out_dir       = opt.out_dir
    data_source   = os.path.join("./data", db_name)
    load = opt.load
    colab = opt.colab
    if colab == 1:
        out_dir = os.path.join("../drive/My Drive/", out_dir)
        data_source = os.path.join("../drive/My Drive", data_source)
    
    '''
    1: To train model and compute FID after training
    0: To compute FID of pre-trained model
    '''
    is_train      = opt.is_train 
                      
    '''
    Number of real or generated samples to compute FID scores
    '''                  
    nb_test_real = opt.nb_test_real
    nb_test_fake = opt.nb_test_fake
    real_dir     = opt.real_dir
                        
    '''
    ********************************************************************
    * Network architectures and objective losses
    ********************************************************************
    '''
    model     = opt.model   # the baseline model
    
    '''
    network architecture supports: 'dcgan'
    '''
    nnet_type = 'dcgan' # we only support small dcgan for mnist
    '''
    objective loss type supports: 'log'
    '''
    loss_type = 'log'   # we only support log function for mnist dataset
            
                    
    '''
    ********************************************************************
    * Training, network architectures and model parameters
    ********************************************************************
    '''    
    n_steps   = opt.n_steps    # the number of iterations
    noise_dim = opt.noise_dim  # the noise dimension
        
    '''
    The unit dimensions for network architectures.
    @df_dim: feature map unit for discriminator.
    @gf_dim: feature map unit for generator.
    @lr: learning rate
    @beta1, beta2 parameters for Adam optimizer
    '''
    df_dim = 64
    gf_dim = 64
    lr     = 2e-4
    beta1  = 0.5
    beta2  = 0.9
    
    '''
    The weight of gradient penalty term (<= 0: without gradient penalty)
    '''
    lambda_gp   = opt.lambda_gp
    
    '''
    Dataset parameters
    '''
    batch_size = opt.batch_size # bach size for each iteration.
    percent    = opt.percent    # the percentage of the dataset.
    
    '''
    ********************************************************************
    * Training and testing
    ********************************************************************
    '''
    ext_name = 'batch_size_%d_'    % (batch_size) + \
               'percent_%d_'       % (percent)    + \
               'n_steps_%d_'       % (n_steps)    + \
               'lambda_gp_%.02f'   % (lambda_gp)
        
    '''
    The ouput of the program
    '''
    model_dir = db_name + '_' + model + '_'  \
                                             + nnet_type + '_' \
                                             + loss_type + '_' \
                                             + ext_name
                                                  
    base_dir   = os.path.join(out_dir, model_dir, db_name)

    mkdirs(base_dir) # create base_dir folder

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source, \
                               batch_size=batch_size)
    dcgan = DCGAN(model=model, \
                  is_train=is_train, \
                  nb_test_real=nb_test_real, \
                  nb_test_fake=nb_test_fake, \
                  real_dir=real_dir, \
                  loss_type=loss_type, \
                  lambda_gp=lambda_gp, \
                  noise_dim=noise_dim, \
                  lr=lr, \
                  beta1=beta1, \
                  beta2=beta2, \
                  nnet_type=nnet_type, \
                  df_dim=df_dim, \
                  gf_dim=gf_dim, \
                  dataset=dataset, \
                  n_steps=n_steps, \
                  out_dir=base_dir)

    if is_train == 1:
        # setup gan model and train

        dcgan.train(load)
        compute_fid_score(dbname=db_name, \
                          input_dir=out_dir, \
                          model=model_dir, \
                          nb_train=nb_test_real, \
                          nb_test=nb_test_fake)

    elif is_train == 0:
        # compute fid score
        compute_fid_score(dbname=db_name, \
                          input_dir=out_dir, \
                          model=model_dir, \
                          nb_train=nb_test_real, \
                          nb_test=nb_test_fake)