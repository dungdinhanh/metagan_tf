import math
import os
import sys

import numpy as np
import tensorflow as tf

from modules.mnist    import load_mnist, load_small_mnist, stacked_mnist_batch
from modules.imutils  import imread
from modules.dbutils  import list_dir, prepare_image_list

class Dataset(object):

    def __init__(self, name='mnist', source='./data/mnist/', batch_size = 64, percent = 100, seed = 0):

        self.name            = name
        self.source          = source
        self.batch_size      = batch_size
        self.seed            = seed
        np.random.seed(seed) # To make your "random" minibatches the same for experiments

        self.count           = 0
        self.percent         = percent
        
        # the dimension of vectorized and orignal data samples
        if self.name in ['mnist']:
            self.data_vec_dim = 784   #28x28
            self.data_origin_shape = [28, 28, 1]
        else:
            self.data_vec_dim = 0     #0
            self.data_origin_shape = [0, 0, 0]
            print('[dataset.py - __init__] dbname = %s is unrecognized.\n' % (self.name))
            exit()
        
        tf.set_random_seed(self.seed)  # Fix the random seed for randomized tensorflow operations.
        
        if name == 'mnist':
            self.data, self.labels = load_mnist(self.source, self.percent)
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
        
    def db_name(self):
        return self.name
        
    def db_source(self):
        return self.source    

    def data_dim(self):
        return self.data_vec_dim
        
    def data_shape(self):
        return self.data_origin_shape
                    
    def mb_size(self):
        return self.batch_size

    def next_batch(self):

        if self.name in ['mnist']:
            if self.count == len(self.minibatches):
                self.count = 0
                self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
            batch = self.minibatches[self.count]
            self.count = self.count + 1
            return batch.T        
               

    def fixed_mini_batches(self, mini_batch_size = 64):
        if self.name in ['mnist']:
            X = self.data
            Y = self.labels
            m = X.shape[0]
            mini_batches = []
            mini_labels  = []
            num_complete_minibatches = int(math.floor(m/self.batch_size)) # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, num_complete_minibatches):
                mini_batch = X[k * self.batch_size : (k+1) * self.batch_size, :]
                mini_label = Y[k * self.batch_size : (k+1) * self.batch_size]
                mini_batches.append(mini_batch)
                mini_labels.append(mini_label)

            # Handling the end case (last mini-batch < mini_batch_size)
            #if m % mini_batch_size != 0:
            #    mini_batch = X[num_complete_minibatches * self.batch_size : m, :]
            #    mini_label = Y[num_complete_minibatches * self.batch_size : m]
            #    mini_batches.append(mini_batch)
            #    mini_labels.append(mini_label)
            
            return mini_batches, mini_labels
            
    # Random minibatches for training
    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        if self.name in ['mnist']:
            m = X.shape[1]    # number of training examples
            mini_batches = []
            mini_labels  = []
                            
            # Step 1: Shuffle (X, Y)
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation]

            # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
            num_complete_minibatches = int(math.floor(m/self.batch_size)) # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[:, k * self.batch_size : (k+1) * self.batch_size]
                mini_batches.append(mini_batch_X)
                mini_batch_Y = shuffled_Y[:, k * self.batch_size : (k+1) * self.batch_size]
                mini_labels.append(mini_batch_Y)
            
            # Handling the end case (last mini-batch < mini_batch_size)
            #if m % mini_batch_size != 0:
            #    mini_batch_X = shuffled_X[:, num_complete_minibatches * self.batch_size : m]
            #    mini_batches.append(mini_batch_X)
            
            return mini_batches, mini_labels
                
    # generate real samples for fid
    def generate_nsamples(self, nsamples):
        if nsamples == -1 or nsamples == np.shape(self.data)[0]:
           return self.data
        else:
           cnt = 0
           data = []
           while cnt < nsamples:
                X_mb = self.next_batch()
                data.append(X_mb)
                cnt = cnt + np.shape(X_mb)[0]
           data = np.concatenate(data, axis=0)
           return data[0:nsamples,:]
