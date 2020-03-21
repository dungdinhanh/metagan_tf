import numpy as np
import random
import math
from   skimage import io
import tensorflow as tf
import copy
import scipy

'''
************************************************************************
* IO image
************************************************************************
'''

def imread(path, is_grayscale=False):
    img = scipy.ndimage.imread(path).astype(np.float)
    return np.array(img)
    
def imwrite(image, path):
    """ save an [-1.0, 1.0] image """
    if image.ndim == 3 and image.shape[2] == 1: # grayscale images
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return io.imsave(path, image)
    
def immerge_row_col(N):
    c = int(np.floor(np.sqrt(N)))
    for v in range(c,N):
        if N % v == 0:
            c = v
            break
    r = N / c
    return r, c
    
def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)
    @images: is in shape of N * H * W(* C=1 or 3)
    """
    row = int(row)
    col = int(col)
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image
    return img
    
def imsave_batch(X, data_shape, im_save_path):
    im_save = np.reshape(X,(-1, data_shape[0], data_shape[1], data_shape[2]))
    ncols, nrows = immerge_row_col(np.shape(im_save)[0])
    im_merge = immerge(im_save, ncols, nrows)
    imwrite(im_merge, im_save_path)
