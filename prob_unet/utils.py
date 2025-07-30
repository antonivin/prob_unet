import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm_notebook
import tifffile as tiff

tf.compat.v1.enable_eager_execution()

def get_data(path, im_height, im_width, train=True):
    print(path)
    ids = next(os.walk(path + "CompX/"))[2]
    X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    if train:
        Y = np.zeros((len(ids), im_height, im_width, 2), dtype=np.float32)
    print('Getting images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):

        bvpX = tiff.imread(path + 'CompX/' + id_)
        bvpY = tiff.imread(path + 'CompY/' + id_)
        bvpBC = tiff.imread(path + 'BC/' + id_)

        arrX = np.array(bvpX, dtype=np.float32)
        arrY = np.array(bvpY, dtype=np.float32)
        arrBC = np.array(bvpBC, dtype=np.float32)

        # Load masks
        if train:
            name = id_[:-4] + '.tif'  # -mask.png
            fdmX = tiff.imread(path + 'PR_049/FDMX/' + name)
            fdmY = tiff.imread(path + 'PR_049/FDMY/' + name)

            fdmArrX = np.array(fdmX, dtype=np.float32)
            fdmArrY = np.array(fdmY, dtype=np.float32)


        # Save images
        X[n, ..., 0] = arrX
        X[n, ..., 1] = arrY
        X[n, ..., 2] = arrBC
        if train:
            Y[n, ..., 0] = fdmArrX
            Y[n, ..., 1] = fdmArrY

    print(X.shape)
    print(Y.shape)
    print('Done!')
    if train:
        return X, Y
    else:
        return X

def reparameterize(mu, logvar):
    eps = tf.random.normal(tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps

def prob_unet_loss(y_true, y_pred, mu_q, logvar_q, mu_p, logvar_p):

    mse = tf.reduce_mean(tf.square(y_true - y_pred))