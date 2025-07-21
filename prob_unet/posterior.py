import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Reshape
from tensorflow.keras.layers import Dense

class PosteriorNet(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = tf.keras.Sequential([
            Conv2D(32,3, padding='same', activation='relu'),
            Conv2D(32, 3, padding='same', activation='relu'),
            GlobalAveragePooling2D()
        ])
        self.mu = Dense(latent_dim, name = 'post_mu')
        self.logvar = Dense(latent_dim, name='post_logvar')

    def call(self, x, y):
        xy = tf.concat([x, y], axis = -1)
        xy = self.conv(xy)
        return self.mu(xy), self.logvar(xy)