import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, Dense
from tensorflow.keras import backend as K


def upsample_block(x, skip, filters, kernel_size = 3):
    x = Conv2DTranspose(filters, kernel_size, strides=2, padding='same')(x)
    x = tf.concat([x, skip], axis=-1)

    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer="he_normal", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer="he_normal", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

class decoder(tf.keras.layers.Layer):
    def __init__(self, filters=[256, 128, 64, 32], kernel_size=3):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.final_conv = Conv2D(2, (1, 1), activation='tanh')

    def call(self, x, skips, z):
        dim = K.int_shape(x)
        flatten_dim = dim[1] * dim[2] * dim[3]
        z_proj = Dense(flatten_dim, activation='relu')(z)
        z_proj = tf.reshape(z_proj, tf.shape(x))
        x = tf.concat([x, z_proj], axis=-1)

        for f, skip in zip(reversed(self.filters), reversed(skips)):
            x = upsample_block(x, skip, f, self.kernel_size)

        return self.final_conv(x)