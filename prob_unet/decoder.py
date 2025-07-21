import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, Dense

class decoder(tf.keras.layers.Layer):
    def __init__(self, filters = [256, 128, 64, 32], kernel_size = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.upsample_blocks = []
        for f in filters:
            self.upsample_blocks.append(self._upsample_block(f))
        self.final_conv = Conv2D(2,(1, 1), activation='tanh')

    def _upsample_block(self, filters):
        return tf.keras.Sequential([
            Conv2DTranspose(filters, self.kernel_size, strides = 2, padding='same'),
            Concatenate(),
            Conv2D(filters, self.kernel_size, padding='same', kernel_initializer="he_normal", use_bias=False),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, self.kernel_size, padding='same', kernel_initializer="he_normal", use_bias=False),
            BatchNormalization(),
            ReLU(),
        ])

    def call(self, x, skips, z):
        z_proj = Dense(tf.reduce_prod(x.shape[1:]), activation='relu')(z)
        z_proj = tf.reshape(z_proj, tf.shape(x))
        x = tf.concat([x, z_proj], axis = -1)
        for up, skip in zip(self.upsample_blocks, reversed(skips)):
            x = up(x)
            x = tf.concat([x, skip], axis = -1)
        return self.final_conv(x)