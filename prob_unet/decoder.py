import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, Dense, Activation
from tensorflow.keras import backend as K


class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, activation='relu'):
        super().__init__()
        self.conv_transpose = Conv2DTranspose(filters, kernel_size, strides=2, padding='same')
        self.concat = Concatenate(axis=-1)
        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.act1 = Activation(activation)
        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.act2 = Activation(activation)

    def call(self, x, skip):
        x = self.conv_transpose(x)           # Upsample
        x = self.concat([x, skip])           # Concatenate skip connection
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x

class decoder(tf.keras.layers.Layer):
    def __init__(self, filters=[256, 128, 64, 32], kernel_size=3):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.final_conv = Conv2D(2, (1,1), activation='tanh')
        self.dense_proj = None
        self.upsample_blocks = [UpsampleBlock(f, kernel_size) for f in reversed(filters)]

    def build(self, input_shape):
        flatten_dim = input_shape[1] * input_shape[2] * input_shape[3]
        self.dense_proj = tf.keras.layers.Dense(flatten_dim, activation='relu')
        super().build(input_shape)

    def call(self, x, skips, z):
        z_proj = self.dense_proj(z)
        z_proj = tf.reshape(z_proj, tf.shape(x))
        x = tf.concat([x, z_proj], axis=-1)

        for block, skip in zip(self.upsample_blocks, reversed(skips)):
            x = block(x, skip)

        return self.final_conv(x)