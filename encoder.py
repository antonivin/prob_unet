import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

class encoder(tf.keras.layers.Layer):
    def __init__(self, filters = [32, 64, 128, 256, 512], kernel_size = 7):
        super().__init__()
        self.blocks = []
        self.kernel_size = kernel_size
        self.pools = []
        for f in filters[:-1]:
            self.blocks.append(self._conv_block(f))
            self.pools.append(MaxPooling2D())
        self.bottleneck = self._conv_block(filters[-1])

    def _conv_block(self, filters):
        return tf.keras.Sequential([
            Conv2D(filters, self.kernel_size, padding='same', kernel_initializer="he_normal", use_bias=False),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, self.kernel_size, padding='same', kernel_initializer="he_normal", use_bias=False),
            BatchNormalization(),
            ReLU(),
            ])

    def call(self, x):
            skips = []
            for conv, pool in zip(self.blocks, self.pools):
                x = conv(x)
                skips.append(x)
                x = pool(x)
            x = self.bottleneck(x)
            return x, skips