import tensorflow as tf
from tensorflow.python.keras.utils.version_utils import training
from prob_unet.utils import get_data

from prob_unet.model import ProbabilisticUNet

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        # tf.config.experimental.set_virtual_device_configuration(gpus[2], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0)])
        except RuntimeError as e:
            print(e)

    im_width = 128
    im_height = 128
    path_train = '/home/maria/pycharm/train/'

    train_data, train_labels = get_data(path_train, im_height, im_width, train=True)

    model = ProbabilisticUNet(latent_dim=6)
    optimizer = tf.keras.optimizers.Adam()
    dummyX = tf.random.normal((1, 128, 128, 1))
    dummyY = tf.random.normal((1, 128, 128, 1))
    _ = model(dummyX, training=True)
    model.summary()


