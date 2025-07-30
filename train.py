import tensorflow as tf
from prob_unet.utils import get_data, prob_unet_loss
from sklearn.model_selection import train_test_split
from prob_unet.model import ProbabilisticUNet


@tf.function
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        y_pred, mu_post, logvar_post, mu_prior, logvar_prior = model(x, y, training=True)
        loss, mse, kl = prob_unet_loss(y, y_pred, mu_post, logvar_post, mu_prior, logvar_prior)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


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
    batch_size = 128
    epochs = 500
    path_train = '/ba-120/ablage/Anto/probabilistic_UNET/train/'

    data, labels = get_data(path_train, im_height, im_width, train=True)

    train_data, val_data, train_sol , val_sol = train_test_split(data, labels, test_size=0.2, random_state=1)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_sol))
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_sol))
    val_dataset = val_dataset.batch(batch_size)

    model = ProbabilisticUNet(latent_dim=6)
    optimizer = tf.keras.optimizers.Adam()


    for epoch in range (epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss, mse, kl = train_step(model, x_batch, y_batch, optimizer)
            if step % 50 == 0:
                print(f"Step {step}: Loss = {loss:.4f}, MSE = {mse:.4f}, KL = {kl:.4f}")


