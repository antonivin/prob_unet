import tensorflow as tf
from .encoder import encoder
from .decoder import decoder
from .prior import PriorNet
from .posterior import PosteriorNet
from .utils import reparameterize

class ProbabilisticUNet(tf.keras.Model):
    def __init__(self, latent_dim = 6):
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.prior_net = PriorNet(latent_dim)
        self.posterior_net = PosteriorNet(latent_dim)
        self.latent_dim = latent_dim

    def call(self, x, y= None, training = True):
        enc_out, skips = self.encoder(x)

        if training:
            mu_post, logvar_post = self.posterior_net(x, y)
            z = reparameterize(mu_post, logvar_post)
            mu_prior, logvar_prior = self.prior_net(x)
            y_pred = self.decoder(enc_out, skips, z)
            return y_pred, mu_post, logvar_post, mu_prior, logvar_prior
        else:
            mu_prior, logvar_prior = self.prior_net(x)
            z = reparameterize(mu_prior, logvar_prior)
            y_pred = self.decoder(enc_out, skips, z)
            return y_pred

