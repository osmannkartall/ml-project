import numpy as np
from utils import (data, sample_real, sample_fake, sample_latent_point,
                   show_accuracy)
from model import Discriminator, Generator, GAN

def train(X, generator, discriminator, GAN, size_latent, batch=256, epoch=25):
  count_batch = int(X.shape[0] / batch)
  for i in range(epoch):
    print("EPOCH: %d" % (i+1))
    for j in range(count_batch):
      real_images, real_image_labels = sample_real(X, int(batch / 2))
      fake_images, fake_image_labels = sample_fake(generator, int(batch / 2),
                                                   size_latent)
      latent_points = sample_latent_point(batch, size_latent)
      latent_points_labels = np.ones((batch, 1))

      discriminator.train_on_batch(
          np.vstack((real_images, fake_images)),
          np.vstack((real_image_labels, fake_image_labels)))
      
      GAN.train_on_batch(latent_points, latent_points_labels)
    show_accuracy(X, generator, discriminator, size_latent)
  generator.save("gan.h5")

def run():
  size_latent = 128
  X, input_shape = data()
  discriminator = Discriminator(input_shape).model
  generator = Generator(size_latent, input_shape).model
  gan = GAN(generator, discriminator).model
  train(X, generator, discriminator, gan, size_latent)

run()