from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import (Dense, Dropout, Flatten, Conv2D, Conv2DTranspose,
                          LeakyReLU, Reshape)

class Discriminator:
  def __init__(self, input_shape):
    self.input_shape = input_shape
    self.create_model()

  def create_model(self): 
    self.model = Sequential()
    self.model.add(Conv2D(filters=64, kernel_size=3, strides=2,
                          padding="same", input_shape=self.input_shape))
    self.model.add(LeakyReLU(alpha=0.2))
    self.model.add(Conv2D(filters=64, kernel_size=3, strides=2, 
                          padding="same"))
    self.model.add(LeakyReLU(alpha=0.2))
    self.model.add(Flatten())
    self.model.add(Dense(1, activation="sigmoid"))
    self.model.compile(loss="binary_crossentropy", 
                       optimizer=Adam(learning_rate=0.0008, beta_1=0.7), 
                       metrics=["accuracy"])

class Generator:
  def __init__(self, size_latent, input_shape):
    self.size_latent = size_latent
    self.input_shape = input_shape
    self.create_model()

  def create_model(self): 
    h = self.input_shape[0] // 4
    w = self.input_shape[1] // 4
    num_low_resolution_kernel = 100
    self.model = Sequential()
    self.model.add(Dense(num_low_resolution_kernel * h * w,
                         input_dim=self.size_latent))
    self.model.add(LeakyReLU(alpha=0.2))
    self.model.add(Reshape((h, w, num_low_resolution_kernel)))
    self.model.add(Conv2DTranspose(num_low_resolution_kernel, kernel_size=4,
                                   strides=2, padding='same'))
    self.model.add(LeakyReLU(alpha=0.2))
    self.model.add(Conv2DTranspose(num_low_resolution_kernel, kernel_size=4,
                                   strides=2, padding='same'))
    self.model.add(LeakyReLU(alpha=0.2))
    self.model.add(Conv2D(1, (h,w), activation="tanh", padding='same'))

class GAN:
  def __init__(self, generator, discriminator):
    discriminator.trainable = False
    self.generator = generator
    self.discriminator = discriminator
    self.create_model()
  
  def create_model(self): 
    self.model = Sequential()
    self.model.add(self.generator)
    self.model.add(self.discriminator)
    self.model.compile(loss='binary_crossentropy',
                       optimizer=Adam(learning_rate=0.0008, beta_1=0.7))