import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from keras.datasets.mnist import load_data

def data():
  (X_train, _), (_, _) = load_data()
  input_shape = (X_train.shape[1], X_train.shape[2], 1)
  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],
                             X_train.shape[2], 1))
  X_train = X_train.astype("float32")
  X_train /= 255.0
  return X_train, input_shape

def sample_real(X, num_sample):
  y = np.ones((num_sample, 1))
  return X[np.random.randint(0, X.shape[0], num_sample)], y

def sample_latent_point(num_sample, size_latent):
  latent_points = np.random.randn(size_latent * num_sample)
  latent_points = latent_points.reshape(num_sample, size_latent)
  return latent_points

def sample_fake(generator, num_sample, size_latent):
  latent_points = sample_latent_point(num_sample, size_latent)
  X = generator.predict(latent_points)
  y = np.zeros((num_sample, 1))
  return X, y

def plot(images):
  num = images.shape[0]
  plt.figure(figsize=(20, 10))
  columns = 10
  for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.imshow(images[i].reshape((28,28)), cmap="gray", aspect="auto")
  plt.show()

def show_image(images):
  columns = 10
  fig = plt.figure(figsize=(columns, columns))
  gs = gridspec.GridSpec(columns, columns)
  gs.update(wspace=0, hspace=0)
  for i, image in enumerate(images):
    ax = plt.subplot(gs[i])
    plt.axis("off")
    ax.set_aspect("equal")
    plt.imshow(image.reshape(28, 28), cmap="gray")
  plt.show()
  plt.close(fig)

def show_accuracy(X, generator, discriminator, size_latent, num_sample=100):
  print("Discriminator Accuracy:")
  X, y = sample_real(X, num_sample)
  print("\ton real images: ", discriminator.evaluate(X, y)[1])
  X, y = sample_fake(generator, num_sample, size_latent)
  print("\ton fake images: ", discriminator.evaluate(X, y)[1])
  show_image(X)