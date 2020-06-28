from utils import sample_latent_point
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('gan.h5')

def generate_image():
	fake_image = model.predict(sample_latent_point(1, 256))
	plt.imshow(fake_image[0, :, :, 0], cmap="gray")
	plt.show()

generate_image()