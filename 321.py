import os
import glob
from PIL import Image
import numpy as np

data_path = 'N:/monet_jpg'
image_paths = glob.glob(os.path.join(data_path, '*.jpg'))

def load_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path)
        img = img.resize((64, 64))  # Resize the image
        img_array = np.array(img) / 255.0  # Normalize the pixel values
        images.append(img_array)
    return np.array(images)

images = load_images(image_paths)



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Dropout, Flatten
from tensorflow.keras.models import Sequential

# Define the generator model
def create_generator(latent_dim):
    model = Sequential()
    model.add(Dense(8*8*256, use_bias=False, input_shape=(latent_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((8, 8, 256)))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# Define the discriminator model
def create_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=(64, 64, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))
    return model



from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

latent_dim = 128
generator = create_generator(latent_dim)
discriminator = create_discriminator()

# Compile the discriminator model
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss=BinaryCrossentropy(from_logits=True))

# Create the combined GAN model
def create_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss=BinaryCrossentropy(from_logits=True))
    return model

gan = create_gan(generator, discriminator)

# Define the GAN training function
def train_gan(images, batch_size, epochs, latent_dim, generator, discriminator, gan):
    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train the discriminator
        real_images = images[np.random.randint(0, len(images), size=batch_size)]
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        fake_images = generator.predict(noise)

        discriminator.train_on_batch(real_images, real_labels)
        discriminator.train_on_batch(fake_images, fake_labels)

        # Train the generator
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        gan.train_on_batch(noise, real_labels)

        # Print the progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1} completed")

# Train the GAN model
batch_size = 64
epochs = 500
train_gan(images, batch_size, epochs, latent_dim, generator, discriminator, gan)



import matplotlib.pyplot as plt

def generate_images(generator, latent_dim, num_images):
    noise = np.random.normal(0, 1, size=(num_images, latent_dim))
    generated_images = generator.predict(noise)

    # Scale the images back to [0, 1] range
    generated_images = (generated_images + 1) / 2

    return generated_images

def display_images(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.ravel()

    for i, image in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis('off')

    plt.show()

num_images = 100
generated_images = generate_images(generator, latent_dim, num_images)
display_images(generated_images, 10, 10)
