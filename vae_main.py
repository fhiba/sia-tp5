import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
# Assuming you have a VAE model class defined in vae_model.py
from src.variational_autoencoder import VariationalAutoencoder, feature_scaling
import base64
import math
import matplotlib.pyplot as plt
import copy


# Function to reshape and display an image
def display_image(vector_image, image_size=(72, 72)):
    # Reshape the vector to 72x72x3 (assuming the image has 3 color channels)
    reshaped_image = vector_image.reshape(image_size[0], image_size[1], 4)

    # Display the image
    plt.imshow(reshaped_image.astype(np.uint8))
    plt.axis('off')
    plt.show()


# Load your emoji dataset
emoji_data = pd.read_csv('data/emoji/full_emoji.csv')

# Define the standard size for all images
standard_size = (72, 72)

# Convert image data from CSV to numpy arrays
images = []
for img_data in emoji_data['Apple']:
    if len(images) > 20:
        break
    if pd.notna(img_data):
        try:
            # Extract the base64 part after the comma
            img = img_data.split(',')[1]
            # Decode the base64 string and open the image
            image = Image.open(BytesIO(base64.b64decode(img)))
            # Convert image to RGB (3 channels)
            image = image.convert('RGBA')
            # Resize image to the standard size
            image = image.resize(standard_size)
            # Convert image to numpy array
            image = np.array(image)
            # Flatten the image and append to the list
            images.append(image.flatten())
        except Exception as e:
            print(f"Error processing image: {e}")

images = images[1:8]

# Convert list of flattened images to a single numpy array
images_copy = copy.deepcopy(images)
images_array = np.array(copy.deepcopy(images))
# display_image(images[2])
# print(len(images[2]))
# exit()

# Split the dataset into training and validation sets
# train_images, val_images = train_test_split(
#     images, test_size=0.2, random_state=42)

# Define hyperparameters
learning_rate = 0.1
input_size = images_array.shape[1]  # Assuming images have the same dimensions
hidden_node_sizes = [30, 10]  # Example architecture, adjust as needed
latent_dim = 10  # Example latent dimension, adjust as needed
input_range = (np.min(images), np.max(images))  # Assuming pixel values range from 0 to 255
expected_range = (np.min(images), np.max(images))
# Create and initialize VAE model
vae_model = VariationalAutoencoder(
    learning_rate=learning_rate,
    input_size=input_size,
    hidden_node_sizes=hidden_node_sizes,
    latent_dim=latent_dim,
    input_range=input_range,
    expected_range=expected_range
)

# Train the VAE model
max_epochs = 100  # Example number of epochs, adjust as needed
vae_model.train(images, images, max_epochs=max_epochs)

# Save the trained model
# model_file = "vae_model_emoji.pkl"  # Choose a filename for the saved model
# vae_model.save_model(model_file)

# _, res = vae_model.decoder.forward_propagation([1, 0.000001])
# img = feature_scaling(
#             res[-1], (0, 1), expected_range)
# display_image(img)
# exit()
# res = vae_model.decoder.predict([0.3])
# display_image(res)
# res = vae_model.decoder.predict([0.4])
# display_image(res)
# res = vae_model.decoder.predict([0.5])
# display_image(res)
# res = vae_model.decoder.predict([0.8])
# display_image(res)
# exit()

for i in range(len(images)):
    # display_image(images_copy[i])
    result = vae_model.predict(images_copy[i])
    display_image(result)
