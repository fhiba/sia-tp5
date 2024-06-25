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
import seaborn as sns
from data.font import Font3, Font3Tags

plt.rcParams['figure.dpi'] = 250

def render_font(font):
    for index, char in enumerate(font):
        print(f"Character 0x{60+index:02x}:")
        render_char(char)
        print()


def render_char(char):
    char_representation = []
    for byte in char:
        bits = bin(byte)[2:].zfill(5)  # Convert byte to 5-bit binary string
        char_representation.append(bits.replace('0', ' ').replace('1', '#'))
    for row in char_representation:
        print(row)


def get_bit_array(font):
    font_bit_arrays = []
    for char in font:
        char_bit_array = []
        for byte in char:
            # Convert byte to 5-bit binary list
            bits = [int(bit) for bit in bin(byte)[2:].zfill(5)]
            char_bit_array.extend(bits)
        font_bit_arrays.append(char_bit_array)
    return font_bit_arrays


def render_result(result):
    if result.size != 35:
        raise ValueError("The input matrix must be 5x5.")

    # Reshape the flat matrix into a 5x5 array
    matrix = result.reshape(7, 5)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.heatmap(matrix, annot=False, cbar=False, square=True, linewidth=2,
                linecolor='black', cmap='coolwarm', xticklabels=False, yticklabels=False, ax=ax)
    ax.axis('off')

    # Convert the plot to numerical data
    fig.canvas.draw()
    image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # Close the figure to avoid displaying it

    return image_data


def render_results(results):
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))

    for i, result in enumerate(results):
        ax = axes[i // 8, i % 8]
        ax.set_title(f"Result {i+1}")
        plot = render_result(result)
        ax.imshow(plot)
        ax.axis('off')

    plt.tight_layout()

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

# images = [images[0], images[9]]
images = images[0:5]

# Convert list of flattened images to a single numpy array
images_copy = copy.deepcopy(images)
images_array = np.array(copy.deepcopy(images))

# images = get_bit_array(Font3)
# images_array = np.array(images)


# display_image(images[2])
# print(len(images[2]))
# exit()

# Split the dataset into training and validation sets
# train_images, val_images = train_test_split(
#     images, test_size=0.2, random_state=42)

# Define hyperparameters
learning_rate = 0.1
input_size = images_array.shape[1]  # Assuming images have the same dimensions
hidden_node_sizes = [10]  # Example architecture, adjust as needed
latent_dim = 2  # Example latent dimension, adjust as needed
# Assuming pixel values range from 0 to 255
input_range = (np.min(images), np.max(images))
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
max_epochs = 100 # Example number of epochs, adjust as needed
vae_model.train(images, images, max_epochs=max_epochs)

# Save the trained model
# model_file = "vae_model_emoji.pkl"  # Choose a filename for the saved model
# vae_model.save_model(model_file)

# _, res = vae_model.decoder.forward_propagation([1, 0.000001])
# img = feature_scaling(
#             res[-1], (0, 1), expected_range)
# display_image(img)
# res = vae_model.decoder.predict([0.3])
# display_image(res)
# res = vae_model.decoder.predict([0.4])
# display_image(res)
# res = vae_model.decoder.predict([0.5])
# display_image(res)
# res = vae_model.decoder.predict([0.8])
# display_image(res)
# exit()

latent_images = []
results = []
for i in range(len(images)):
    # display_image(images_copy[i])
    latent = vae_model.predict_latent(images[i])
    latent_images.append(latent)
    result = vae_model.predict(latent)
    results.append(result)
    display_image(results[-1])

# render_results(results)
# render_results(np.array(images))
# plt.show()

xy_items = latent_images

# Separate the list into x and y components
x = [item[1] for item in xy_items]
y = [item[2] for item in xy_items]

# Create the plot
plt.scatter(x, y)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of (x, y) items')

plt.show()
