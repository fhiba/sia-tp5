import numpy as np
from src.autoencoder import Autoencoder
from data.font import Font3
import copy
import seaborn as sns
import matplotlib.pyplot as plt

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



# add noise to a flatten matrix
def add_toggled_noise(letters, noise):
    all_pixels = range(len(letters[0]))
    pixel_amount = len(all_pixels)
    for i in range(len(letters)):
        noised_pixels = np.random.choice(all_pixels, int(np.round(noise * pixel_amount, 0)))
        for pixel in noised_pixels:
            letters[i][pixel] = 1 - letters[i][pixel]
        
    return letters

# add noise to a flatten matrix
def add_zeroed_noise(letters, noise):
    all_pixels = range(len(letters[0]))
    pixel_amount = len(all_pixels)
    for i in range(len(letters)):
        noised_pixels = np.random.choice(all_pixels, int(np.round(noise * pixel_amount, 0)))
        for pixel in noised_pixels:
            letters[i][pixel] = 0

    return letters
 

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



if __name__ == "__main__":

    b = get_bit_array(Font3)
    print(np.shape(b))

    learning_rates = 0.005
    hidden_layers = [30, 25, 20, 15]
    # results = []
    # fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    noised_letters = add_zeroed_noise(copy.deepcopy(b),0.1)
    # for i, result in enumerate(np.array(noised_letters)):
    #     ax = axes[i // 8, i % 8]
    #     ax.set_title(f"Result {i+1}")
    #     plot = render_result(result)
    #     ax.imshow(plot)
    #     ax.axis('off')
    #
    # plt.tight_layout()
    # plt.show()
    ae = Autoencoder(learning_rates, len(np.array(noised_letters)[0]), hidden_layers, (0, 1), (0, 1))
    error = ae.train(noised_letters,b,25000)
    results = []
    errors = []
    latent_iamges = []
    for i in range(len(b)):
        result, error = ae.mlp.predict_with_error(b[i], b[i])
        results.append(result)
        errors.append(error)
        latent_iamges.append(ae.encode(b[i]))

    print(errors)
    render_results(results)
    render_results(np.array(b))
    plt.show()

    



