import numpy as np
from src.autoencoder import Autoencoder
import copy
import seaborn as sns
import matplotlib.pyplot as plt
Font3 = [
    [0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00],   # 0x60, `
    [0x00, 0x0e, 0x01, 0x0d, 0x13, 0x13, 0x0d],   # 0x61, a
    [0x10, 0x10, 0x10, 0x1c, 0x12, 0x12, 0x1c],   # 0x62, b
    [0x00, 0x00, 0x00, 0x0e, 0x10, 0x10, 0x0e],   # 0x63, c
    [0x01, 0x01, 0x01, 0x07, 0x09, 0x09, 0x07],   # 0x64, d
    [0x00, 0x00, 0x0e, 0x11, 0x1f, 0x10, 0x0f],   # 0x65, e
    [0x06, 0x09, 0x08, 0x1c, 0x08, 0x08, 0x08],   # 0x66, f
    [0x0e, 0x11, 0x13, 0x0d, 0x01, 0x01, 0x0e],   # 0x67, g
    [0x10, 0x10, 0x10, 0x16, 0x19, 0x11, 0x11],   # 0x68, h
    [0x00, 0x04, 0x00, 0x0c, 0x04, 0x04, 0x0e],   # 0x69, i
    [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0c],   # 0x6a, j
    [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],   # 0x6b, k
    [0x0c, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],   # 0x6c, l
    [0x00, 0x00, 0x0a, 0x15, 0x15, 0x11, 0x11],   # 0x6d, m
    [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],   # 0x6e, n
    [0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e],   # 0x6f, o
    [0x00, 0x1c, 0x12, 0x12, 0x1c, 0x10, 0x10],   # 0x70, p
    [0x00, 0x07, 0x09, 0x09, 0x07, 0x01, 0x01],   # 0x71, q
    [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],   # 0x72, r
    [0x00, 0x00, 0x0f, 0x10, 0x0e, 0x01, 0x1e],   # 0x73, s
    [0x08, 0x08, 0x1c, 0x08, 0x08, 0x09, 0x06],   # 0x74, t
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0d],   # 0x75, u
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x0a, 0x04],   # 0x76, v
    [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0a],   # 0x77, w
    [0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11],   # 0x78, x
    [0x00, 0x11, 0x11, 0x0f, 0x01, 0x11, 0x0e],   # 0x79, y
    [0x00, 0x00, 0x1f, 0x02, 0x04, 0x08, 0x1f],   # 0x7a, z
    [0x06, 0x08, 0x08, 0x10, 0x08, 0x08, 0x06],   # 0x7b, {
    [0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x04],   # 0x7c, |
    [0x0c, 0x02, 0x02, 0x01, 0x02, 0x02, 0x0c],   # 0x7d, }
    [0x08, 0x15, 0x02, 0x00, 0x00, 0x00, 0x00],   # 0x7e, ~
    [0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f]    # 0x7f, DEL
]

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

    



