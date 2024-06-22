from datetime import datetime
from src.multi import MultiLayerPerceptron
from src.autoencoder import Autoencoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

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


def relu(value):
    return np.maximum(0, value)


def relu_derivative(value):
    return np.where(value > 0, 1, 0)


if __name__ == "__main__":
    render_char(Font3[2])

    b = get_bit_array(Font3)
    print(np.shape(b))

    learning_rates = [0.001]
    hidden_layers = [
        [28, 22, 17, 10, 5]
    ]

    # Iterate through all combinations of learning rates and hidden layers
    for learning_rate, hidden_layer in itertools.product(learning_rates, hidden_layers):
        print(hidden_layer)
        # Create and train the autoencoder
        ae = Autoencoder(learning_rate, len(
            b[0]), hidden_layer, (0, 1), (0, 1))
        error = ae.train(b, b, 10000)

        # Save the model with a timestamp and the error rate in the filename
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        ae.save_model(f"./models/ae/autoencoder_lr{learning_rate}_hidden{hidden_layer}_{datetime_string}_error{error}.pkl")

    exit()
    learning_rate = 0.001
    hidden_layer = [30, 10]
    ae = Autoencoder(learning_rate, len(b[0]), hidden_layer, (0, 1), (0, 1))

    # mlp = MultiLayerPerceptron(
    #     0.001, 35, [10, 10, 2, 10, 10], 35, (0, 1), (0, 1))

    error = ae.train(b, b, 10000)
    print(error)

    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime(
        "%Y-%m-%d_%H-%M-%S")

    ae.save_model("./models/autoencoder_" +
                  datetime_string + "_" + str(error) + ".pkl")


    exit()
    mlp = MultiLayerPerceptron.load_model("models/model.pkl")
    results = []
    for i in range(len(b)):
        result = mlp.predict(b[i])
        results.append(result)

    fig, axes = plt.subplots(4, 8, figsize=(20, 10))

    # Plot each result on the grid
    for i, result in enumerate(results):
        ax = axes[i // 8, i % 8]
        ax.set_title(f"Result {i+1}")
        plot = render_result(result)
        ax.imshow(plot)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
