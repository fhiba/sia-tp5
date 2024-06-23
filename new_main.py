from datetime import datetime
from src.multi import MultiLayerPerceptron
from src.autoencoder import Autoencoder
from data.font import Font3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


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
        [32, 24, 16, 8]
    ]

    # Iterate through all combinations of learning rates and hidden layers
    for learning_rate, hidden_layer in itertools.product(learning_rates, hidden_layers):
        print(hidden_layer)
        # Create and train the autoencoder
        ae = Autoencoder(learning_rate, len(
            b[0]), hidden_layer, (0, 1), (0, 1))
        error = ae.train(b, b, 25000)

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
