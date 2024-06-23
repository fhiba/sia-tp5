from datetime import datetime
from src.multi import MultiLayerPerceptron
from src.autoencoder import Autoencoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import random

plt.rcParams['figure.dpi'] = 250

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


def render_results(results):
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))

    for i, result in enumerate(results):
        ax = axes[i // 8, i % 8]
        ax.set_title(f"Result {i+1}")
        plot = render_result(result)
        ax.imshow(plot)
        ax.axis('off')

    plt.tight_layout()


def relu(value):
    return np.maximum(0, value)


def relu_derivative(value):
    return np.where(value > 0, 1, 0)


if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <string_argument>")
        sys.exit(1)

    # Get the string argument from the command line
    string_argument = sys.argv[1]

    b = get_bit_array(Font3)
    font_bit_array = b

    ae = Autoencoder.load_model(string_argument)

    # Retrain loaded model
    # error = ae.train(b, b, 100000)
    # current_datetime = datetime.now()
    # datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # ae.save_model(f"./models/special/autoencoder_lr{ae.mlp.original_learning_rate}_hidden{ae.mlp.dimensions}_{datetime_string}_error{error}.pkl")

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
    print(latent_iamges)

    # Example list of (x, y) items
    xy_items = latent_iamges

    # Corresponding tags
    tags = [
        '\'', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'# , 'DEL'
    ]

    # Separate the list into x and y components
    x = [item[0] for item in xy_items]
    y = [item[1] for item in xy_items]

    # Create the plot
    plt.scatter(x, y)

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of (x, y) items')

    # Add tags to each point
    for i, tag in enumerate(tags):
        plt.text(x[i] - 0.01, y[i], tag, fontsize=12, ha='right')

    # Select two random points from the list
    point1, point2 = random.sample(xy_items, 2)

    # Calculate the slope (m) and the y-intercept (b)
    x1, y1 = point1
    x2, y2 = point2

    # Ensure the points are not identical (which would cause division by zero)
    while x1 == x2:
        point2 = random.sample(xy_items, 1)[0]
        x2, y2 = point2

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    print(f"Selected points: {point1}, {point2}")
    print(f"Linear function: y = {m}x + {b}")

    # Create the plot
    # plt.scatter(*zip(*xy_items))

    # Plot the linear function
    x_values = [min(x1, x2), max(x1, x2)]
    y_values = [m * x + b for x in x_values]
    plt.plot(x_values, y_values, 'r', label=f'y = {m:.2f}x + {b:.2f}')
    # Display the plot
    # plt.show()

    x_values = np.linspace(min(x1, x2), max(x1, x2), 32)
    y_values = m * x_values + b

    results = []
    # a_latent_image = ae.encode(font_bit_array[1])
    # b_latent_image = ae.encode(font_bit_array[2])
    # result_a = ae.decode(a_latent_image)
    # result_b = ae.decode(b_latent_image)
    # results.append(result_a)
    # results.append(result_b)
    # render_results(results)
    # plt.show()
    # exit()

    for pair in zip(x_values, y_values):
        latent_iamge = np.array(pair)
        result = ae.decode(latent_iamge)
        results.append(result)
        latent_iamges.append(latent_iamge)
    render_results(results)
    plt.show()

    results = []
    for _ in range(32):
        image = [np.random.random(), np.random.random()]
        results.append(ae.decode(image))
    print(results)
    render_results(results)
    plt.show()
