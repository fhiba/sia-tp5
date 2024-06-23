from src.multi_layer_perceptron import MultiLayerPerceptron
from src.encoder import Encoder
from src.decoder import Decoder
from data.font import Font3
import numpy as np


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


if __name__ == "__main__":

    render_char(Font3[2])

    b = get_bit_array(Font3)
    print(np.shape(b))

    array_32_items = np.random.choice([0, 1], size=64)

    array_2d = array_32_items.reshape(32, 2)
    encoder = Encoder(0.0001, b, 10, 1, 2, array_2d)
    decoder = Decoder(0.0001, array_2d, 2, 1, len(b[0]), b)

    for i in range(1, 100):
        latent_image = encoder.predict(b)
        decoder.X = np.insert(latent_image, 0, 1, axis=1)
        decoder.train_adam(1)
        encoder.last_delta = decoder.last_delta
        encoder.train_adam(1)
    latent_image = encoder.predict(b)
    result = decoder.predict(latent_image)
    print(b)
    print(result)

    exit()


