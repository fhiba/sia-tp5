from src.multi import MultiLayerPerceptron
import numpy as np
import pickle


def batch(X):
    return X


def exp_activation_func(value):
    return 1 / (1 + np.exp(-value))


def exp_activation_func_derivative(value):
    activation_function = exp_activation_func(value)
    return activation_function * (1 - activation_function)


def feature_scaling(value: float, from_int: tuple[float, float], to_int: tuple[float, float]) -> float:
    numerator = value - from_int[0]
    denominator = from_int[1] - from_int[0]
    return (numerator / denominator) * (to_int[1] - to_int[0]) + to_int[0]


class Autoencoder():
    def __init__(
        self,
        learning_rate: float,
        input_size: int,
        hidden_node_sizes: [int],
        input_range,
        expected_range,
        activation_func=exp_activation_func,
        activation_func_derivative=exp_activation_func_derivative,
        training_strategy=batch,
        activation_func_range=(0, 1),
        percentage_threshold=0.000001
    ):
        reverse = hidden_node_sizes[::-1]
        self.mlp = MultiLayerPerceptron(learning_rate,
                                        input_size,
                                        hidden_node_sizes +
                                        [2] + reverse,
                                        input_size,
                                        input_range,
                                        expected_range,
                                        activation_func,
                                        activation_func_derivative,
                                        training_strategy,
                                        activation_func_range,
                                        percentage_threshold
                                        )
        self.latent_image_dimension = 1 + len(hidden_node_sizes)

    def train(self, dataset, expected, max_epochs: int = 1000):
        return self.mlp.train(dataset, expected, max_epochs)

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    def encode(self, input):
        for i in range(len(input)):
            input[i] = feature_scaling(
                input[i], self.mlp.input_range, self.mlp.activation_func_range)
        x_input = np.array([1] + input)  # Add bias to input
        activations = [x_input.T]

        # Forward propagate input through each layer
        for i in range(self.latent_image_dimension):
            h = activations[-1].dot(self.mlp.weights[i])  # Compute weighted sum

            activation = self.mlp.activation_func(h)  # Output layer
            activations.append(activation)

        # Return output of the last layer (predictions)
        return feature_scaling(
            activations[-1], self.mlp.activation_func_range, self.mlp.expected_range)

    def decode(self, input):
        for i in range(len(input)):
            input[i] = feature_scaling(
                input[i], self.mlp.input_range, self.mlp.activation_func_range)
        x_input = np.array(input)  # Add bias to input
        activations = [x_input.T]

        # Forward propagate input through each layer
        for i in range(self.latent_image_dimension, len(self.mlp.weights)):
            h = activations[-1].dot(self.mlp.weights[i])  # Compute weighted sum

            activation = self.mlp.activation_func(h)  # Output layer
            activations.append(activation)

        # Return output of the last layer (predictions)
        return feature_scaling(
            activations[-1], self.mlp.activation_func_range, self.mlp.expected_range)


    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
