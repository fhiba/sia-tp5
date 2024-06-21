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
        self.mlp = MultiLayerPerceptron(learning_rate,
                                        input_size,
                                        hidden_node_sizes +
                                        [2] + hidden_node_sizes,
                                        input_size,
                                        input_range,
                                        expected_range,
                                        activation_func,
                                        activation_func_derivative,
                                        training_strategy,
                                        activation_func_range,
                                        percentage_threshold
                                        )

    def train(self, dataset, expected, max_epochs: int = 1000):
        return self.mlp.train(dataset, expected, max_epochs)


    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
