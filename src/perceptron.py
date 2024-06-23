import sys
from typing import Optional
import numpy as np


def online_training_method(length: int) -> list[int]:
    return np.random.randint(0, length, 1)


def batch_training_method(length: int) -> list[int]:
    return range(length)


class Perceptron:
    def __init__(
        self,
        data: np.array,
        expected_value: np.array,
        learning_rate: float = 0.000001,
        training_method=online_training_method
    ):
        self.learning_rate = learning_rate
        self.min_error = sys.maxsize
        self.data = np.insert(data, 0, 1, axis=1)
        self.weights = np.random.uniform(
            low=-1, high=1, size=self.data.shape[1])
        self.min_weights = self.weights
        self.expected_value = expected_value
        self.data_len = len(self.data)
        self.training_method = training_method

    def projection(self, x: list[int]) -> float:
        return x[0] * self.weights[0] + \
            x[1] * self.weights[1] + self.weights[2]

    def step_activation(self, val: float) -> int:
        if (val >= 0):
            return 1
        else:
            return -1

    def predict(self, x: list[int]) -> int:
        return self.step_activation(self.projection(x))

    def train_indexes(self, indexes, epochs: Optional[int] = 1000):
        for epoch in range(epochs):
            self.update_weights(indexes)

            error = self.compute_error()
            if error < self.min_error:
                self.max_error = error
                self.min_weights = self.weights

            # Me fijo si el error esta dentro de los margenes que busco
            if self.is_converged():
                break
        self.weights = self.min_weights
        return epoch+1, self.is_converged()

    def train(self, epochs: Optional[int] = 1000):
        indexes = self.get_indexes()
        return self.train_indexes(indexes, epochs)

    def update_weights(self, indexes):
        deltas = self.compute_deltas(indexes)
        self.weights = self.weights + np.sum(deltas, axis=0)

    def __str__(self) -> str:
        output = "Expected - Actual\n"

        for expected, actual in zip(self.expected_value, self.get_outputs()):
            output += f"{expected:<10} {actual}\n"

        output += f"\nWeights: {self.weights}"
        return output

    def get_outputs(self):
        excitations = np.dot(self.data, self.weights)
        return np.vectorize(self.activation_func)(excitations)

    def get_range_outputs(self, indexes: list[int]):
        excitations = np.dot(self.data[indexes], self.weights)
        return np.vectorize(self.activation_func)(excitations)

    def get_indexes(self):
        return self.training_method(self.data_len)

    def compute_error(self):
        raise NotImplementedError

    def is_converged(self):
        raise NotImplementedError

    def compute_deltas(self, indexes):
        raise NotImplementedError

    def activation_func(self, value):
        raise NotImplementedError
