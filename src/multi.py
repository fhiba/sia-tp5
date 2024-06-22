import numpy as np
import pickle


def batch(X):
    return X


def exp_activation_func(value):
    return 1 / (1 + np.exp(-value))


def exp_activation_func_derivative(value):
    # activation_function = exp_activation_func(value)
    # return activation_function * (1 - activation_function)
    return value * (1 - value)


def feature_scaling(value: float, from_int: tuple[float, float], to_int: tuple[float, float]) -> float:
    numerator = value - from_int[0]
    denominator = from_int[1] - from_int[0]
    return (numerator / denominator) * (to_int[1] - to_int[0]) + to_int[0]


class MultiLayerPerceptron():

    def __init__(
        self,
        learning_rate: float,
        input_size: int,
        hidden_node_sizes: [int],
        output_size: int,
        input_range,
        expected_range,
        activation_func=exp_activation_func,
        activation_func_derivative=exp_activation_func_derivative,
        training_strategy=batch,
        activation_func_range=(0, 1),
        percentage_threshold=0.000001
    ):
        # Add one to the input to have the bias
        dimensions = [input_size + 1] + hidden_node_sizes + [output_size]
        matrices = []
        for i in range(len(dimensions) - 1):
            rows = dimensions[i]
            cols = dimensions[i + 1]
            matrix = np.random.randn(rows, cols)
            matrices.append(matrix)
        self.weights = matrices
        self.min_weights = self.weights
        self.min_error = 100000000

        self.dimensions = dimensions

        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative
        self.training_strategy = training_strategy

        self.input_range = input_range
        self.expected_range = expected_range
        self.activation_func_range = activation_func_range
        self.percentage_threshold = percentage_threshold
        self.initialize_adam_params()  # Initialize Adam parameters

    def initialize_adam_params(self):
        self.m = [np.zeros_like(weight) for weight in self.weights]
        self.v = [np.zeros_like(weight) for weight in self.weights]
        self.beta1 = 0.9  # Recommended value
        self.beta2 = 0.999  # Recommended value
        self.epsilon = 1e-8  # Recommended value
        self.t = 0  # Time step counter

    def predict_with_error(self, input, expected):
        result = self.predict(input)
        error = self.compute_error(np.array(result), np.array(expected))
        return result, error

    def predict(self, input):
        for i in range(len(input)):
            input[i] = feature_scaling(
                input[i], self.input_range, self.activation_func_range)
        x_input = np.array([1] + input)  # Add bias to input
        activations = [x_input.T]

        # Forward propagate input through each layer
        for i in range(len(self.weights)):
            h = activations[-1].dot(self.weights[i])  # Compute weighted sum

            activation = self.activation_func(h)  # Output layer
            activations.append(activation)

        # Return output of the last layer (predictions)
        return feature_scaling(
            activations[-1], self.activation_func_range, self.expected_range)

    def train(self, dataset, expected, max_epochs: int = 1000):
        for i in range(len(expected)):
            expected[i] = feature_scaling(
                np.array(expected[i]), self.expected_range, self.activation_func_range).tolist()

        for epoch in range(max_epochs):
            inputs = self.training_strategy(dataset)
            inputs = [[1] + s for s in inputs]
            dWs = []
            outputs = []
            for input, target in zip(inputs, expected):
                h_outputs, v_inputs = self.forward_propagation(input)
                outputs.append(v_inputs[-1])

                dWs.append(self.backward_propagation(
                    h_outputs, v_inputs, target))

            error = self.compute_error(np.array(outputs), np.array(expected))
            if epoch % 1000 == 0:
                print("epoch:", epoch)
                print("min error:", self.min_error)

            if self.min_error > error:
                self.min_error = error
                self.min_weights = self.weights

            if self.is_converged(error):
                break

            self.update_weights_adam(dWs)
        self.weights = self.min_weights
        return self.min_error

    def forward_propagation(self, input):
        h_outputs = []
        v_inputs = [np.array(input)]
        for i in range(len(self.weights)):
            h = v_inputs[-1].dot(self.weights[i])
            v = self.activation_func(h)
            h_outputs.append(h)
            v_inputs.append(v)
        return h_outputs, v_inputs

    def backward_propagation(self, h_outputs, v_inputs, expected):
        output_error = np.array(expected) - v_inputs[-1]
        delta = np.array(
            output_error * self.activation_func_derivative(h_outputs[-1]))
        deltas = [delta]
        dWs = [np.array(self.learning_rate * np.outer(delta, v_inputs[-2]).T)]
        # Backpropagate the error
        for i in range(len(self.weights) - 2, -1, -1):
            delta = deltas[-1].dot(self.weights[i + 1].T) * \
                self.activation_func_derivative(h_outputs[i])
            dWs.append(np.array(self.learning_rate *
                       np.outer(delta, v_inputs[i]).T))
            deltas.append(delta)
        dWs.reverse()
        return dWs

    def update_weights(self, dWs):
        # Update the weights using the calculated differentials
        for i in range(len(self.weights)):
            for dW in dWs:
                self.weights[i] += dW[i]

    def update_weights_adam(self, dWs):
        # Adam update rules
        self.t += 1
        alpha = self.learning_rate
        for i in range(len(self.weights)):
            for dW in dWs:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * dW[i]
                self.v[i] = self.beta2 * self.v[i] + \
                    (1 - self.beta2) * (dW[i] ** 2)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                self.weights[i] += alpha * m_hat / \
                    (np.sqrt(v_hat) + self.epsilon)

    def is_converged(self, error):
        expected_amplitude = self.expected_range[1] - self.expected_range[0]
        return error < self.percentage_threshold * expected_amplitude

    def compute_error(self, predicted, expected):
        output_errors = predicted - expected
        return np.mean(np.square(output_errors))

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
