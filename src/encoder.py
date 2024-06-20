from src.multi_layer_perceptron import MultiLayerPerceptron
import numpy as np

class Encoder(MultiLayerPerceptron):
    def __init__(self, learning_rate, inputs, hidden_nodes, hidden_layers, output_nodes, expected_outputs):
        super().__init__(learning_rate, inputs, hidden_nodes, hidden_layers,
                         output_nodes, expected_outputs)
        self.last_delta = 0.1

    def backward_propagation(self, h1_outputs, V1_inputs, h2_output, O_predicted):
        output_errors = self.last_delta
        dO = output_errors * self.activation_func_derivative(h2_output)
        dW_output = self.learning_rate * dO.dot(V1_inputs[-1].T)
        self.weights[-1] += dW_output

        delta_next = dO

        for i in range(len(self.weights) - 2, -1, -1):
            weights_without_bias = self.weights[i + 1][:, 1:]
            delta_current = weights_without_bias.T.dot(
                delta_next) * self.activation_func_derivative(h1_outputs[i])
            dW_hidden = self.learning_rate * delta_current.dot(V1_inputs[i].T)
            num_cols_diff = self.weights[i].shape[1] - dW_hidden.shape[1]
            dW_hidden = np.concatenate((dW_hidden, np.zeros(
                (dW_hidden.shape[0], num_cols_diff))), axis=1)
            self.weights[i] += dW_hidden
            delta_next = delta_current

    def backward_propagation_adam(self, h1_outputs, V1_inputs, h2_output, O_predicted):
        output_errors = self.last_delta
        dO = output_errors * self.activation_func_derivative(h2_output)
        dW_output = self.learning_rate * dO.dot(V1_inputs[-1].T)
        dW = []
        dW.append(dW_output)

        delta_next = dO

        for i in range(len(self.weights) - 2, -1, -1):
            # Exclude bias weights
            weights_without_bias = self.weights[i + 1][:, 1:]
            delta_current = weights_without_bias.T.dot(
                delta_next) * self.activation_func_derivative(h1_outputs[i])
            dW_hidden = self.learning_rate * delta_current.dot(V1_inputs[i].T)
            num_cols_diff = abs(self.weights[i].shape[1] - dW_hidden.shape[1])
            dW_hidden = np.concatenate((dW_hidden, np.zeros(
                (dW_hidden.shape[0], num_cols_diff))), axis=1)
            dW = [dW_hidden] + dW

            delta_next = delta_current
        self.last_delta = delta_next
        self.update_weights_adam(dW)
