import numpy as np
import pickle
from src.autoencoder import Autoencoder
from src.multi import MultiLayerPerceptron
import matplotlib.pyplot as plt


# Function to reshape and display an image
def display_image(vector_image, image_size=(72, 72)):
    # Reshape the vector to 72x72x3 (assuming the image has 3 color channels)
    reshaped_image = vector_image.reshape(image_size[0], image_size[1], 4)

    # Display the image
    plt.imshow(reshaped_image.astype(np.uint8))
    plt.axis('off')
    plt.show()


def batch(X):
    return X


def sigmoid_activation_func(value):
    clipped_value = np.clip(value, -500, 500)
    return 1 / (1 + np.exp(-clipped_value))


def sigmoid_activation_func_derivative(value):
    sigmoid_output = sigmoid_activation_func(value)
    return sigmoid_output * (1 - sigmoid_output)


def feature_scaling(value: float, from_int: tuple[float, float], to_int: tuple[float, float]) -> float:
    numerator = value - from_int[0]
    denominator = from_int[1] - from_int[0]
    return (numerator / denominator) * (to_int[1] - to_int[0]) + to_int[0]


class VariationalAutoencoder():
    def __init__(
        self,
        learning_rate: float,
        input_size: int,
        hidden_node_sizes: [int],
        latent_dim: int,
        input_range,
        expected_range,
        activation_func=sigmoid_activation_func,
        activation_func_derivative=sigmoid_activation_func_derivative,
        training_strategy=batch,
        activation_func_range=(0, 1),
        percentage_threshold=0.000001,
        adaptative_eta=False,
        kl_weight=1.0
    ):
        reverse = hidden_node_sizes[::-1]
        self.latent_size = latent_dim
        self.expected_range = expected_range
        self.activation_func_range = activation_func_range
        self.min_error = 100000000
        self.percentage_threshold = percentage_threshold
        self.kl_weight = kl_weight
        self.encoder = MultiLayerPerceptron(
            learning_rate,
            input_size,
            hidden_node_sizes,
            latent_dim * 2,
            input_range,
            (0, 1),
            activation_func,
            activation_func_derivative,
            training_strategy,
            activation_func_range,
            percentage_threshold,
            adaptative_eta
        )
        self.decoder = MultiLayerPerceptron(
            learning_rate,
            latent_dim,
            reverse,
            input_size,
            (0, 1),
            expected_range,
            activation_func,
            activation_func_derivative,
            training_strategy,
            activation_func_range,
            percentage_threshold,
            adaptative_eta
        )
        print(self.encoder.dimensions)
        print(self.decoder.dimensions)

    def compute_error(self, predicted, expected):
        output_errors = predicted - expected
        return np.mean(np.square(output_errors))

    def compute_error_derivative(self, predicted, expected):
        output_errors = predicted - expected
        return 2 * abs(output_errors)

    def reparam(self, means: np.array, log_covs: np.array):
        epsilon = np.random.normal(size=means.shape)
        return epsilon, means + epsilon * np.exp(log_covs / 2) - 1

    def _vae_loss(self, expected, predicted, z_mean, z_log_var):
        # reconstruction_loss = self._binary_cross_entropy(expected, predicted)
        reconstruction_loss = self._binary_cross_entropy(expected, predicted)
        kl_loss = self._kl_divergence(z_mean, z_log_var) * self.kl_weight
        d_reconstruction_loss = self._binary_cross_entropy_derivative(
            expected, predicted)
        return reconstruction_loss, kl_loss, d_reconstruction_loss

    def _binary_cross_entropy(self, expected, predicted, epsilon=1e-7):
        P = np.clip(predicted, epsilon, 1 - epsilon)
        return np.mean(-expected * np.log(P) - (1 - expected) * np.log(1 - P))

    def _binary_cross_entropy_derivative(self, expected, predicted, epsilon=1e-7):
        P = np.clip(predicted, epsilon, 1 - epsilon)
        return (P - expected) / (P * (1 - P))

    def _kl_divergence(self, z_mean: np.array, z_log_var: np.array):
        return -0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))

    def is_converged(self, error):
        expected_amplitude = self.expected_range[1] - self.expected_range[0]
        return error < self.percentage_threshold * expected_amplitude

    def train(self, dataset, expected, max_epochs: int = 1000):
        for i in range(len(expected)):
            expected[i] = feature_scaling(
                np.array(expected[i]), self.expected_range, self.activation_func_range).tolist()

        inputs = [np.concatenate(([1], s)) for s in dataset]
        for epoch in range(max_epochs):
            decoder_dWs = []
            encoder_dWs = []
            img = []
            outputs = []
            total_loss = 0
            for input, target in zip(inputs, expected):
                encoder_h_outputs, encoder_v_inputs = activations = self.encoder.forward_propagation(
                    input)

                activations = encoder_v_inputs[-1]

                means = activations[:self.latent_size]
                log_covs = activations[self.latent_size:]

                epsilon, z = self.reparam(means, log_covs)

                z = np.concatenate(([1], z))
                decoder_h_outputs, decoder_v_inputs = self.decoder.forward_propagation(
                    z)

                outputs.append(decoder_v_inputs[-1])

                reconstruction_loss, kl_loss, d_reconstruction_loss = self._vae_loss(
                    np.array(target), decoder_v_inputs[-1], means, log_covs)
                total_loss += reconstruction_loss + kl_loss

                error = total_loss

                prev_layer_delta_sum, decoder_dW = self.decoder.backward_propagation_error(
                    decoder_h_outputs, decoder_v_inputs, target, -1 * d_reconstruction_loss)
                decoder_dWs.append(decoder_dW)

                kl_gradients = np.append(means, 0.5 * (np.exp(log_covs) - 1))
                rec_gradients = np.append(
                    prev_layer_delta_sum[1:],
                    prev_layer_delta_sum[1:] * epsilon *
                    np.exp(log_covs * 0.5) * 0.5
                )
                gradients = [x + y for x,
                             y in zip(kl_gradients, rec_gradients)]

                _, encoder_dW = self.encoder.backward_propagation_error(
                    encoder_h_outputs, encoder_v_inputs, gradients, gradients)
                encoder_dWs.append(encoder_dW)

            self.decoder.update_weights_adam(decoder_dWs)
            self.encoder.update_weights_adam(encoder_dWs)
            error = self.compute_error(np.array(outputs), np.array(expected))

            if epoch % 100 == 0:
                print("epoch:", epoch)
                print("min error:", self.min_error)
                # display_image(np.array(img))

            if self.min_error > error:
                self.min_error = error
                self.encoder.min_weights = self.encoder.weights
                self.decoder.min_weights = self.decoder.weights

            if self.is_converged(error):
                break

        self.encoder.weights = self.encoder.min_weights
        self.decoder.weights = self.decoder.min_weights
        return self.min_error

    def predict_latent(self, input):
        input = np.concatenate(([1], input))
        print(input)
        _, v_in = self.encoder.forward_propagation(input)
        means = v_in[-1][:self.latent_size]
        log_covs = v_in[-1][self.latent_size:]
        epailon, z = self.reparam(means, log_covs)
        z = np.concatenate(([1], z))
        return z

    def predict(self, latent_img):
        print(latent_img)
        res = self.decoder.predict(latent_img)
        print(res)
        return res
