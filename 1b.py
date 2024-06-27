import numpy as np
from src.autoencoder import Autoencoder
from data.font import Font3
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
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



# add noise to a flatten matrix
def add_toggled_noise(letters, noise):
    all_pixels = range(len(letters[0]))
    pixel_amount = len(all_pixels)
    for i in range(len(letters)):
        noised_pixels = np.random.choice(all_pixels, int(np.round(noise * pixel_amount, 0)))
        for pixel in noised_pixels:
            noise_level = np.random.random()
            letters[i][pixel] += noise_level
            if(letters[i][pixel] >=1):
                letters[i][pixel] = 1
        
    return letters

# add noise to a flatten matrix
def add_zeroed_noise(letters, noise):
    all_pixels = range(len(letters[0]))
    pixel_amount = len(all_pixels)
    for i in range(len(letters)):
        noised_pixels = np.random.choice(all_pixels, int(np.round(noise * pixel_amount, 0)))
        for pixel in noised_pixels:
            letters[i][pixel] = 0

    return letters
 

def is_same_letter(originals: list[float], predictions: list[float], max_errors=4):
    wrong_letters = []
    wrong_predictions = []
    for i in range(len(originals)):
        errors = 0
        letter = originals[i]
        letter_pred = predictions[i]
        for j in range(len(letter)):
            if ((letter[j] == 1 and letter_pred[j] < 0.7) or (letter[j] == 0 and letter_pred[j] > 0.3)):
                errors += 1
                if errors > max_errors:
                    wrong_letters.append(i)
                    wrong_predictions.append(letter_pred)
                    break
    return wrong_letters, wrong_predictions

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

def add_noise(array: np.array, noise_level: float) -> np.array:
    """Add Gaussian noise to the given array.

    Args:
        array: Array to add noise to
        noise_level: Noise level in [0, 1]

    Returns:
        A new array with added noise
    """
    return np.clip(array + np.random.normal(loc=0, scale=noise_level, size=array.shape), 0, 1)


if __name__ == "__main__":

    b = get_bit_array(Font3)
    print(np.shape(b))
    
    learning_rates = 0.0005
    hidden_layers6 = [35,25,20,15,10,5]
    hidden_layers2 = [35,25]
    hidden_layers4 = [30,25,20,15]

    hidden_layers = [[35,25,20,15,10,5]]
    noise_level = 0.1
    
    noised_letters = add_toggled_noise(copy.deepcopy(b),noise_level)
    print(np.shape(noised_letters))
    
    # configs = [2,4,6]
    # config_results = []
    # for hidden in hidden_layers :
    #     for _ in range(10):
    #         ae = Autoencoder(learning_rates, len(np.array(noised_letters)[0]), hidden, (0, 1), (0, 1))
    #         start = time.time()
    #         error, all_errors = ae.train_with_noise(noised_letters,b,add_toggled_noise,0.3,25000)
    #         end = time.time()
    #         config_results.append({'Config': len(hidden), 'Training Time': (end-start), 'Min Error': error})
    #
    # df = pd.DataFrame(config_results)
    # # print(df)
    # df.to_csv('config_results2.csv', index=False)
    # exit()

    # ae = Autoencoder(learning_rates,len(np.array(noised_letters)[0]),hidden_layers6,(0,1),(0,1))
    # error, all_errors = ae.train_with_noise(noised_letters,b,add_toggled_noise,noise_level,25000)

    # plt.figure(figsize=(10, 6))

    # ae.save_model(f"./models/ae/autoencoder_lr{learning_rates}_hidden{len(hidden_layers6)}_error{error}_noise{noise_level}.pkl")
# # Iterate through each configuration and plot
    # exit()
#     for config in configs:
#         config_data = df[df['Config'] == config]
#         plt.scatter(config_data['Training Time'], config_data['Min Error'], label=config, alpha=0.7)
#
# # Adding labels and title
#     plt.xlabel('Training Time (minutes)')
#     plt.ylabel('Minimum Error')
#     plt.title('Comparison of Different Configurations')
#     plt.grid(True)
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
    ae = Autoencoder.load_model("./models/ae/autoencoder_lr0.0005_noise0.1_hidden4_error0.006253562707453896.pkl")
    results = []
    errors = []
    latent_iamges = []
    pixel_good  = 0
    for _ in range(10):
        input_noised_letters = add_toggled_noise(copy.deepcopy(b),noise_level)
        for i in range(len(b)):
            result, error = ae.mlp.predict_with_error(input_noised_letters[i], b[i])
            results.append(result)
            errors.append(error)
            # if(error < ae.mlp.min_error):
                # pixel_good.append(i)
        
            latent_iamges.append(ae.encode(b[i]))
        wrong_letter, wrong_pred = is_same_letter(b,results)
        # print(wrong_letter)
        pixel_good += len(wrong_letter)
        results = []
    
    
    # print(pixel_good)
    
    # print(wrong_letter)
    print(pixel_good)
    print(len(b)*10)
    # render_results(np.array(input_noised_letters))
    # render_results(np.array(results))
    # plt.show()
    exit()

    summary_df = df.groupby('Config').agg({'Training Time': 'first', 'Min Error': ['mean', 'std']}).reset_index()
    summary_df.columns = ['Config', 'Training Time', 'Mean Min Error', 'Std Min Error']

# Print or manipulate the Pandas DataFrame
    print("Summary DataFrame:")
    print(summary_df)

# Plotting with error bars
    plt.figure(figsize=(10, 6))

# Iterate through each configuration and plot with error bars
    for i, config in enumerate(configs):
        config_data = df[df['Config'] == config]
        mean_error = summary_df.loc[i, 'Mean Min Error']
        std_error = summary_df.loc[i, 'Std Min Error']
        plt.errorbar(summary_df.loc[i, 'Training Time'], mean_error, yerr=std_error, fmt='o', label=config, alpha=0.7)

# Adding labels and title
    plt.xlabel('Training Time (minutes)')
    plt.ylabel('Minimum Error')
    plt.title('Comparison of Different Configurations with Error Bars')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    exit()

      # print(errors)
    # render_results(results)
    print(error)
    print(end - start)
    # render_results(np.array(input_noised_letters))
    

# Example data (replace with your actual data)
      # Example: epochs from 1 to 100
      # Your actual error values corresponding to each epoch
    

# Example data (replace with your actual data)
    epochs = np.arange(1, 25001)

# Define number of bins or groups
    num_bins = 1000
    bins = np.linspace(1, len(epochs), num_bins + 1, dtype=int)

# Aggregate errors within each bin
    mean_errors = [np.mean(all_errors[bins[i]:bins[i+1]]) for i in range(num_bins)]
    std_errors = [np.std(all_errors[bins[i]:bins[i+1]]) for i in range(num_bins)]

# Plotting aggregated data
    plt.errorbar(range(1, num_bins + 1), mean_errors, yerr=std_errors, fmt='o', color='blue', ecolor='lightgray', elinewidth=2, capsize=5, capthick=2)

# Customize your plot
    plt.xlabel('Groups of Epochs')
    plt.ylabel('Mean Error')
    plt.title('Aggregated Epochs vs Mean Error')
    plt.grid(True)
    plt.show()

        
    plt.plot(epochs, all_errors, 'o', markersize=2, color='blue', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Epochs vs Error')
    plt.grid(True)
    plt.show()


