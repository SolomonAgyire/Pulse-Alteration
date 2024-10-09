import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Required for heatmap

# Load the data
def load_data(filepath):
    params = []
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                print('Header:', line.strip())  # Print the header, first line
            else:
                elements = line.strip().split(',')
                params.append([float(elm) for elm in elements])
    return np.array(params)

# Filepath for the triple pulse data
filepath = 'double_pulses_output.txt'
double_pulse = load_data(filepath)
pulse_data = double_pulse

# Function to plot multiple samples
def plot_samples(data, title, num_samples=100):
    plt.figure(figsize=(20, 15))  # Adjust figure size as needed
    for i in range(min(num_samples, len(data))):  # Ensure we don't exceed available samples
        plt.subplot(10, 10, i + 1)  # Create a 10x10 grid for subplots
        plt.plot(data[i, :], label=f'Sample {i}')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.title(f'{title} {i}', fontsize=8)
        plt.tight_layout()
    plt.show()

# Plot the first 100 samples
plot_samples(pulse_data, 'Double Pulses', num_samples=100)

