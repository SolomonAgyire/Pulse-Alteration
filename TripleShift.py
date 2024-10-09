import numpy as np
import matplotlib.pyplot as plt

# Function to load data from a file
def load_data(filepath):
    params = []
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            elements = line.strip().split()
            params.append([float(elm) for elm in elements])
    return np.array(params)

# Function to calculate and shift pulse
def shift_pulse_clean(pulse, pulse_length):
    min_shift = int((1/250) * pulse_length)  # a little over 0
    max_shift = int(0.6 * pulse_length)      # Maximum shift

    # Shift using any random number between the min and max shift
    shift = np.random.randint(min_shift, max_shift)
    shifted_pulse = np.roll(pulse, shift)

    return shifted_pulse

# Function to subtract the average of the first 10 samples from the pulse (to remove offset)
def remove_offset(pulse_data):
    first_10_avg = np.mean(pulse_data[:10])  # Calculate the average of the first 10 samples
    return pulse_data - first_10_avg         # Subtract the offset from the pulse

if __name__ == "__main__":
    filepath = 'x1.txt'
    data = load_data(filepath)
    pulse_data = data

    # Remove offset (noise) from the original pulse data before shuffling
    pulse_data = np.array([remove_offset(pulse) for pulse in pulse_data])
    
    # Shuffle dataset after removing the offset
    shuffled_indices = np.random.permutation(len(pulse_data))

    plt.figure(figsize=(10, 5))

    with open('triple_pulses_output.txt', 'w') as output_file:
        for i in range(len(pulse_data)):  # Process each pulse
            original_pulse = pulse_data[i]

            # Print original ad shuffled pulses 
            print(f"Original Pulse Index: {i}, Shuffled Pulse Index: {shuffled_indices[i]}")

            # Check pairing of original and shifted pulses
            if i == shuffled_indices[i]:
                print(f"Warning: Pulse {i} is paired with itself after shuffling!")
            else:
                print(f"Pulse {i} is paired with Pulse {shuffled_indices[i]}")

            # First shift and add to create the double pulse
            shifted_pulse1 = shift_pulse_clean(pulse_data[shuffled_indices[i]], len(original_pulse))
            double_pulse = original_pulse + shifted_pulse1

            # Second shift to create the triple pulse
            shifted_pulse2 = shift_pulse_clean(pulse_data[shuffled_indices[i]], len(original_pulse))
            triple_pulse = double_pulse + shifted_pulse2

            # Write the triple pulse data to a file
            output_file.write(','.join(str(value) for value in triple_pulse) + '\n')

    print("Triple pulses have been saved to 'triple_pulses_output.txt'.")

    # Print the shape of the data (rows, columns)
    print(f'Data shape: {data.shape}')
    print(f'Number of rows: {data.shape[0]}')
    print(f'Number of columns: {data.shape[1]}')
