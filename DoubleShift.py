import numpy as np
import matplotlib.pyplot as plt

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

def shift_pulse_clean(pulse, ofs, pulse_length):
    min_shift = int((1/250) * pulse_length)
    max_shift = int(0.4 * pulse_length)
    shift = np.random.randint(min_shift, max_shift)
    shifted_pulse = np.roll(pulse, shift)
    shifted_pulse[:shift] = ofs  # Fill the start with 'ofs' after shifting
    return shifted_pulse

if __name__ == "__main__":
    filepath = 'data_all_params.txt'
    data = load_data(filepath)
    parameters = data[:, :5]
    pulse_data = data[:, 5:]

    plt.figure(figsize=(10, 5))  # Initialize the plot outside the loop if comparing pulses

    with open('double_pulses_output.txt', 'w') as output_file:
        for i in range(len(pulse_data)):  # Process and plot the first 5 pulses
            original_pulse = pulse_data[i]
            ofs = parameters[i, 4]

            shifted_pulse = shift_pulse_clean(original_pulse, ofs, len(original_pulse))
            double_pulse = original_pulse + shifted_pulse

            double_pulse = np.round(double_pulse, 3)
            
            time_axis = np.arange(len(original_pulse))

            output_file.write(','.join(f"{value:.3f}" for value in double_pulse) + '\n')
    print("Double pulses have been saved to 'double_pulses_output.txt'.")

    # Print the shape of the data (rows, columns)
    print(f'Data shape: {data.shape}')
    print(f'Number of rows: {data.shape[0]}')
    print(f'Number of columns: {data.shape[1]}')

    parameters = data[:, :5]
    pulse_data = data[:, 5:]

    # Optional: Print the shape of parameters and pulse_data if needed
    print(f'Parameters shape: {parameters.shape}')
    print(f'Pulse data shape: {pulse_data.shape}')