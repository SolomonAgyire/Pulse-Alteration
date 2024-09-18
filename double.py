import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

'''# Load the data
file_path = 'x1.txt' 
data = pd.read_csv(file_path, header=None, delim_whitespace=True)  # Adjust delimiter if needed

# Display basic information about the data
print("Data shape:", data.shape)
print("First few rows of the data:")
print(data.head())

# Plotting a few rows to get a sense of the data
plt.figure(figsize=(12, 6))
for i in range(min(5, len(data))):  # Plotting the first 5 rows, or fewer if there are less than 5
    plt.plot(data.iloc[i], label=f'Row {i}')

plt.title('Visualizing Pulse Data from Several Rows')
plt.xlabel('Time Point (or Index)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


'''
'''
import numpy as np

# Define the single pulse function G(x)
def G(x, A1, k1, k2, T1, O):
    return A1 * np.exp(-k1 * (x - T1)) / (1 + np.exp(-k2 * (x - T1))) + O

# Define the double pulse function F(x)
def F(x, A1, k1, k2, T1, A2, k3, k4, T2, O):
    first_pulse = A1 * np.exp(-k1 * (x - T1)) / (1 + np.exp(-k2 * (x - T1)))
    second_pulse = A2 * np.exp(-k3 * (x - T2)) / (1 + np.exp(-k4 * (x - T2)))
    return first_pulse + second_pulse + O

# Parameters for G(x)
A1, k1, k2, T1, O = 1.0, 0.5, 0.5, 10, 0.1

# Parameters for F(x)
A2, k3, k4, T2 = 0.8, 0.7, 0.5, 12  # k4 is the same as k2 according to your description

# Generating data
x_values = np.linspace(0, 30, 400)
G_values = G(x_values, A1, k1, k2, T1, O)
F_values = F(x_values, A1, k1, k2, T1, A2, k3, k4, T2, O)

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(x_values, G_values, label='Single Pulse G(x)')
plt.plot(x_values, F_values, label='Double Pulse F(x)', linestyle='--')
plt.title('Single vs. Double Pulse Simulation')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.legend()
plt.show()'''

data_dir = 'C:\\Users\\sagyi\\OneDrive\\Desktop\\TP'

import numpy as np
import matplotlib.pyplot as plt

params = []
with open(data_dir + '\\x1.txt', 'r') as file:
    i = 0
    for line in file:
        if i == 0:
            print('header:', line.strip())
            i += 1
        else:
            elements = line.strip().split()  # Split by whitespace
            elements = [float(elm) for elm in elements]
            params.append(elements)

# Convert list to numpy array for easier manipulation
params = np.array(params)

# Extract specific parameters
t1 = params[:, 0:1]  # Assuming the first column is time
n_train = len(params)  # Number of training examples based on parameters list

# Random time offsets
dt1 = np.random.uniform(1/250, 0.6, (n_train, 1))
dt2 = np.random.uniform(1/250, 0.6, (n_train, 1))
dt3 = np.random.uniform(1/250, 0.6, (2*n_train, 1))

# Calculate new time points within bounds
t21 = np.clip(t1 + dt1, 0.408, 1-1/250)
t22 = np.clip(t1 + dt2, 0.408, 1-1/250)
t23 = np.clip(t1 + dt3[:n_train], 0.408, 1-1/250)

# Amplitude and decay parameters
a21 = params[:, 1:2]
k31 = params[:, 2:3]
k41 = params[:, 3:4]

# Permutation to introduce variability
per1 = np.random.permutation(n_train)
per2 = np.random.permutation(n_train)

a22 = a21[per1]
k32 = k31[per1]
k42 = k41[per1]

a23 = a21[per2]
k33 = k31[per2]
k43 = k41[per2]

# Normalized time scale based on an assumed data length (update as necessary)
t = np.arange(250).reshape(-1, 1).T / 250  # Modify '250' if your data has a different number of points

# Pulse shape calculations
p21 = a21 * np.exp(-k31 * (t - t21)) / (1 + np.exp(-k41 * (t - t21)))
p22 = a22 * np.exp(-k32 * (t - t22)) / (1 + np.exp(-k42 * (t - t22)))
p23 = a23 * np.exp(-k33 * (t - t23)) / (1 + np.exp(-k43 * (t - t23)))

# Summing up all pulses to simulate a final dataset
final_pulse_data = p21 + p22 + p23

'''# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t.flatten(), final_pulse_data.T, alpha=0.6)
plt.title('Simulated Pulse Data')
plt.xlabel('Time (normalized)')
plt.ylabel('Amplitude')
plt.show()'''



import matplotlib.pyplot as plt

# Assuming final_pulse_data is a numpy array
plt.figure(figsize=(12, 6))
num_rows = min(5, final_pulse_data.shape[0])  # Ensure you do not go out of bounds
for i in range(num_rows):  # Plotting the first 5 rows, or fewer if there are fewer than 5
    plt.plot(final_pulse_data[i], label=f'Row {i}')  # Access rows in numpy array using standard indexing

plt.title('Visualizing Pulse Data from Several Rows')
plt.xlabel('Time Point (or Index)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
