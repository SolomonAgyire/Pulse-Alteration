import numpy as np
import matplotlib.pyplot as plt


params = []

with open('x1.txt', 'r') as file:
    i = 0
    for line in file:
        if i == 0:
            print('header: ', line.strip().split())
            i += 1
        else:
            elements = line.strip().split()
            elements = [float(elm) for elm in elements]
            params.append(elements)

params = np.array(params)

# Extract relevant parameters for the pulses
t1 = params[:, 0:1]  # Starting position of the first pulse
a21 = params[:, 1:2] # Amplitude of the first pulse
k31 = params[:, 2:3] # Rise rate of the first pulse
k41 = params[:, 3:4] # Decay rate of the first pulse

# Number of training samples
n_train = len(params)

# Generate time differences (dt) for the pulses
dt1 = np.random.uniform(1/250, 0.6, (n_train, 1)) # Time difference for first pulse
dt2 = np.random.uniform(1/250, 0.6, (n_train, 1)) # Time difference for second pulse
dt3 = np.random.uniform(1/250, 0.6, (2*n_train, 1)) # Time difference for third pulse

# Calculating the adjusted starting times for the pulses
t21 = np.clip(t1 + dt1, 0.408, 1-1/250) # Adjusted Start time for first pulse
t22 = np.clip(t1 + dt2, 0.408, 1-1/250) # Adjusted Start time for second pulse
t23 = np.clip(t1 + dt3, 0.408, 1-1/250) # Adjusted Start time for third pulse

#parameters to generate variability in the pulses
per1 = np.random.permutation(len(a21))
per2 = np.random.permutation(len(a21))
per3 = np.random.permutation(len(a21))


# Extract parameters for the second pulse
a22 = a21[per1]  # Amplitude of the second pulse
k32 = k31[per1]  # Rise rate of the second pulse
k42 = k41[per1]  # Decay rate of the second pulse

# Extract parameters for the third pulse
a23 = a21[per2]  # Amplitude of the third pulse
k33 = k31[per2]  # Rise rate of the third pulse
k43 = k41[per2]  # Decay rate of the third pulse

# Generate time series for the waveform
t = np.arange(250).reshape(-1, 1).T / 250  # 250 time points

# Generate the waveform for each pulse
p21 = a21 * np.exp(-k31 * (t - t21)) / (1 + np.exp(-k41 * (t - t21)))  # First pulse
p22 = a22 * np.exp(-k32 * (t - t22)) / (1 + np.exp(-k42 * (t - t22)))  # Second pulse
p23 = a23 * np.exp(-k33 * (t - t23)) / (1 + np.exp(-k43 * (t - t23)))  # Third pulse

# Combine the pulses to form the final waveform
Final_waveform = p21 + p22 + p23

# Plot the result
#plt.figure(figsize=(10, 6))
#plt.plot(t.flatten(), Final_waveform.flatten(), label='Triple Pulse Waveform')
#plt.xlabel('Time (s)')
#plt.ylabel('Intensity')
#plt.title('Generated Triple Pulse Waveform')
#plt.legend()
#plt.show()

# Plot the individual waveforms and the combined waveform
plt.figure(figsize=(12, 8))
plt.plot(t, p21.T, label='First Pulse', linestyle='--')
plt.plot(t, p22.T, label='Second Pulse', linestyle='--')
plt.plot(t, p23.T, label='Third Pulse', linestyle='--')
plt.plot(t, Final_waveform.T, label='Combined Waveform', color='black')
plt.title('Individual and Combined Waveforms from Three Pulses')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()