import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Generate a sine wave
Fs = 1000  # Sampling rate
T = 1 / Fs  # Sample time
t = np.arange(0, 1, T)  # Time vector
f = 5  # Sine wave frequency
x = np.sin(2 * np.pi * f * t)

# Apply a low pass filter
fc = 10  # Cutoff frequency
b, a = signal.butter(4, fc / (Fs / 2), btype="low")
y = signal.lfilter(b, a, x)

# Plot the original and filtered signals
plt.plot(t, x, label="Original")
plt.plot(t, y, label="Filtered")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
