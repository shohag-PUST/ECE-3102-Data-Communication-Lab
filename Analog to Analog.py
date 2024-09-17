import numpy as np
import matplotlib.pyplot as plt

# Define time axis
t = np.linspace(0, 1, 1000)  # Time from 0 to 1 second, 1000 samples

# Message signal (analog signal, e.g., sine wave)
fm = 2  # Message signal frequency (Hz)
message_signal = np.sin(2 * np.pi * fm * t)  # Sinusoidal message signal

# Carrier signal (high frequency)
fc = 20  # Carrier signal frequency (Hz)

# AM Modulation (Amplitude Modulation)
carrier_am = np.cos(2 * np.pi * fc * t)  # Carrier signal
am_signal = (1 + message_signal) * carrier_am  # Amplitude modulated signal

# FM Modulation (Frequency Modulation)
kf = 10  # Frequency sensitivity for FM
fm_signal = np.cos(2 * np.pi * fc * t + kf * np.cumsum(message_signal) / 1000)

# PM Modulation (Phase Modulation)
kp = np.pi  # Phase sensitivity for PM
pm_signal = np.cos(2 * np.pi * fc * t + kp * message_signal)

# Plotting the results
plt.figure(figsize=(12, 10))

# Plot Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Message Signal', color='blue')
plt.title('Message Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot AM Signal
plt.subplot(4, 1, 2)
plt.plot(t, am_signal, label='AM Signal', color='green')
plt.title('Amplitude Modulation (AM)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot FM Signal
plt.subplot(4, 1, 3)
plt.plot(t, fm_signal, label='FM Signal', color='red')
plt.title('Frequency Modulation (FM)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot PM Signal
plt.subplot(4, 1, 4)
plt.plot(t, pm_signal, label='PM Signal', color='purple')
plt.title('Phase Modulation (PM)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
