import numpy as np
import matplotlib.pyplot as plt

fc = 5
fc_fsk0 = 3
fc_fsk1 = 8
fs = 1000
bit_duration = 1
t = np.linspace(0, bit_duration, fs * bit_duration)

data = np.random.randint(0, 2, 5)
print("Random Data:", data)

ask_signal = np.array([])
fsk_signal = np.array([])
psk_signal = np.array([])

for bit in data:
    carrier_ask = np.cos(2 * np.pi * fc * t)
    ask_bit = bit * carrier_ask
    ask_signal = np.concatenate((ask_signal, ask_bit))

    if bit == 0:
        carrier_fsk = np.cos(2 * np.pi * fc_fsk0 * t)
    else:
        carrier_fsk = np.cos(2 * np.pi * fc_fsk1 * t)
    fsk_signal = np.concatenate((fsk_signal, carrier_fsk))

    if bit == 0:
        carrier_psk = np.cos(2 * np.pi * fc * t)
    else:
        carrier_psk = np.cos(2 * np.pi * fc * t + np.pi)  # Phase shift of 180° for bit 1
    psk_signal = np.concatenate((psk_signal, carrier_psk))  # Append bit to signal

# Time vector for the entire signal
total_time = np.linspace(0, len(data) * bit_duration, len(data) * len(t))

# Plotting the results
plt.figure(figsize=(12, 10))

# Plot ASK Signal
plt.subplot(3, 1, 1)
plt.plot(total_time, ask_signal, label="ASK Signal", color='blue')
plt.title("Amplitude Shift Keying (ASK)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot FSK Signal
plt.subplot(3, 1, 2)
plt.plot(total_time, fsk_signal, label="FSK Signal", color='green')
plt.title("Frequency Shift Keying (FSK)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot PSK Signal
plt.subplot(3, 1, 3)
plt.plot(total_time, psk_signal, label="PSK Signal", color='red')
plt.title("Phase Shift Keying (PSK)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
