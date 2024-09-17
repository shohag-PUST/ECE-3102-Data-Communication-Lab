'''import numpy as np
import matplotlib.pyplot as plt

# Function to generate binary input
def get_binary_input():
    binary_string = input("Enter a binary string: ")
    binary_list = [int(bit) for bit in binary_string]
    return binary_list

# ASK modulation
def ask_modulation(binary_signal, carrier_freq, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    ask_signal = np.array([])
    for bit in binary_signal:
        if bit == 0:
            carrier = np.sin(2 * np.pi * carrier_freq * t)
        else:
            carrier = 2 * np.sin(2 * np.pi * carrier_freq * t)
        ask_signal = np.concatenate((ask_signal, carrier))
    return ask_signal

# FSK modulation
def fsk_modulation(binary_signal, carrier_freq1, carrier_freq2, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    fsk_signal = np.array([])
    for bit in binary_signal:
        if bit == 0:
            carrier = np.sin(2 * np.pi * carrier_freq1 * t)
        else:
            carrier = np.sin(2 * np.pi * carrier_freq2 * t)
        fsk_signal = np.concatenate((fsk_signal, carrier))
    return fsk_signal

# PSK modulation
def psk_modulation(binary_signal, carrier_freq, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    psk_signal = np.array([])
    if binary_signal[0] == 0:
        last_bit = 1
    elif binary_signal[0] == 1:
        last_bit = -1
    for bit in binary_signal:
        if bit == 0:
            if last_bit == 1:
                last_bit = -last_bit
            carrier = np.sin(2 * np.pi * carrier_freq * t * last_bit)
        else:
            if last_bit == 0:
                last_bit = -last_bit
            carrier = np.sin(2 * np.pi * carrier_freq * t * last_bit + np.pi)
        psk_signal = np.concatenate((psk_signal, carrier))
    return psk_signal

# Plotting function
def plot_signals(binary_signal, ask_signal, fsk_signal, psk_signal):
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.step(range(len(binary_signal)), binary_signal, where='post')
    plt.title('Binary Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(ask_signal)
    plt.title('ASK Modulation')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(fsk_signal)
    plt.title('FSK Modulation')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(psk_signal)
    plt.title('PSK Modulation')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Get binary input
    binary_signal = get_binary_input()

    # Modulation parameters
    carrier_freq = 5  # in Hz
    carrier_freq1 = 5  # in Hz for FSK (0)
    carrier_freq2 = 10  # in Hz for FSK (1)
    sampling_rate = 1000  # in Hz
    duration = 1  # in seconds for each bit

    # Perform modulations
    ask_signal = ask_modulation(binary_signal, carrier_freq, sampling_rate, duration)
    fsk_signal = fsk_modulation(binary_signal, carrier_freq1, carrier_freq2, sampling_rate, duration)
    psk_signal = psk_modulation(binary_signal, carrier_freq, sampling_rate, duration)

    # Plot signals
    plot_signals(binary_signal, ask_signal, fsk_signal, psk_signal)

if __name__ == "__main__":
    main()
'''

'''import matplotlib.pylab as plt
import numpy as num

# AMPLITUDE SHIFT KEYING.....................
F1 = 10
F2 = 2
A = 5
t = num.arange(0, 1, 0.001)
x = A * num.sin(2 * num.pi * F1 * t)
u = []
b = [0.2, 0.4, 0.6, 0.8, 1.0]

s = 1
for i in t:
    if (i == b[0]):
        b.pop(0)
        if (s == 0):
            s = 1
        else:
            s = 0

    u.append(s)
v = []
for i in range(len(t)):
    v.append(A * num.sin(2 * num.pi * F1 * t[i]) * u[i])

plt.subplot(3, 3, 1)
plt.plot(t, x)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Carrier')
plt.grid(True)

plt.subplot(3, 3, 4)
plt.plot(t, u)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Message Signal')
plt.grid(True)

plt.subplot(3, 3, 7)
plt.plot(t, v)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('ASK Signal')
plt.grid(True)
plt.tight_layout()

# FREQUENCY SHIFT KEYING............................

t = num.arange(0, 1, 0.001)
x = A * num.sin(2 * num.pi * F1 * t)

plt.subplot(3, 3, 2)
plt.plot(t, x)
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.title("Carrier")
plt.grid(True)

u = []
b = [0.2, 0.4, 0.6, 0.8, 1.0]
s = 1
for i in t:
    if (i == b[0]):
        b.pop(0)
        if (s == 0):
            s = 1
        else:
            s = 0
    u.append(s)

plt.subplot(3, 3, 5)
plt.plot(t, u)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Message Signal')
plt.grid(True)

v = []
for i in range(len(t)):
    if (u[i] == 1):
        v.append(A * num.sin(2 * num.pi * F1 * t[i]))
    else:
        v.append(num.sin(2 * num.pi * F1 * t[i]) * -1)

plt.subplot(3, 3, 8)
plt.plot(t, v)
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.title("FSK")
plt.grid(True)
plt.tight_layout()

# PHASE SHIFT KEYING................................

t = num.arange(0, 1, 0.001)
x = A * num.sin(2 * num.pi * F1 * t)

plt.subplot(3, 3, 3)
plt.plot(t, x)
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.title("Carrier")
plt.grid(True)

u = []
b = [0.2, 0.4, 0.6, 0.8, 1.0]
s = 1
for i in t:
    if (i == b[0]):
        b.pop(0)
        if (s == 0):
            s = 1
        else:
            s = 0
    u.append(s)

plt.subplot(3, 3, 6)
plt.plot(t, u)
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Message Signal')
plt.grid(True)

v = []
for i in range(len(t)):
    if (u[i] == 1):
        v.append(A * num.sin(2 * num.pi * F1 * t[i]))
    else:
        v.append(A * num.sin(2 * num.pi * F1 * t[i]) * -1)

plt.subplot(3, 3, 9)
plt.plot(t, v)
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.title("PSK")
plt.grid(True)
plt.tight_layout()
plt.show()
'''




import numpy as np
import matplotlib.pyplot as plt

# User Input for binary sequence
data_input = input("Enter binary sequence (e.g., 10101): ")
data = np.array([int(bit) for bit in data_input])

# Predefined parameters
bit_duration = 1  # in seconds
sampling_rate = 1000  # samples per second
carrier_frequency = 5  # in Hz

# Time axis for each bit
t = np.linspace(0, len(data) * bit_duration, len(data) * sampling_rate)

# Define a carrier signal
carrier = np.sin(2 * np.pi * carrier_frequency * t)

# Amplitude Shift Keying (ASK)
def ask_modulate(data, carrier, sampling_rate, bit_duration):
    signal = np.array([])
    for bit in data:
        if bit == 1:
            modulated_bit = carrier[:int(sampling_rate * bit_duration)]
        else:
            modulated_bit = np.zeros(int(sampling_rate * bit_duration))
        signal = np.concatenate((signal, modulated_bit))
    return signal

# Frequency Shift Keying (FSK)
def fsk_modulate(data, carrier_frequency, sampling_rate, bit_duration):
    signal = np.array([])
    for bit in data:
        if bit == 1:
            modulated_bit = np.sin(2 * np.pi * carrier_frequency * t[:int(sampling_rate * bit_duration)])
        else:
            modulated_bit = np.sin(2 * np.pi * (carrier_frequency / 2) * t[:int(sampling_rate * bit_duration)])
        signal = np.concatenate((signal, modulated_bit))
    return signal

# Phase Shift Keying (PSK)
def psk_modulate(data, carrier, sampling_rate, bit_duration):
    signal = np.array([])
    for bit in data:
        if bit == 1:
            modulated_bit = carrier[:int(sampling_rate * bit_duration)]
        else:
            modulated_bit = -carrier[:int(sampling_rate * bit_duration)]  # 180-degree phase shift
        signal = np.concatenate((signal, modulated_bit))
    return signal

# Generate modulated signals based on the user's input
ask_signal = ask_modulate(data, carrier, sampling_rate, bit_duration)
fsk_signal = fsk_modulate(data, carrier_frequency, sampling_rate, bit_duration)
psk_signal = psk_modulate(data, carrier, sampling_rate, bit_duration)

# Create subplots to display the three modulation techniques
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# ASK Plot
axs[0].plot(t, ask_signal)
axs[0].set_title('Amplitude Shift Keying (ASK)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')

# FSK Plot
axs[1].plot(t, fsk_signal)
axs[1].set_title('Frequency Shift Keying (FSK)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Amplitude')

# PSK Plot
axs[2].plot(t, psk_signal)
axs[2].set_title('Phase Shift Keying (PSK)')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()
