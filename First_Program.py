'''import matplotlib.pyplot as plt
import numpy as np


def get_user_input():
    user_input = input("Please enter a binary sequence (e.g., 1100101): ")
    if not all(c in '01' for c in user_input):
        print("Invalid input. Please enter a binary sequence containing only 0s and 1s.")
        return None
    return list(map(int, user_input))


def generate_unipolar_nrz(data, V):
    return [V if bit == 1 else 0 for bit in data]


def generate_unipolar_rz(data, V):
    return [V if bit == 1 else 0 for bit in data for _ in (0, 1)]


def generate_polar_nrz(data, V):
    return [V / 2 if bit == 1 else -V / 2 for bit in data]


def generate_polar_rz(data, V):
    return [V / 2 if bit == 1 else 0 if bit == 0 else -V / 2 for bit in data for _ in (0, 1)]


def generate_bipolar_nrz(data, V):
    bipolar = []
    last_non_zero = -V
    for bit in data:
        if bit == 0:
            bipolar.append(0)
        else:
            last_non_zero = -last_non_zero
            bipolar.append(last_non_zero)
    return bipolar


def generate_manchester(data, V):
    manchester = []
    for bit in data:
        if bit == 0:
            manchester.extend([-V / 2, V / 2])
        else:
            manchester.extend([V / 2, -V / 2])
    return manchester


def plot_signals(data, unipolar_nrz, unipolar_rz, polar_nrz, polar_rz, bipolar_nrz, manchester):
    fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

    time_nrz = np.arange(len(data))
    time_rz = np.arange(0, len(data), 0.5)
    time_manchester = np.arange(0, len(data), 0.5)

    # Unipolar NRZ Plot
    axs[0].step(time_nrz, unipolar_nrz, where='mid', color='blue')
    axs[0].set_title('Unipolar NRZ Encoding (0 to V)')
    axs[0].set_ylabel('Voltage')
    axs[0].grid(True)

    # Unipolar RZ Plot
    axs[1].step(time_rz, unipolar_rz, where='post', color='blue')
    axs[1].set_title('Unipolar RZ Encoding (0 to V)')
    axs[1].set_ylabel('Voltage')
    axs[1].grid(True)

    # Polar NRZ Plot
    axs[2].step(time_nrz, polar_nrz, where='mid', color='green')
    axs[2].set_title('Polar NRZ Encoding (V/2 to -V/2)')
    axs[2].set_ylabel('Voltage')
    axs[2].grid(True)

    # Polar RZ Plot
    axs[3].step(time_rz, polar_rz, where='post', color='green')
    axs[3].set_title('Polar RZ Encoding (V/2 to -V/2)')
    axs[3].set_ylabel('Voltage')
    axs[3].grid(True)

    # Bipolar NRZ Plot
    axs[4].step(time_nrz, bipolar_nrz, where='mid', color='red')
    axs[4].set_title('Bipolar NRZ Encoding (V to -V)')
    axs[4].set_ylabel('Voltage')
    axs[4].grid(True)

    # Manchester Plot
    axs[5].step(time_manchester, manchester, where='post', color='purple')
    axs[5].set_title('Manchester Encoding (Split Phase)')
    axs[5].set_xlabel('Time')
    axs[5].set_ylabel('Voltage')
    axs[5].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = get_user_input()
    if data is not None:
        V = 5  # Define the voltage level
        unipolar_nrz_signal = generate_unipolar_nrz(data, V)
        unipolar_rz_signal = generate_unipolar_rz(data, V)
        polar_nrz_signal = generate_polar_nrz(data, V)
        polar_rz_signal = generate_polar_rz(data, V)
        bipolar_nrz_signal = generate_bipolar_nrz(data, V)
        manchester_signal = generate_manchester(data, V)

        plot_signals(data, unipolar_nrz_signal, unipolar_rz_signal, polar_nrz_signal, polar_rz_signal,
                     bipolar_nrz_signal, manchester_signal)
'''





'''import numpy as np
import matplotlib.pyplot as plt

# Define the data sequence (e.g., a random binary sequence)
data = np.array([1, 0, 1, 1, 0, 1])

# Time parameters
bit_duration = 1  # 1 second per bit
sampling_rate = 100  # Samples per second
t = np.linspace(0, len(data)*bit_duration, len(data)*sampling_rate)

# NRZ Encoding
def nrz(data, sampling_rate):
    signal = np.repeat(data, sampling_rate)
    return signal

# RZ Encoding
def rz(data, sampling_rate):
    signal = []
    for bit in data:
        if bit == 1:
            signal.extend([1] * (sampling_rate//2) + [0] * (sampling_rate//2))
        else:
            signal.extend([0] * sampling_rate)
    return np.array(signal)

# Manchester Encoding
def manchester(data, sampling_rate):
    signal = []
    for bit in data:
        if bit == 1:
            signal.extend([1] * (sampling_rate//2) + [0] * (sampling_rate//2))
        else:
            signal.extend([0] * (sampling_rate//2) + [1] * (sampling_rate//2))
    return np.array(signal)

# Create signals
nrz_signal = nrz(data, sampling_rate)
rz_signal = rz(data, sampling_rate)
manchester_signal = manchester(data, sampling_rate)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 6))

# Plot NRZ
axs[0].step(t, nrz_signal, where='post')
axs[0].set_title("NRZ Encoding")
axs[0].set_ylim([-0.2, 1.2])

# Plot RZ
axs[1].step(t, rz_signal, where='post')
axs[1].set_title("RZ Encoding")
axs[1].set_ylim([-0.2, 1.2])

# Plot Manchester
axs[2].step(t, manchester_signal, where='post')
axs[2].set_title("Manchester Encoding")
axs[2].set_ylim([-0.2, 1.2])

# Set labels
for ax in axs:
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

plt.tight_layout()
plt.show()
'''






'''import numpy as np
from matplotlib import pyplot as plt

# UNIPOLER.................................
# data = np.random.randint(0,2,25)
data = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
time = np.arange(len(data))

plt.subplot(2, 3, 1)
plt.step(time, data, where='post')
plt.title('Unipolar')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)
plt.grid(True)
plt.yticks([-2, -1, 0, 1, 2, 3])
plt.xticks(time)

# NRZ-L.......................................
# data = np.random.randint(0,2,10)
# time = np.arange(len(data))
signal = np.zeros(len(data), dtype=int)

for i in range(len(data)):
    if data[i] == 0:
        signal[i] = -1
    else:
        signal[i] = 1

plt.subplot(2, 3, 2)
plt.step(time, signal, where='post')
plt.title('NRZ-L')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.text(0, 2, data)
plt.grid(True)
plt.yticks([-2, -1, 0, 1, 2, 3])
plt.xticks(time)

# NRZ-I.......................................
# data = np.random.randint(0,2,10)
# time = np.arange(len(data))

signal = np.zeros(len(data), dtype=int)
flg = True

for i in range(len(data)):
    if data[i] == 1:
        flg = not flg
    if flg:
        signal[i] = 1
    else:
        signal[i] = -1

plt.subplot(2, 3, 3)
plt.step(time, signal, where='post')
plt.title('NRZ-I')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)

# plt.text(3, 8)
plt.grid(True)
plt.yticks([-2, -1, 0, 1, 2, 3])
plt.xticks(time)

# RZ...........................................
# data = np.random.randint(0,2,10)
time = np.linspace(0, len(data), len(data) * 2)
signal = np.zeros(2 * len(data), dtype=int)

for i in range(0, 2 * len(data), 2):
    if data[i // 2] == 1:
        signal[i] = 1
    else:
        signal[i] = -1
    signal[i + 1] = 0

plt.subplot(2, 3, 4)
plt.step(time, signal, where='post')
plt.title('RZ')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)

# plt.text(3, 8)
plt.grid(True)
plt.yticks([-2, -1, 0, 1, 2, 3])
plt.xticks(np.arange(len(data)))

# MANCHESTER.........................................
# data = np.random.randint(0,2,10)
time_org = np.arange(len(data))
signal = np.zeros(len(data) * 2, dtype=int)

for i in range(0, len(data) * 2, 2):
    if data[i // 2] == 0:
        signal[i] = 1
        signal[i + 1] = -1
    else:
        signal[i] = -1
        signal[i + 1] = 1

print(signal)
time = np.arange(len(signal))
plt.subplot(2, 3, 5)
plt.step(time, signal, where='post')
plt.title('Manchestor')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)

plt.grid(True)
plt.yticks([-2, -1, 0, 1, 2, 3])
plt.xticks(time_org * 2, time_org)

# DIFFERENTIAL MANCHESTER...............................
# data = np.random.randint(0,2,10)
time_org = np.arange(len(data))
signal = np.zeros(len(data) * 2, dtype=int)

start = False
for i in range(0, len(data) * 2, 2):
    if start:
        if data[i // 2] == 0:
            signal[i] = -1 * signal[i - 1]
            signal[i + 1] = signal[i - 1]
        else:
            signal[i] = signal[i - 1]
            signal[i + 1] = -1 * signal[i - 1]
    else:
        start = True
        if data[i // 2] == 0:
            signal[i] = -1
            signal[i + 1] = 1
        else:
            signal[i] = 1
            signal[i + 1] = -1

print(signal)
time = np.arange(len(signal))
plt.subplot(2, 3, 6)
plt.step(time, signal, where='post')
plt.title('Differential Manchestor')
plt.xlabel('Time')
plt.ylabel('Ammplitude')
plt.text(0, 2, data)

plt.grid(True)
plt.yticks([-2, -1, 0, 1, 2, 3])
plt.xticks(time_org * 2, time_org)
plt.subplots_adjust(hspace=1)
plt.show()'''




'''import numpy as np
import matplotlib.pyplot as plt

# User input for binary sequence
data_input = input("Enter binary sequence (e.g., 1010101): ")
data = np.array([int(bit) for bit in data_input])

# Predefined parameters
bit_duration = 1  # in seconds
sampling_rate = 100  # samples per second

# Time axis
t = np.linspace(0, len(data) * bit_duration, len(data) * sampling_rate)

# Unipolar NRZ encoding
def unipolar_nrz(data, sampling_rate):
    return np.repeat(data, sampling_rate)

# Polar NRZ-L encoding
def polar_nrz_l(data, sampling_rate):
    signal = np.where(data == 1, 1, -1)
    return np.repeat(signal, sampling_rate)

# Polar NRZ-I encoding
def polar_nrz_i(data, sampling_rate):
    signal = np.zeros_like(data)
    signal[0] = 1 if data[0] == 1 else -1  # Start with 1 or -1 based on the first bit
    for i in range(1, len(data)):
        signal[i] = signal[i-1] if data[i] == 0 else -signal[i-1]
    return np.repeat(signal, sampling_rate)

# Bipolar RZ encoding
def bipolar_rz(data, sampling_rate):
    signal = []
    for bit in data:
        if bit == 1:
            signal.extend([1] * (sampling_rate // 2) + [0] * (sampling_rate // 2))
        else:
            signal.extend([0] * sampling_rate)
    return np.array(signal)

# Manchester encoding
def manchester(data, sampling_rate):
    signal = []
    for bit in data:
        if bit == 1:
            signal.extend([1] * (sampling_rate // 2) + [-1] * (sampling_rate // 2))
        else:
            signal.extend([-1] * (sampling_rate // 2) + [1] * (sampling_rate // 2))
    return np.array(signal)

# Differential Manchester encoding
def diff_manchester(data, sampling_rate):
    signal = []
    prev = -1  # Start with a negative transition
    for bit in data:
        if bit == 1:
            signal.extend([prev] * (sampling_rate // 2) + [-prev] * (sampling_rate // 2))
        else:
            signal.extend([-prev] * (sampling_rate // 2) + [prev] * (sampling_rate // 2))
        prev = signal[-1]
    return np.array(signal)

# Polar AMI encoding
def polar_ami(data, sampling_rate):
    signal = []
    prev = 1
    for bit in data:
        if bit == 1:
            prev = -prev
            signal.extend([prev] * sampling_rate)
        else:
            signal.extend([0] * sampling_rate)
    return np.array(signal)

# B8ZS encoding
def b8zs(data, sampling_rate):
    signal = []
    prev = 1
    zero_count = 0
    for bit in data:
        if bit == 1:
            prev = -prev
            signal.extend([prev] * sampling_rate)
            zero_count = 0
        else:
            zero_count += 1
            if zero_count == 8:  # Replace 8 consecutive 0s with B8ZS substitution
                signal[-7 * sampling_rate:] = [-prev, 0, prev, 0, 0, -prev, prev, 0] * (sampling_rate // 8)
                zero_count = 0
            else:
                signal.extend([0] * sampling_rate)
    return np.array(signal)

# Generate signals for each encoding scheme
unipolar_signal = unipolar_nrz(data, sampling_rate)
polar_nrz_l_signal = polar_nrz_l(data, sampling_rate)
polar_nrz_i_signal = polar_nrz_i(data, sampling_rate)
bipolar_rz_signal = bipolar_rz(data, sampling_rate)
manchester_signal = manchester(data, sampling_rate)
diff_manchester_signal = diff_manchester(data, sampling_rate)
polar_ami_signal = polar_ami(data, sampling_rate)
b8zs_signal = b8zs(data, sampling_rate)

# Create subplots
fig, axs = plt.subplots(8, 1, figsize=(10, 16))

# Plot each encoding scheme
axs[0].step(t, unipolar_signal, where='post')
axs[0].set_title('Unipolar NRZ')
axs[0].set_ylim([-1.5, 1.5])

axs[1].step(t, polar_nrz_l_signal, where='post')
axs[1].set_title('Polar NRZ-L')
axs[1].set_ylim([-1.5, 1.5])

axs[2].step(t, polar_nrz_i_signal, where='post')
axs[2].set_title('Polar NRZ-I')
axs[2].set_ylim([-1.5, 1.5])

axs[3].step(t, bipolar_rz_signal, where='post')
axs[3].set_title('Bipolar RZ')
axs[3].set_ylim([-1.5, 1.5])

axs[4].step(t, manchester_signal, where='post')
axs[4].set_title('Manchester')
axs[4].set_ylim([-1.5, 1.5])

axs[5].step(t, diff_manchester_signal, where='post')
axs[5].set_title('Differential Manchester')
axs[5].set_ylim([-1.5, 1.5])

axs[6].step(t, polar_ami_signal, where='post')
axs[6].set_title('Polar AMI')
axs[6].set_ylim([-1.5, 1.5])

axs[7].step(t, b8zs_signal, where='post')
axs[7].set_title('B8ZS')
axs[7].set_ylim([-1.5, 1.5])

# Set labels
for ax in axs:
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

plt.tight_layout()
plt.show()
'''






import matplotlib.pyplot as plt

# Function to plot signals
def plot_signal(signal, title):
    plt.step(range(len(signal)), signal, where='post')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.ylim(-1.5, 1.5)  # Adjust y-axis limits for better visualization
    plt.grid(True)
    #plt.show()

# Unipolar function
def unipolar(signal):
    return [1 if bit == 1 else 0 for bit in signal]

# Polar NRZ-L function
def polar_nrz_l(signal):
    return [1 if bit == 1 else -1 for bit in signal]

# Polar NRZ-R function
def polar_nrz_i(signal):
    polar_signal = []
    last_bit = 1
    flag = 0
    for bit in signal:
        if bit == 1 and flag == 0:
            #polar_signal.append(-1)
            last_bit = 1
            flag = 1
        elif bit == 0 and flag == 0:
            #polar_signal.append(1)
            last_bit = -1
            flag = 1

        if bit == 1 and flag:
            polar_signal.append(-last_bit)
            last_bit = -last_bit
        elif bit == 0 and flag:
            polar_signal.append(last_bit)
    return polar_signal

# RZ function
def rz(signal):
    rz_signal = []
    for bit in signal:
        rz_signal.extend([1, 0] if bit == 1 else [-1, 0])
    return rz_signal

# Manchester encoding function
def manchester(signal):
    manchester_signal = []
    for bit in signal:
        manchester_signal.extend([1, -1] if bit == 1 else [-1, 1])
    return manchester_signal

# Differential Manchester encoding function
def differential_manchester(signal):
    differential_manchester_signal = []
    last_state = -1
    for bit in signal:
        if bit == 0:
            differential_manchester_signal.extend([-last_state, last_state])
        else:
            differential_manchester_signal.extend([last_state, -last_state])
            last_state = -last_state
    return differential_manchester_signal

# AMI encoding function
def ami(signal):
    ami_signal = []
    last_voltage = 1
    for bit in signal:
        if bit == 1:
            ami_signal.append(last_voltage)
            last_voltage *= -1
        else:
            ami_signal.append(0)
    return ami_signal

# B8ZS encoding function
def b8zs(signal):
    b8zs_signal = []
    last_pulse = 1
    zero_count = 0

    for i, bit in enumerate(signal):
        if bit == 1:
            zero_count = 0
            last_pulse *= -1
            b8zs_signal.append(last_pulse)
        else:
            zero_count += 1
            b8zs_signal.append(0)
            if zero_count == 8:
                b8zs_signal[-8:] = [0, 0, 0, last_pulse, -last_pulse, 0, -last_pulse, last_pulse]
                zero_count = 0

    return b8zs_signal


# HDB3 encoding function
def hdb3(signal):
    hdb3_signal = []
    last_pulse = -1
    zero_count = 0
    ones_count = 0

    for bit in signal:
        if bit == 1:
            hdb3_signal.append(last_pulse * -1)
            last_pulse *= -1
            zero_count = 0
            ones_count += 1
        else:
            zero_count += 1
            if zero_count == 4:
                if ones_count % 2 == 0:  # Even number of 1s since last substitution
                    hdb3_signal[-3:] = [last_pulse, 0, 0]
                    hdb3_signal.append(last_pulse)
                else:  # Odd number of 1s since last substitution
                    hdb3_signal[-3:] = [0, 0, 0]
                    hdb3_signal.append(last_pulse * -1)
                last_pulse *= -1
                zero_count = 0
                ones_count = 0
            else:
                hdb3_signal.append(0)

    return hdb3_signal



# Function to get binary input
def get_binary_input():
    binary_string = input("Enter a binary string (e.g., 101010): ")
    binary_list = [int(bit) for bit in binary_string]
    return binary_list

# Get binary input from user
signal = get_binary_input()

# Plotting signals
plt.subplot(3,3,1)
plot_signal(unipolar(signal), 'Unipolar')
plt.subplot(3,3,2)
plot_signal(polar_nrz_l(signal), 'Polar NRZ-L')
plt.subplot(3,3,3)
plot_signal(polar_nrz_i(signal), 'Polar NRZ-I')
plt.subplot(3,3,4)
plot_signal(rz(signal), 'Polar RZ')
plt.subplot(3,3,5)
plot_signal(manchester(signal), 'Polar Biphase Manchester')
plt.subplot(3,3,6)
plot_signal(differential_manchester(signal), 'Polar Biphase Differential Manchester')
plt.subplot(3,3,7)
plot_signal(ami(signal), 'Bipolar AMI')
plt.subplot(3,3,8)
plot_signal(b8zs(signal), 'Bipolar B8ZS')
plt.subplot(3,3,9)
plot_signal(hdb3(signal),'Bipolar HDB3')
plt.tight_layout()
plt.show()