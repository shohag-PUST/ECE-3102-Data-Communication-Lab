import numpy as np
import matplotlib.pyplot as plt


def pam_pcm(frequency, sampling_rate, quantization_levels=16):
    # Continuous time for analog signal
    t_continuous = np.linspace(0, 1, 1000)

    # Generating the analog signal (sine wave)
    analog_signal = np.sin(2 * np.pi * frequency * t_continuous)

    # Sampled time for PAM signal
    t_sampled = np.linspace(0, 1, sampling_rate)

    # Generating the PAM signal by sampling the analog signal
    pam_signal = np.sin(2 * np.pi * frequency * t_sampled)

    # Quantizing the sampled signal for PCM
    pcm_quantized = np.round((pam_signal + 1) * (quantization_levels / 2)).astype(int)

    # Encoding the quantized values into binary format
    pcm_encoded = [bin(value)[2:].zfill(4) for value in pcm_quantized]

    # Convert the binary PCM into a single array of bits
    binary_signal = np.array([int(bit) for bit in ''.join(pcm_encoded)])

    # Binary signal's time base
    t_binary = np.linspace(0, 1, len(binary_signal))

    # Plotting the results
    plt.figure(figsize=(12, 12))

    # Plot the analog signal
    plt.subplot(4, 1, 1)
    plt.plot(t_continuous, analog_signal, label='Analog Signal', color='b')
    plt.title('Analog Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot the PAM signal
    plt.subplot(4, 1, 2)
    plt.stem(t_sampled, pam_signal, label='PAM Signal', basefmt=" ", linefmt='g', markerfmt='go')
    plt.title(f'Pulse Amplitude Modulation (PAM) (Sampling Rate = {sampling_rate} Hz)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot the PCM signal with quantized values
    plt.subplot(4, 1, 3)
    pcm_signal = (pcm_quantized / (quantization_levels - 1)) * 2 - 1
    plt.stem(t_sampled, pcm_signal, label='PCM Signal', basefmt=" ", linefmt='r', markerfmt='ro')

    # Adding binary labels to the PCM signal points
    for i, value in enumerate(pcm_encoded):
        plt.text(t_sampled[i], pcm_signal[i], f'{value}', fontsize=8, ha='center', va='bottom')

    plt.title('PCM Signal with Binary Values')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot the binary signal from PCM encoding
    plt.subplot(4, 1, 4)
    plt.step(t_binary, binary_signal, where='post', label='Binary Signal', color='purple')
    plt.title('Binary Signal from PCM Encoding')
    plt.xlabel('Time')
    plt.ylabel('Binary Value')
    plt.ylim(-0.5, 1.5)
    plt.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Taking user input for frequency and sampling rate
frequency = float(input("Enter the frequency of the analog signal (Hz): "))
sampling_rate = int(input("Enter the sampling rate (samples per second): "))

# Running the pam_pcm function with the given inputs
pam_pcm(frequency, sampling_rate)
