import numpy as np
import matplotlib.pyplot as plt

N = 100
fc = 5
fs = 100
symbol_rate = 2
t = np.arange(0, N / symbol_rate, 1 / fs)

data_bits = np.random.randint(0, 2, N * 2)

symbols = np.reshape(data_bits, (-1, 2))

phase_map = {
    (0, 0): 0,  # 0°
    (0, 1): np.pi / 2,  # 90°
    (1, 1): np.pi,  # 180°
    (1, 0): 3 * np.pi / 2  # 270°
}

modulated_signal = np.zeros(len(t))

for i, symbol in enumerate(symbols):
    phase = phase_map[tuple(symbol)]
    modulated_signal[i * fs // symbol_rate: (i + 1) * fs // symbol_rate] = np.cos(
        2 * np.pi * fc * t[i * fs // symbol_rate: (i + 1) * fs // symbol_rate] + phase)

plt.figure(figsize=(10, 6))
plt.plot(t[:500], modulated_signal[:500])  # Plot first 500 samples
plt.title("4-PSK Modulated Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

I_signal = np.cos(2 * np.pi * fc * t[:N])
Q_signal = np.sin(2 * np.pi * fc * t[:N])

plt.figure(figsize=(6, 6))
plt.scatter(I_signal, Q_signal, color='blue')
plt.title("4-PSK Constellation Diagram")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.show()

demodulated_bits = []
for i in range(N):
    received_phase = np.angle(modulated_signal[i * fs // symbol_rate: (i + 1) * fs // symbol_rate].mean())

    if 0 <= received_phase < np.pi / 2:
        demodulated_bits.extend([0, 0])
    elif np.pi / 2 <= received_phase < np.pi:
        demodulated_bits.extend([0, 1])
    elif np.pi <= received_phase < 3 * np.pi / 2:
        demodulated_bits.extend([1, 1])
    else:
        demodulated_bits.extend([1, 0])

print("Original Bits:   ", data_bits[:20])
print("Demodulated Bits:", demodulated_bits[:20])
