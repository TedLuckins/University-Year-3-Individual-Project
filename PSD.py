import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt


def Windows(signal, window_type):
    if window_type == "hann":
        return np.hanning(len(signal)) * signal
    elif window_type == "hamming":
        return np.hamming(len(signal)) * signal
    else:
        return signal


def Periodogram_PSD(signal, sample_rate):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1 / sample_rate)
    fft_values = np.fft.rfft(signal)

    psd = (np.abs(fft_values) ** 2) / N
    return freqs, psd


def Average_Periodogram(signal, sample_rate, num_averages, window_type):
    N = len(signal)
    psd_sum = np.zeros(N // 2 + 1)

    for _ in range(num_averages):
        signal_no_dc = signal - np.mean(signal)

        windowed_signal = Windows(signal_no_dc, window_type)

        freqs, psd = Periodogram_PSD(windowed_signal, sample_rate)

        psd_sum += psd

    psd_avg = psd_sum / num_averages
    return freqs, psd_avg


def Calculate_PSD(filename, output_csv, data_fraction, num_averages, window_type):
    df = pd.read_csv(filename)
    num_rows = len(df)
    last_rows = int(num_rows * data_fraction)
    df = df.iloc[-last_rows:]

    time = df["Time"].values
    voltage = df["V"].values

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    f, psd_avg = Average_Periodogram(voltage, fs, num_averages, window_type)

    omega = 2 * np.pi * f

    max_psd = np.max(psd_avg)
    normalised_psd = psd_avg / max_psd

    data = {
        'Frequency': f,
        'Angular Frequency': omega,
        'Real Power': psd_avg,
        'Normalised Power': normalised_psd
    }

    df_psd = pd.DataFrame(data)
    df_psd.to_csv(output_csv, index=False)

    print(f"PSD data (real and normalised power) has been saved to {output_csv}")


def plot_normalised_psd(csv_file, title="Normalised Power Spectral Density"):
    df = pd.read_csv(csv_file)

    omega = df["Angular Frequency"].values
    normalised_psd = np.log(df["Normalised Power"].values)

    plt.figure(figsize=(8, 6))
    plt.plot(omega, normalised_psd, label="Normalised PSD", lw=0.8, color="red")

    plt.xlabel("Angular Frequency")
    plt.ylabel("Normalised Power")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 35)
    plt.ylim(-20, 0)
    plt.show()


Calculate_PSD("Simulation_Test_dt=0.0001.csv", "PSD_Test_dt=0.0001_avg=100.csv", data_fraction=1.0, num_averages=10, window_type="Hann")

plot_normalised_psd("PSD_Test_dt=0.0001_avg=100.csv", "Normalised Power Spectral Density")
