import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt

def Calculate_PSD(filename, output_csv, data_fraction):
    df = pd.read_csv(filename)

    num_rows = len(df)
    last_rows = int(num_rows * data_fraction)
    df = df.iloc[-last_rows:]

    time = df["Time"].values
    voltage = df["V"].values

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    f, psd = welch(voltage, fs=fs, nperseg=max(1, len(voltage) // 2))

    omega = 2 * np.pi * f

    max_psd = np.max(psd)
    normalised_psd = psd / max_psd

    data = {
        'Frequency': f,
        'Angular Frequency': omega,
        'Real Power': psd,
        'Normalised Power': normalised_psd
    }
    df_psd = pd.DataFrame(data)
    df_psd.to_csv(output_csv, index=False)

    print(f"PSD data (real and normalised power) has been saved to {output_csv}")



def plot_psd(csv_file, title="Power Spectral Density"):
    df = pd.read_csv(csv_file)

    omega = df["Angular Frequency"].values
    real_psd = df["Real Power"].values

    plt.figure(figsize=(8, 6))
    plt.semilogy(omega, real_psd, label="Real PSD", lw=0.8, color="blue")

    plt.xlabel("Angular Frequency (rad/s)")
    plt.ylabel("Power")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 35)
    plt.show()

def plot_normalised_psd(csv_file, title="Normalised Power Spectral Density"):
    df = pd.read_csv(csv_file)

    omega = df["Angular Frequency"].values
    normalised_psd = df["Normalised Power"].values

    plt.figure(figsize=(8, 6))
    plt.semilogy(omega, normalised_psd, label="Normalised PSD", lw=0.8, color="red")

    plt.xlabel("Angular Frequency")
    plt.ylabel("Normalised Power")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 35)
    plt.show()




Calculate_PSD("Simulation_Test.csv", "PSD_Test.csv", 0.1)

plot_psd("PSD_Test.csv", "Power Spectral Density")

plot_normalised_psd("PSD_Test.csv", "Normalised Power Spectral Density")


