# Libraries
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from Calculations import Load_F_Values, has_reached_equilibrium
from Memristor_PSD import  Average_Periodogram

start_time = time.time()

def detect_frequency_locking(voltage, dt, drive_omega, tolerance=0.01, num_averages=1000, window_type="hann"):
    fs = 1.0 / dt
    num_points = int(1000 / dt)
    voltage_to_analyse = voltage[-num_points:] if len(voltage) >= num_points else voltage
    freqs, psd = Average_Periodogram(voltage_to_analyse, fs, num_averages, window_type)
    omega = 2 * np.pi * freqs
    dominant_index = np.argmax(psd)
    dominant_omega = omega[dominant_index]
    is_locked = np.abs(dominant_omega - drive_omega) < tolerance
    return is_locked, dominant_omega

def Runge_Kutta_Method(Deriv_Func, t, state, dt, *params):
    k1 = Deriv_Func(t, state, *params)
    k2 = Deriv_Func(t + 0.5 * dt, state + 0.5 * k1 * dt, *params)
    k3 = Deriv_Func(t + 0.5 * dt, state + 0.5 * k2 * dt, *params)
    k4 = Deriv_Func(t + dt, state + k3 * dt, *params)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0)

def Memristor_Deriv(t, state, V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, precomputed_F):
    X, Y, Z, V = state
    x_V = x_0 - l * np.sqrt(2) * V

    if x_V < -3:
        F_vals = precomputed_F[-3]
    elif x_V > 3:
        F_vals = precomputed_F[3]
    else:
        x_V1 = round(np.floor(x_V / 0.001) * 0.001, 3)
        x_V2 = round(x_V1 + 0.001, 3)
        F_vals1 = precomputed_F[x_V1]
        F_vals2 = precomputed_F[x_V2]
        F_vals = {
            (0, 0): F_vals1[(0, 0)] + ((F_vals2[(0, 0)] - F_vals1[(0, 0)]) / 0.001) * (x_V - x_V1),
            (0, 1): F_vals1[(0, 1)] + ((F_vals2[(0, 1)] - F_vals1[(0, 1)]) / 0.001) * (x_V - x_V1),
            (1, 1): F_vals1[(1, 1)] + ((F_vals2[(1, 1)] - F_vals1[(1, 1)]) / 0.001) * (x_V - x_V1)
        }

    F0_0, F0_1, F1_1 = F_vals[(0, 0)], F_vals[(0, 1)], F_vals[(1, 1)]
    V_dot = V_n - (1 + R_n * (F0_0 * (1 + Z) / 2 + F0_1 * X + F1_1 * (1 - Z) / 2)) * V
    X_dot = Omega * Y + 2 * V_dot * Z - Gamma * X
    Y_dot = -Omega * X - Gamma * Y
    Z_dot = -2 * V_dot * X - alpha * Gamma * (Z - Z_T)

    return np.array([X_dot, Y_dot, Z_dot, V_dot])

def simulate_until_equilibrium(V_n_func, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, initial_conditions, max_steps, dt, precomputed_F):
    steps_per_check = 1000
    state = np.array(initial_conditions)
    results = []

    for step in range(1, max_steps + 1):
        t = step * dt
        V_n = V_n_func(t)
        state = Runge_Kutta_Method(Memristor_Deriv, t, state, dt, V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, precomputed_F)
        results.append([t] + list(state))

        if step % steps_per_check == 0:
            results_array = np.array(results)
            V_values = results_array[:, -1]
            if has_reached_equilibrium(V_values, threshold=1e-7):
                break

    return np.array(results), state

def save_arnold_tongue_to_csv(V_amp_values, freq_values, lock_statuses, filename):
    df = pd.DataFrame({
        "V_amp": V_amp_values,
        "Frequency": freq_values,
        "Is_Locked": lock_statuses
    })
    df.to_csv(filename, index=False)
    print(f"Arnold Tongue results saved to {filename}")

def sweep_synchronisation(Vn_base, omega_range, Vamp_range, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, initial_conditions, max_steps, dt, precomputed_F, filename, early_stopping=True):
    V_amps = []
    omegas = []
    lock_statuses = []

    for omega_drive in omega_range:
        found_lock = False
        for V_amp in Vamp_range:
            def V_n(t):
                return V_amp * np.sin(omega_drive * t) + Vn_base

            sim_data, _ = simulate_until_equilibrium(V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, initial_conditions, max_steps, dt, precomputed_F)

            voltage_signal = sim_data[:, 4]
            is_locked, dominant_omega = detect_frequency_locking(voltage_signal, dt, omega_drive)

            V_amps.append(V_amp)
            omegas.append(omega_drive)
            lock_statuses.append(int(is_locked))

            Time = time.time() - start_time
            print(f"ω_drive={omega_drive:.4f}, V_amp={V_amp:.4f} -> Locked={is_locked}, Dominant ω={dominant_omega:.4f}, time={Time:.2f}s")


            if early_stopping and is_locked:
                print(f"Early stopping triggered for ω_drive={omega_drive:.2f}")
                break

    save_arnold_tongue_to_csv(V_amps, omegas, lock_statuses, filename)

def plot_arnold_tongue(filename):
    data = pd.read_csv(filename)
    V_amps = data["V_amp"].values
    freqs = data["Frequency"].values
    locked = data["Is_Locked"].values

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(freqs, V_amps, c=locked, cmap='coolwarm', s=50)
    plt.xlabel('Driving Angular Frequency ')
    plt.ylabel('V_amp')
    plt.title('Arnold Tongue Diagram (Frequency Locking)')
    plt.colorbar(sc, label='Lock Status (1=Locked, 0=Unlocked)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

sweep_synchronisation(
    Vn_base=0.23,
    omega_range=np.arange(6.9, 7.15, 0.01),
    Vamp_range=np.arange(0.0001, 0.0151, 0.0002),
    R_n=5.0,
    Z_T=1.0,
    alpha=1.0,
    Omega=7.0,
    Gamma=0.1,
    l=0.5,
    Lambda=0.13,
    x_0=0.8,
    # initial_conditions=[0, 0, 1.0, 0],
    initial_conditions= [0.12396569638120777,0.043274266345808055,0.9823908299140194,0.15596345284021695],
    max_steps=100000,
    dt=0.01,
    precomputed_F=Load_F_Values("Conductance_Functions.csv"),
    filename="Arnold_Tongue_Data_Test_6.9-7.15_dt=0.01_no_adapt.csv"
)

plot_arnold_tongue("Arnold_Tongue_Data_Test_6.9-7.15_dt=0.01_no_adapt.csv")

end_time = time.time()
total_time = end_time - start_time
print(f"Simulation completed in {total_time:.2f} seconds.")
