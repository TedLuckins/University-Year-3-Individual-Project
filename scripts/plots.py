# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Plotting functions
def Plot_Attractor(filename, title):
    df = pd.read_csv(filename)
    x = df["X"]
    y = df["Y"]
    z = df["Z"]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, lw=0.6)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    plt.show()

def Plot_Conductance_Functions(filename, title):
    df = pd.read_csv(filename)
    x_V = df["x_V"]
    F0_0 = df["F0_0"]
    F0_1 = df["F0_1"]
    F1_1 = df["F1_1"]
    plt.plot(x_V, F0_0, label="F0_0", lw=0.8)
    plt.plot(x_V, F0_1, label="F0_1", lw=0.8)
    plt.plot(x_V, F1_1, label="F1_1", lw=0.8)
    plt.xlabel("x_V")
    plt.ylabel("Fi_j")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# def Plot_Voltage_Time(filename, title):
#     df = pd.read_csv(filename)
#     time = df["Time"]
#     voltage = df["V"]
#     plt.figure(figsize=(8, 6))
#     plt.plot(time, voltage, label="Voltage (V)", lw=0.8)
#     plt.xlabel("Time")
#     plt.ylabel("Voltage")
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def Plot_Voltage_Time(filename1, filename2, title):
    df = pd.read_csv(filename1)
    time = df["Time"]
    voltage = df["V"]
    plt.figure(figsize=(8, 6))
    plt.plot(time, voltage, label="Voltage (Ascending Vn)", lw=0.8)
    df = pd.read_csv(filename2)
    time = df["Time"]
    voltage = df["V"]
    plt.plot(time, voltage, label="Voltage (Descending Vn)", lw=0.8)
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_hysteresis(filename_up, filename_down):
    df_up = pd.read_csv(filename_up)
    df_down = pd.read_csv(filename_down)

    plt.fill_between(df_up["Vn"], df_up["V_min"], df_up["V_max"], color='blue', alpha=0.3, label="Increasing Vn")
    plt.fill_between(df_down["Vn"], df_down["V_min"], df_down["V_max"], color='red', alpha=0.3, label="Decreasing Vn")

    plt.plot(df_up["Vn"], df_up["V_max"], color='blue', marker="o", linestyle="none", markersize=0.01)
    plt.plot(df_up["Vn"], df_up["V_min"], color='blue', marker="o", linestyle="none", markersize=0.01)
    plt.plot(df_down["Vn"], df_down["V_max"], color='red', marker="s", linestyle="none", markersize=0.01)
    plt.plot(df_down["Vn"], df_down["V_min"], color='red', marker="s", linestyle="none", markersize=0.01)

    plt.xlabel("Vn")
    plt.ylabel("V (Equilibrium)")
    plt.legend()
    plt.title("Hysteresis in V vs. Vn")
    plt.grid()
    plt.show()



def plot_oscillating_points(filename, threshold):
    df = pd.read_csv(filename)

    df["V_amplitude"] = df["V_max"] - df["V_min"]
    oscillating = df[df["V_amplitude"] > threshold]
    
    plt.figure(figsize=(8, 6))
    plt.plot(oscillating["Vn"], oscillating["Rn"], 'ro', markersize=1, label="Oscillating")

    plt.xlabel("$V_n$")
    plt.ylabel("$R_n$")
    plt.title("Oscillating Regions of V")
    plt.legend()
    plt.grid(True)
    plt.show()




# # Calls Plotting functions
# # Plots Lorenz Attractor (Euler) - Wikipedia Example
# Plot_Attractor("Lorenz_Attractor/lorenz_attractor_euler.csv", "Lorenz Attractor (Euler)")
#
# # Plots Lorenz Attractor (Runge-Kutta) - Wikipedia Example
# Plot_Attractor("Lorenz_Attractor/lorenz_attractor_runge-kutta.csv", "Lorenz Attractor (Runge-Kutta)")
#
# # Plots Rossler Attractor (Euler) - Wikipedia Example
# Plot_Attractor("Rossler_Attractor/rossler_attractor_euler.csv", "Rossler Attractor (Euler)")
#
# Plots Rossler Attractor (Runge-Kutta) - Wikipedia Example
# Plot_Attractor("Rossler_Attractor/rossler_attractor_runge_kutta.csv", "Rossler Attractor (Runge-Kutta)")
#
# # Plots Bloch Vector components in 3D space over time
# Plot_Attractor("memristor_simulation3.csv", "Memristor Attractor (Runge-Kutta)")
#
# # Plots the conductance functions for each quantum state over a range of x_V values
# Plot_Conductance_Functions("Conductance_Functions.csv", "Conductance Functions")
#
# Plots Voltage against time
# Plot_Voltage_Time("C:\\Users\\ted_l\\OneDrive - Loughborough University\\Year 3 Individual Project\\Finlay's Paper\\Figure 4 (b)\\Forward, Vn1-5,1000s each.csv", "C:\\Users\\ted_l\\OneDrive - Loughborough University\\Year 3 Individual Project\\Finlay's Paper\\Figure 4 (b)\\Backward, Vn1-5,1000s each.csv", "Voltage vs Time (Memristor Simulation)")

# Plot_Voltage_Time("C:\\Users\\ted_l\\OneDrive - Loughborough University\\Year 3 Individual Project\\memristor_simulation_Vn=0.1-0.5.csv", "C:\\Users\\ted_l\\OneDrive - Loughborough University\\Year 3 Individual Project\\memristor_simulation_Vn=0.5-0.1.csv", "Voltage vs Time (Memristor Simulation)")

# Call the Hysteresis Plotting function
# plot_hysteresis("Memristor/Hysteresis/memristor_increasing_dVn0.1_dt0.01.csv",
#                 "Memristor/Hysteresis/memristor_decreasing_dVn0.1_dt0.01.csv")

# Call the ocillating points plotting function
plot_oscillating_points("memristor_oscillation_regions_Test2.csv", threshold=1e-6)

