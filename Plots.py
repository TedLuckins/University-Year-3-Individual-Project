# Libraries
import pandas as pd
import matplotlib.pyplot as plt

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


#Calls Plotting functions
# Plots Lorenz Attractor (Euler) - Wikipedia Example
Plot_Attractor("lorenz_attractor_euler.csv", "Lorenz Attractor (Euler)")

# Plots Lorenz Attractor (Runge-Kutta) - Wikipedia Example
Plot_Attractor("lorenz_attractor_runge-kutta.csv", "Lorenz Attractor (Runge-Kutta)")

# Plots Rossler Attractor (Euler) - Wikipedia Example
Plot_Attractor("rossler_attractor_euler.csv", "Rossler Attractor (Euler)")

# Plots Rossler Attractor (Runge-Kutta) - Wikipedia Example
Plot_Attractor("rossler_attractor_runge-kutta.csv", "Rossler Attractor (Runge-Kutta)")

# Plots Bloch Vector components in 3D space over time
Plot_Attractor("memristor_simulation.csv", "Memristor Attractor (Runge-Kutta)")

# Plots the conductance functions for each quantum state over a range of x_V values
Plot_Conductance_Functions("Conductance_Functions.csv", "Conductance Functions")
