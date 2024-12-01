import pandas as pd
import matplotlib.pyplot as plt

# Plots X, Y, Z in 3D
def Plot_Attractor_From_CSV(filename, title):
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


# # Plot Lorenz Attractor (Euler)
# Plot_Attractor_From_CSV("lorenz_attractor_euler.csv", "Lorenz Attractor (Euler)")
#
# # Plot Lorenz Attractor (Runge-Kutta)
# Plot_Attractor_From_CSV("lorenz_attractor_runge-kutta.csv", "Lorenz Attractor (Runge-Kutta)")
#
# # Plot Rossler Attractor (Euler)
# Plot_Attractor_From_CSV("rossler_attractor_euler.csv", "Rossler Attractor (Euler)")
#
# # Plot Rossler Attractor (Runge-Kutta)
# Plot_Attractor_From_CSV("rossler_attractor_runge-kutta.csv", "Rossler Attractor (Runge-Kutta)")

# # Plot Memristor Attractor
Plot_Attractor_From_CSV("memristor_simulation.csv", "Memristor Attractor (Runge-Kutta)")
