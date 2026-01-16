import numpy as np
import pandas as pd
import time

start_time = time.time()

from Calculations import Load_F_Values, Save_Results, Runge_Kutta_Method, Memristor_Deriv, has_reached_equilibrium, simulate_until_equilibrium

# Parameters for Memristor System
alpha = 1.0
Omega = 7.0
Gamma = 0.1
Z_T = 1.0
Lambda = 0.13
l = 0.5
x_0 = 0.8
R_n = 5.0
V_n = 0.23

#Initial_Conditions= [0, 0, 1, 0]
Initial_Conditions= [0.12396569638120777,0.043274266345808055,0.9823908299140194,0.15596345284021695]
#Initial_Conditions= [-0.010710836167613596,9.529732000075222e-05,0.999913142660738,-0.005220531236939342] #f=7
#Initial_Conditions= [-0.001453856, -4.93E-06, 0.999998411, -0.000726388] #f=50

dt = 0.0001
max_steps = 10000000

# Precompute values
precomputed_F = Load_F_Values("Conductance_Functions.csv")


results, current_initial_conditions = simulate_until_equilibrium(V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, Initial_Conditions, max_steps, dt, precomputed_F)

Save_Results(results, "Simulation_Test_dt=0.0001_t=1000.csv", "Memristor")




end_time = time.time()
total_time = end_time - start_time
print(f"Simulation completed in {total_time:.2f} seconds.")
