import numpy as np
import time

from Calculations import Load_F_Values, Save_Results, has_reached_equilibrium

start_time = time.time()

def Runge_Kutta_Method(Deriv_Func, t, state, dt, *params):
    k1 = Deriv_Func(t, state, *params)
    k2 = Deriv_Func(t + 0.5 * dt, state + 0.5 * k1 * dt, *params)
    k3 = Deriv_Func(t + 0.5 * dt, state + 0.5 * k2 * dt, *params)
    k4 = Deriv_Func(t + dt, state + k3 * dt, *params)

    new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0)
    return new_state

def Memristor_Deriv(t, state, V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, precomputed_F):
    X, Y, Z, V = state
    x_V = x_0 - l * np.sqrt(2) * V

    if x_V < -3:
        F_vals = precomputed_F[-3]
        F0_0 = F_vals[(0, 0)]
        F0_1 = F_vals[(0, 1)]
        F1_1 = F_vals[(1, 1)]
    elif x_V > 3:
        F_vals = precomputed_F[3]
        F0_0 = F_vals[(0, 0)]
        F0_1 = F_vals[(0, 1)]
        F1_1 = F_vals[(1, 1)]
    else:
        x_V1 = round(np.floor(x_V / 0.001) * 0.001, 3)
        x_V2 = round(x_V1 + 0.001, 3)
        F_vals1 = precomputed_F[x_V1]
        F_vals2 = precomputed_F[x_V2]
        F0_0 = F_vals1[(0, 0)] + ((F_vals2[(0, 0)] - F_vals1[(0, 0)]) / 0.001) * (x_V - x_V1)
        F0_1 = F_vals1[(0, 1)] + ((F_vals2[(0, 1)] - F_vals1[(0, 1)]) / 0.001) * (x_V - x_V1)
        F1_1 = F_vals1[(1, 1)] + ((F_vals2[(1, 1)] - F_vals1[(1, 1)]) / 0.001) * (x_V - x_V1)

    Time = time.time() - start_time
    print(f"Time {Time}")

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




# Parameters for Memristor System
alpha = 1.0
Omega = 7.0
Gamma = 0.1
Z_T = 1.0
Lambda = 0.13
l = 0.5
x_0 = 0.8
R_n = 5.0
Vn = 0.23
V_amp = 0.01
f = 6

def V_n(t):
    return V_amp * np.sin(f * t) + 0.23

# Initial_Conditions= [0, 0, 1, 0]
Initial_Conditions= [0.4900079343682886,-0.08648602168376343,0.5873657793316545,0.6513976208268857]


dt = 0.001
max_steps = 1000000

# Precompute values
precomputed_F = Load_F_Values("Conductance_Functions.csv")


results, current_initial_conditions = simulate_until_equilibrium(V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, Initial_Conditions, max_steps, dt, precomputed_F)

Save_Results(results, "AC_Simulation_Test_dt=0.001_t=1000.csv", "Memristor")




end_time = time.time()
total_time = end_time - start_time
print(f"Simulation completed in {total_time:.2f} seconds.")
