# Libraries
import numpy as np
import pandas as pd
from numba import njit
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.special import hermite, factorial





# Numerical integration method functions
#@njit
def Euler_Method(Deriv_Func, state, dt, *params):
    return state + Deriv_Func(state, *params) * dt

#@njit
def Runge_Kutta_Method(Deriv_Func, state, dt, *params):
    k1 = Deriv_Func(state, *params)
    k2 = Deriv_Func(state + 0.5 * k1 * dt, *params)
    k3 = Deriv_Func(state +0.5 * k2 * dt, *params)
    k4 = Deriv_Func(state + k3 * dt, *params)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt/6.0)


# System derivation functions
#@njit
def Lorenz_Deriv(state, s, r, b):
    X, Y, Z = state
    X_dot = s * (Y - X)
    Y_dot = r * X - Y - X * Z
    Z_dot = X * Y - b * Z
    return np.array([X_dot, Y_dot, Z_dot])

#@njit
def Rossler_Deriv(state, a, b, c):
    X, Y, Z = state
    X_dot = -Y - Z
    Y_dot = X + a * Y
    Z_dot = b + Z * (X - c)
    return np.array([X_dot, Y_dot, Z_dot])

def Memristor_Deriv(state, V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, precomputed_F):
    X, Y, Z, V = state
    x_V = round(x_0 - l * np.sqrt(2) * V, 3)
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
        # Interpolates
        x_V1 = round(np.floor(x_V / 0.001) * 0.001, 3)
        x_V2 = round(x_V1 + 0.001, 3)
        F_vals1 = precomputed_F[x_V1]
        F_vals2 = precomputed_F[x_V2]
        F0_0 = ((F_vals2[(0, 0)]-F_vals1[(0, 0)])/0.001) * x_V
        F0_1 = ((F_vals2[(0, 1)]-F_vals1[(0, 1)])/0.001) * x_V
        F1_1 = ((F_vals2[(1, 1)]-F_vals1[(1, 1)])/0.001) * x_V



    V_dot = V_n - (1 + R_n * (F0_0 * (1 + Z) / 2 + F0_1 * X + F1_1 * (1 - Z) / 2)) * V
    X_dot = Omega * Y + 2 * V_dot * Z - Gamma * X
    Y_dot = -Omega * X - Gamma * Y
    Z_dot = -2 * V_dot * X - alpha * Gamma * (Z - Z_T)

    return np.array([X_dot, Y_dot, Z_dot, V_dot])

# Memristor Conductance Function Calculations
def Calc_F(i, j, x_V, l, Lambda):
    H_i = hermite(i)
    H_j = hermite(j)

    def integrand(x_tilde):
        H_i_val = H_i(x_tilde / l)
        H_j_val = H_j(x_tilde / l)
        Sech_val = 1 / np.cosh((x_tilde + x_V) / Lambda)
        return np.exp(-x_tilde ** 2 / l ** 2) * H_i_val * H_j_val * Sech_val

    coefficient = 1 / (l * np.sqrt(2 ** (i + j) * factorial(i) * factorial(j) * np.pi))
    integral, _ = quad(integrand, -10.0, 10.0)
    return coefficient * integral

def Precompute_CalcF(x_V_range, l, Lambda):
    precomputed_values = {}
    for x_V in x_V_range:
        precomputed_values[x_V] = {
            (0, 0): Calc_F(0, 0, x_V, l, Lambda),
            (0, 1): Calc_F(0, 1, x_V, l, Lambda),
            (1, 1): Calc_F(1, 1, x_V, l, Lambda),
        }
    return precomputed_values

def Load_F_Values(filename):
    df = pd.read_csv(filename)
    precomputed_F = {}

    for _, row in df.iterrows():
        #Index = int((row["x_V"] + 3)/0.001)
        precomputed_F[row["x_V"]] = {
            (0, 0): row["F0_0"],
            (0, 1): row["F0_1"],
            (1, 1): row["F1_1"],
        }
    return precomputed_F

# Simulates a System across time
def Simulate_System(Deriv_Func, Step_Func, Initial_Conditions, params, dt, num_steps, out_steps):
    n = len(Initial_Conditions)
    state = np.array(Initial_Conditions)
    results = np.empty((out_steps + 1, n + 1))
    interval = num_steps // out_steps
    results[0, 0] = 0.0
    results[0, 1:] = state
    for i in range(1, num_steps + 1):
        state = Step_Func(Deriv_Func, state, dt, *params)
        if i % interval == 0 or i == num_steps:
            results[i // interval, 0] = i * dt
            results[i // interval, 1:] = state
        print(f"step {i} is completed")
    return results


# Saves outputs from simulated systems
def Save_Results(results, filename, system):
    if system == "Lorenz":
        columns = ["Time", "X", "Y", "Z"]
    elif system == "Rossler":
        columns = ["Time", "X", "Y", "Z"]
    elif system == "Memristor":
        columns = ["Time", "X", "Y", "Z", "V"]
    elif system == "Conductance Functions":
        columns = ["x_V", "F0_0", "F0_1", "F1_1"]
    else:
        columns = ["Time"] + [f"State{i + 1}" for i in range(results.shape[1] - 1)]
    df = pd.DataFrame(results, columns = columns)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Finds the maximum dt value for a specified results accuracy
def Op_dt(Deriv_Func, Step_Func, Initial_Conditions, params, max_dt, num_steps, accuracy):
    dt = max_dt
    state = np.array(Initial_Conditions)
    current_result = state.copy()
    for _ in range(num_steps):
        current_result = Step_Func(Deriv_Func, current_result, dt, *params)
    while dt > 1e-12:
        small_dt = dt * 0.1
        small_result = state.copy()
        for _ in range(int(num_steps * max_dt/small_dt)):
            small_result = Step_Func(Deriv_Func, small_result, small_dt, *params)
        calc_accuracy = np.linalg.norm(current_result - small_result)
        if calc_accuracy < accuracy:
            print(f"Optimal dt: {dt}")
            return dt
        else:
            current_result = small_result
            dt - small_dt
    print("Optimal dt was not found within limits")
    return dt


# Dictionary
Calculations = {
    "Euler": Euler_Method,
    "Runge-Kutta": Runge_Kutta_Method,
    "Lorenz": Lorenz_Deriv,
    "Rossler": Rossler_Deriv,
    "Memristor": Memristor_Deriv
}
def has_reached_equilibrium(V_values, threshold=1e-7):
    last_10_percent = V_values[int(len(V_values) * 0.99):]
    if (np.max(last_10_percent) - np.min(last_10_percent)) < threshold:
        return True
    else:
        return False


def simulate_until_equilibrium(V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0, initial_conditions, max_steps, dt, precomputed_F):
    steps_per_check = 1000
    state = np.array(initial_conditions)
    results = []

    for step in range(1, max_steps + 1):
        state = Runge_Kutta_Method(Memristor_Deriv, state, dt, V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0,
                                   precomputed_F)
        results.append([step * dt] + list(state))

        if step % steps_per_check == 0:
            results_array = np.array(results)
            V_values = results_array[:, -1]
            if has_reached_equilibrium(V_values, threshold=1e-7):
                break

    return np.array(results), state


def get_equilibrium_V(results):
    V_values = results[:, -1]
    last_10_percent = V_values[int(len(V_values) * 0.99):]

    if (np.max(last_10_percent) - np.min(last_10_percent)) < 1e-8:
        return last_10_percent[-1], last_10_percent[-1]
    else:
        return np.min(last_10_percent), np.max(last_10_percent)


def save_results_to_csv(Vn_values, V_results_min, V_results_max, filename):
    df = pd.DataFrame({
        "Vn": Vn_values,
        "V_min": V_results_min,
        "V_max": V_results_max
    })
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def save_results_to_csv(Vn_values, Rn_values, V_results_min, V_results_max, filename):
    df = pd.DataFrame({
        "Vn": Vn_values,
        "Rn": Rn_values,
        "V_min": V_results_min,
        "V_max": V_results_max
    })
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
