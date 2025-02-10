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
    x_V = x_0 - l * np.sqrt(2) * V
    x_V_nearest = min(precomputed_F.keys(), key=lambda k: abs(k - x_V))
    F_vals = precomputed_F[x_V_nearest]

    F0_0 = F_vals[(0, 0)]
    F0_1 = F_vals[(0, 1)]
    F1_1 = F_vals[(1, 1)]

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


# Checks if the system has reached stable point
def has_reached_equilibrium(V_values, threshold=1e-4, oscillation_check=False):
    last_10_percent = V_values[int(len(V_values) * 0.9):]
    max_change = np.max(np.abs(np.diff(last_10_percent)))

    if not oscillation_check:
        return max_change < threshold
    else:
        return len(set(np.round(last_10_percent, 4))) > 3


def simulate_until_equilibrium(V_n, initial_conditions, max_steps=1000000, dt=0.001):
    steps_per_check = 10000
    state = np.array(initial_conditions)
    results = []

    for step in range(1, max_steps + 1):
        state = Runge_Kutta_Method(Memristor_Deriv, state, dt, V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0,
                                   precomputed_F)
        results.append([step * dt] + list(state))

        if step % steps_per_check == 0:
            results_array = np.array(results)
            V_values = results_array[:, -1]

            if has_reached_equilibrium(V_values, oscillation_check=True):
                break

    return np.array(results), state


def get_equilibrium_V(results):
    V_values = results[:, -1]
    last_10_percent = V_values[int(len(V_values) * 0.9):]

    if np.std(last_10_percent) < 1e-7:
        return last_10_percent[-1], last_10_percent[-1]
    else:
        return np.min(last_10_percent), np.max(last_10_percent)

# Dictionary
Calculations = {
    "Euler": Euler_Method,
    "Runge-Kutta": Runge_Kutta_Method,
    "Lorenz": Lorenz_Deriv,
    "Rossler": Rossler_Deriv,
    "Memristor": Memristor_Deriv
}

# # Call Simulation and Save Functions

# # Calls dt optimisation function
# optimal_dt = Op_dt(
#     Calculations["Runge-Kutta"],
#     Calculations["Lorenz"],
#     [0., 1., 1.05],
#     (10, 28, 2.667),
#     max_dt=0.01,
#     num_steps=1000,
#     accuracy=1e-4
# )

# # Simulates and saves Lorenz attractor with Euler method
# results = Simulate_System(
#     Calculations["Lorenz"],
#     Calculations["Euler"],
#     [0., 1., 1.05],
#     (10, 28, 2.667),
#     dt=0.0001,
#     num_steps=1000000,
#     out_steps=10000
# )
# Save_Results(results, "lorenz_attractor_euler.csv", "Lorenz")
#
# # Simulates and saves Lorenz attractor with Runge-Kutta method
# results = Simulate_System(
#     Calculations["Lorenz"],
#     Calculations["Runge-Kutta"],
#     [0., 1., 1.05],
#     (10, 28, 2.667),
#     dt=0.0001,
#     num_steps=1000000,
#     out_steps=10000
# )
# Save_Results(results,"lorenz_attractor_runge-kutta.csv", "Lorenz")

# # Simulates and saves Rossler attractor with Euler method
# results = Simulate_System(
#     Calculations["Rossler"],
#     Calculations["Euler"],
#     [0., 1., 1.05],
#     (0.1, 0.1, 14),
#     dt=0.01,
#     num_steps=10000,
#     out_steps=10000
# )
# Save_Results(results,"rossler_attractor_euler.csv","Rossler")
#
# # Simulates and saves Rossler attractor with Rung-Kutta method
# results = Simulate_System(
#     Calculations["Rossler"],
#     Calculations["Runge-Kutta"],
#     [0., 1., 1.05],
#     (0.1, 0.1, 14),
#     dt=0.01,
#     num_steps=10000,
#     out_steps=10000
# )
# Save_Results(results,"rossler_attractor_runge_kutta.csv","Rossler")

# Parameters for Memristor System
alpha = 1.0
Omega = 7.0
Gamma = 0.1
Z_T = 1.0
Lambda = 0.13
l = 0.5
x_0 = 0.8
V_n = 1.0
R_n = 5.0

# # Precompute values
# x_V_range = np.arange(-3, 3.01, 0.01)
# precomputed_F = Precompute_CalcF(x_V_range, l, Lambda)

# Forward through Vn
#Initial_Conditions_Memristor = [0, 0, 1.0, 0] # Vn = 1.0
#Initial_Conditions_Memristor = [0.422911721, -0.02626182, 0.230480932, 0.479448137] # Vn = 2.0
#Initial_Conditions_Memristor = [0.174542556, 0.326727203, 0.173154776, 0.827412143] # Vn = 3.0
#Initial_Conditions_Memristor = [-0.424124651, 0.190172592, 0.690308487, 1.073813632] # Vn = 4.0
#Initial_Conditions_Memristor = [9.06E-16, -6.34E-14, 0.999999999999444, 3.999733589] # Forward: Vn = 5.0

# Backward through Vn
#Initial_Conditions_Memristor = [3.80570226763632E-16, -2.66399158734637E-14, 0.999999999999444, 4.99999858258624] # Forward: Vn = 5.0
#Initial_Conditions_Memristor = [3.84194705113673E-16, -2.68936293579729E-14, 0.999999999999444, 4.99999858258624] # Vn = 4.0
#Initial_Conditions_Memristor = [2.17468701007642E-16, -1.52228090705507E-14, 0.999999999999444, 3.99973358904326] # Vn = 3.0
#Initial_Conditions_Memristor = [9.05743531569221E-16, -6.34072284232692E-14, 0.999999999999444, 2.95258673229717] # Vn = 2.0
#Initial_Conditions_Memristor = [0.323472997577606, 0.182707906350382, 0.171117250812354, 0.844043980921247] # Forward: Vn = 1.0



Initial_Conditions_Memristor = [0, 0, 1.0, 0]

# Simulates and saves memristor system with Runge-Kutta method
# results_memristor = Simulate_System(
#     Step_Func=Runge_Kutta_Method,
#     Deriv_Func=lambda state, *params: Memristor_Deriv(state, *params, precomputed_F),
#     Initial_Conditions=Initial_Conditions_Memristor,
#     params=(V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0),
#     dt=0.001,
#     num_steps=1000000,
#     out_steps=100000
# )
# Save_Results(results_memristor, "memristor_simulation_B_Vn=1.csv", "Memristor")


# # Conductance Function Simulation for each quantum state
# x_V_Range = np.arange(-3, 3.01, 0.01)
# Lambda = 0.13
# l = 0.5
# results_conductance_function = []
# for x_V in x_V_Range:
#     F0_0 = Calc_F(0, 0, x_V = x_V, l = l, Lambda = Lambda)
#     F0_1 = Calc_F(0, 1, x_V = x_V, l = l, Lambda = Lambda)
#     F1_1 = Calc_F(1, 1, x_V = x_V, l = l, Lambda = Lambda)
#     results_conductance_function.append([x_V, F0_0, F0_1, F1_1])
#
# results_conductance_function = np.array(results_conductance_function)
# Save_Results(results_conductance_function, "Conductance_Functions.csv", "Conductance Functions")




# Save for memristor equalibrium V
def save_results_to_csv(Vn_values, V_results_min, V_results_max, filename):
    df = pd.DataFrame({
        "Vn": Vn_values,
        "V_min": V_results_min,
        "V_max": V_results_max
    })
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")



Vn_values_up = np.arange(0.01, 5.01, 0.01)
Vn_values_down = np.arange(5, 0.00, -0.01)

V_results_up_min = []
V_results_up_max = []
V_results_down_min = []
V_results_down_max = []


current_initial_conditions = Initial_Conditions_Memristor.copy()

# Forward simulation
for V_n in Vn_values_up:
    results, current_initial_conditions = simulate_until_equilibrium(V_n, current_initial_conditions)
    V_min, V_max = get_equilibrium_V(results)
    V_results_up_min.append(V_min)
    V_results_up_max.append(V_max)

save_results_to_csv(Vn_values_up, V_results_up_min, V_results_up_max, "memristor_increasing_dVn0.01_dt0.001.csv")


current_initial_conditions = Initial_Conditions_Memristor.copy()

# Backward simulation
for V_n in Vn_values_down:
    results, current_initial_conditions = simulate_until_equilibrium(V_n, current_initial_conditions)
    V_min, V_max = get_equilibrium_V(results)
    V_results_down_min.append(V_min)
    V_results_down_max.append(V_max)

save_results_to_csv(Vn_values_down, V_results_down_min, V_results_down_max, "memristor_decreasing_dVn0.01_dt0.001.csv")
