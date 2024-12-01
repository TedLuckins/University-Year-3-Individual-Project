# Libraries
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.special import hermite, factorial


# Numerical integration method functions
def Euler_Method(Deriv_Func, state, dt, *params):
    return state + Deriv_Func(state, *params) * dt

def Runge_Kutta_Method(Deriv_Func, state, dt, *params):
    k1 = Deriv_Func(state, *params)
    k2 = Deriv_Func(state + 0.5 * k1 * dt, *params)
    k3 = Deriv_Func(state +0.5 * k2 * dt, *params)
    k4 = Deriv_Func(state + k3 * dt, *params)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt/6.0)


# System derivation functions
def Lorenz_Deriv(state, s, r, b):
    X, Y, Z = state
    X_dot = s * (Y - X)
    Y_dot = r * X - Y - X * Z
    Z_dot = X * Y - b * Z
    return np.array([X_dot, Y_dot, Z_dot])

def Rossler_Deriv(state, a, b, c):
    X, Y, Z = state
    X_dot = -Y - Z
    Y_dot = X + a * Y
    Z_dot = b + Z * (X - c)
    return np.array([X_dot, Y_dot, Z_dot])


def Memristor_Deriv(state, V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0):
    X, Y, Z, V = state
    x_V = x_0 - l * np.sqrt(2) * V
    F0_0 = Calc_F(0, 0, x_V, l, Lambda)
    F0_1 = Calc_F(0, 1, x_V, l, Lambda)
    F1_1 = Calc_F(1, 1, x_V, l, Lambda)
    V_dot = V_n - (1 + R_n * (F0_0 * (1 + Z)/2 + F0_1 * X + F1_1 * (1-Z)/2)) * V
    X_dot = Omega * Y + 2 * V_dot * Z - Gamma * X
    Y_dot = -Omega * X - Gamma * Y
    Z_dot = -2 * V_dot * X - alpha * Gamma * (Z - Z_T)
    return np.array([X_dot, Y_dot, Z_dot, V_dot])


def Calc_F(i, j, x_V, l, Lambda):
    H_i = hermite(i)
    H_j = hermite(j)
    def integrand(x_tilde):
        H_i_val = H_i(x_tilde/l)
        H_j_val = H_j(x_tilde/l)
        Sech_val = 1/np.cosh((x_tilde + x_V)/Lambda)
        return np.exp(-x_tilde**2/l**2) * H_i_val * H_j_val * Sech_val
    coefficient = 1/(l*np.sqrt(2**(i + j) * factorial(i) * factorial(j) * np.pi))
    integral, _ = quad(integrand, -np.inf, np.inf)
    return coefficient * integral

# Simulates a System across time
def Simulate_System(Deriv_Func, Step_Func, Initial_Conditions, params, dt, num_steps, out_steps):
    n = len(Initial_Conditions)
    state = np.array(Initial_Conditions)
    results = np.empty((out_steps + 1, n + 1))
    interval = num_steps // out_steps
    results[0,0] = 0.0
    results[0,1:] = state
    for i in range(1, num_steps + 1):
        state = Step_Func(Deriv_Func, state, dt, *params)
        if i% interval == 0 or i == num_steps:
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

# Dictionary
Calculations = {
    "Euler": Euler_Method,
    "Runge-Kutta": Runge_Kutta_Method,
    "Lorenz": Lorenz_Deriv,
    "Rossler": Rossler_Deriv,
    "Memristor": Memristor_Deriv
}

# Call Functions

# optimal_dt = Op_dt(
#     Calculations["Runge-Kutta"],
#     Calculations["Lorenz"],
#     [0., 1., 1.05],
#     (10, 28, 2.667),
#     max_dt=0.01,
#     num_steps=1000,
#     accuracy=1e-4
# )


results = Simulate_System(
    Calculations["Lorenz"],
    Calculations["Euler"],
    [0., 1., 1.05],
    (10, 28, 2.667),
    dt=0.0001,
    num_steps=1000000,
    out_steps=10000
)
Save_Results(results, "lorenz_attractor_euler.csv", "Lorenz")

results = Simulate_System(
    Calculations["Lorenz"],
    Calculations["Runge-Kutta"],
    [0., 1., 1.05],
    (10, 28, 2.667),
    dt=0.0001,
    num_steps=1000000,
    out_steps=10000
)
Save_Results(results,"lorenz_attractor_runge-kutta.csv", "Lorenz")

results = Simulate_System(
    Calculations["Rossler"],
    Calculations["Euler"],
    [0., 1., 1.05],
    (0.1, 0.1, 14),
    dt=0.01,
    num_steps=10000,
    out_steps=10000
)
Save_Results(results,"rossler_attractor_euler.csv","Rossler")

results = Simulate_System(
    Calculations["Rossler"],
    Calculations["Runge-Kutta"],
    [0., 1., 1.05],
    (0.1, 0.1, 14),
    dt=0.01,
    num_steps=10000,
    out_steps=10000
)
Save_Results(results,"rossler_attractor_runge_kutta.csv","Rossler")

# Parameters for Memristor System
alpha = 1.0
Omega = 7.0
Gamma = 0.1
Z_T = 1.0
Lambda = 0.13
l = 0.5
x_0 = 0.8
V_n = 0.23
R_n = 5.0
Initial_Conditions_Memristor = [0.0, 0.0, 0.0, 0.1]

results_memristor = Simulate_System(
    Step_Func=Runge_Kutta_Method,
    Deriv_Func=Memristor_Deriv,
    Initial_Conditions=Initial_Conditions_Memristor,
    params=(V_n, R_n, Z_T, alpha, Omega, Gamma, l, Lambda, x_0),
    dt=0.01,
    num_steps=10000,
    out_steps=1000
)
Save_Results(results_memristor, "memristor_simulation.csv", "Memristor")


# Conductance Function Simulation for each quantum state
x_V_Range = np.arange(-3, 3.01, 0.01)
Lambda = 0.13
l = 0.5
results_conductance_function = []
for x_V in x_V_Range:
    F0_0 = Calc_F(0, 0, x_V = x_V, l = l, Lambda = Lambda)
    F0_1 = Calc_F(0, 1, x_V = x_V, l = l, Lambda = Lambda)
    F1_1 = Calc_F(1, 1, x_V = x_V, l = l, Lambda = Lambda)
    results_conductance_function.append([x_V, F0_0, F0_1, F1_1])

results_conductance_function = np.array(results_conductance_function)
Save_Results(results_conductance_function, "Conductance_Functions.csv", "Conductance Functions")
