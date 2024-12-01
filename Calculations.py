import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import hermite, factorial


def Euler_Method(Deriv_Func, state, dt, *params):
    return state + Deriv_Func(state, *params) * dt


def Runge_Kutta_Method(Deriv_Func, state, dt, *params):
    k1 = Deriv_Func(state, *params)
    k2 = Deriv_Func(state + 0.5 * k1 * dt, *params)
    k3 = Deriv_Func(state + 0.5 * k2 * dt, *params)
    k4 = Deriv_Func(state + k3 * dt, *params)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0)


def Lorenz_Derivatives(state, s, r, b):
    x, y, z = state
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


def Rossler_Derivatives(state, a, b, c):
    x, y, z = state
    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)
    return np.array([x_dot, y_dot, z_dot])


def Calculate_F(i, j, x_V, l, lambda_):
    H_i = hermite(i)
    H_j = hermite(j)

    def integrand(x_tilde):
        H_i_val = H_i(x_tilde / l)
        H_j_val = H_j(x_tilde / l)
        sech_val = 1 / np.cosh((x_tilde + x_V) / lambda_)
        return np.exp(-x_tilde ** 2 / l ** 2) * H_i_val * H_j_val * sech_val

    coefficient = 1 / (l * np.sqrt(2 ** (i + j) * factorial(i) * factorial(j) * np.pi))
    integral, _ = quad(integrand, -np.inf, np.inf)
    return coefficient * integral


def Memristor_Derivatives(state, Vn, Rn, ZT, alpha, Omega, l, lambda_, x0):
    Z, X, Y, V = state
    x_V = x0 - l * np.sqrt(2) * V

    F0_0 = Calculate_F(0, 0, x_V, l, lambda_)
    F1_1 = Calculate_F(1, 1, x_V, l, lambda_)
    F0_1 = Calculate_F(0, 1, x_V, l, lambda_)

    dV_dt = Vn - (1 + Rn * (F0_0 * (1 + Z) / 2 + F1_1 * (1 - Z) / 2 + F0_1 * X)) * V
    dZ_dt = -2 * dV_dt * X - alpha * Omega * (Z - ZT)
    dX_dt = Omega * Y + 2 * dV_dt * Z - Omega * X
    dY_dt = -Omega * X - Omega * Y

    return np.array([dZ_dt, dX_dt, dY_dt, dV_dt])


def Simulate_Attractor(Step_Func, Deriv_Func, Initial_Conditions, params, dt, num_steps, out_steps):
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


def Save_Results_To_CSV(results, filename, system_type):
    if system_type == "Memristor":
        columns = ["Time", "Z", "X", "Y", "V"]
    elif system_type == "Lorenz":
        columns = ["Time", "X", "Y", "Z"]
    elif system_type == "Rossler":
        columns = ["Time", "X", "Y", "Z"]
    else:
        columns = ["Time"] + [f"State {i + 1}" for i in range(results.shape[1] - 1)]

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def Op_dt(Integration_Func, Deriv_Func, Initial_Conditions, params, max_dt, num_steps, accuracy):
    dt = max_dt
    xyz = np.array(Initial_Conditions)
    current_result = xyz.copy()

    for _ in range(num_steps):
        current_result = Integration_Func(Deriv_Func, current_result, dt, *params)

    while dt > 1e-12:
        small_dt = dt * 0.1
        small_result = xyz.copy()

        for _ in range(int(num_steps / 10)):
            small_result = Integration_Func(Deriv_Func, small_result, small_dt, *params)

        calc_accuracy = np.linalg.norm(current_result - small_result)

        if calc_accuracy < accuracy:
            print(f"Optimal dt: {dt}")
            return dt
        else:
            current_result = small_result
            dt = small_dt

    print("Optimal dt was not found within limits.")
    return dt

# Dictionary
Calculations = {
    "Euler": Euler_Method,
    "Runge-Kutta": Runge_Kutta_Method,
    "Lorenz": Lorenz_Derivatives,
    "Rossler": Rossler_Derivatives,
    "Memristor": Memristor_Derivatives
}

## Example: Simulate Lorenz with Euler Method
# results = Simulate_Attractor(
#     Calculations["Euler"],
#     Calculations["Lorenz"],
#     [0., 1., 1.05],
#     (10, 28, 2.667),
#     dt=0.0001,
#     num_steps=1000000,
#     out_steps=10000
# )
# Save_Results_To_CSV(results, Calculations["Euler"], Calculations["Lorenz"])
#
#
#
# optimal_dt = Op_dt(
#     Calculations["Runge-Kutta"],
#     Calculations["Lorenz"],
#     [0., 1., 1.05],
#     (10, 28, 2.667),
#     max_dt=0.01,
#     num_steps=1000,
#     accuracy=1e-4
# )
#
#
# results = Simulate_Attractor(
#     Calculations["Euler"],
#     Calculations["Lorenz"],
#     [0., 1., 1.05],
#     (10, 28, 2.667),
#     dt=0.0001,
#     num_steps=1000000,
#     out_steps=10000
# )
# Save_Results_To_CSV(results, Calculations["Euler"], Calculations["Lorenz"])
#
# results = Simulate_Attractor(
#     Calculations["Runge-Kutta"],
#     Calculations["Lorenz"],
#     [0., 1., 1.05],
#     (10, 28, 2.667),
#     dt=0.0001,
#     num_steps=1000000,
#     out_steps=10000
# )
# Save_Results_To_CSV(results, Calculations["Runge-Kutta"], Calculations["Lorenz"])
#
# results = Simulate_Attractor(
#     Calculations["Euler"],
#     Calculations["Rossler"],
#     [0., 1., 1.05],
#     (0.1, 0.1, 14),
#     dt=0.01,
#     num_steps=10000,
#     out_steps=10000
# )
# Save_Results_To_CSV(results,  Calculations["Euler"], Calculations["Rossler"])
#
# results = Simulate_Attractor(
#     Calculations["Runge-Kutta"],
#     Calculations["Rossler"],
#     [0., 1., 1.05],
#     (0.1, 0.1, 14),
#     dt=0.01,
#     num_steps=10000,
#     out_steps=10000
# )
# Save_Results_To_CSV(results,  Calculations["Runge-Kutta"], Calculations["Rossler"])


# Parameters for Memristor
alpha = 1.0
Omega = 0.1
ZT = 1.0
lambda_ = 0.13
l = 0.5
x0 = 0.8
Vn = 0.23
Rn = 5.0
Initial_Conditions_Memristor = [0.0, 0.0, 0.0, 0.1]

results_memristor = Simulate_Attractor(
    Step_Func=Runge_Kutta_Method,
    Deriv_Func=Memristor_Derivatives,
    Initial_Conditions=Initial_Conditions_Memristor,
    params=(Vn, Rn, ZT, alpha, Omega, l, lambda_, x0),
    dt=0.01,
    num_steps=10000,
    out_steps=1000
)
Save_Results_To_CSV(results_memristor, "memristor_simulation.csv", "Memristor")
