# Libraries
import numpy as np
import pandas as pd
import time

from Calculations import simulate_until_equilibrium, get_equilibrium_V, Precompute_CalcF

start_time = time.time()

def save_results_to_csv(Vn_values, Rn_values, V_results_min, V_results_max, filename):
    df = pd.DataFrame({
        "Vn": Vn_values,
        "Rn": Rn_values,
        "V_min": V_results_min,
        "V_max": V_results_max
    })
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# Parameters for Memristor System
alpha = 1.0
Omega = 7.0
Gamma = 0.1
Z_T = 0.3
Lambda = 0.13
l = 0.5
x_0 = 0.8
V_n = 1.0
R_n = 5.0

# Precompute values
x_V_range = np.arange(-3, 3.01, 0.01)
precomputed_F = Precompute_CalcF(x_V_range, l, Lambda)

Initial_Conditions_Memristor = [0, 0, 1.0, 0]

dt = 0.01
max_steps = 100000

# Define Vn and Rn ranges
Vn_values = np.arange(0.01, 40.01, 1)
Rn_values = np.arange(1, 11, 10)

V_results_min = []
V_results_max = []
Vn_list = []
Rn_list = []

# Initial conditions
current_initial_conditions = Initial_Conditions_Memristor.copy()


# Sweeps over  Vn and Rn
for R_n in Rn_values:
    for V_n in Vn_values:
        results, current_initial_conditions = simulate_until_equilibrium(R_n, V_n, current_initial_conditions, max_steps, dt)
        V_min, V_max = get_equilibrium_V(results)

        Time = time.time() - start_time

        Vn_list.append(V_n)
        Rn_list.append(R_n)
        V_results_min.append(V_min)
        V_results_max.append(V_max)
        print(f"Vn is {V_n} and time is {Time:.2f}")
    print(f"Rn value {R_n} is complete at time {Time:.2f}")

save_results_to_csv(Vn_list, Rn_list, V_results_min, V_results_max,
                    "memristor_oscillation_regions_Test3.csv")
end_time = time.time()
total_time = end_time - start_time
print(f"Simulation completed in {total_time:.2f} seconds.")