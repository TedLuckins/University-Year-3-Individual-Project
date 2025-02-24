import numpy as np
import pandas as pd
import time

start_time = time.time()

from Calculations import Precompute_CalcF,simulate_until_equilibrium, get_equilibrium_V

# Save for memristor equalibrium V
def save_results_to_csv(Vn_values, V_results_min, V_results_max, filename):
    df = pd.DataFrame({
        "Vn": Vn_values,
        "V_min": V_results_min,
        "V_max": V_results_max
    })
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Parameters for Memristor System
alpha = 1.0
Omega = 7.0
Gamma = 0.1
Z_T = 1.0
Lambda = 0.13
l = 0.5
x_0 = 0.8
#V_n = 1.0
R_n = 5.0

Initial_Conditions_Memristor = [0, 0, 1.0, 0]

# Precompute values
x_V_range = np.arange(-3, 3.01, 0.01)
precomputed_F = Precompute_CalcF(x_V_range, l, Lambda)

Vn_values_up = np.arange(0.01, 1.01, 1.0)
Vn_values_down = np.arange(5, 0.00, -0.01)

V_results_up_min = []
V_results_up_max = []
V_results_down_min = []
V_results_down_max = []


current_initial_conditions = Initial_Conditions_Memristor.copy()

# Forward simulation
for V_n in Vn_values_up:
    results, current_initial_conditions = simulate_until_equilibrium(V_n, R_n, current_initial_conditions)
    V_min, V_max = get_equilibrium_V(results)
    V_results_up_min.append(V_min)
    V_results_up_max.append(V_max)
    Time = time.time() - start_time

save_results_to_csv(Vn_values_up, V_results_up_min, V_results_up_max,
                    "memristor_simulation_Test.csv")


current_initial_conditions = Initial_Conditions_Memristor.copy()

# Backward simulation
for V_n in Vn_values_down:
    results, current_initial_conditions = simulate_until_equilibrium(V_n, current_initial_conditions)
    V_min, V_max = get_equilibrium_V(results)
    V_results_down_min.append(V_min)
    V_results_down_max.append(V_max)

save_results_to_csv(Vn_values_down, V_results_down_min, V_results_down_max,
                    "Memristor/Hysteresis/memristor_decreasing_dVn0.01_dt0.001.csv")

end_time = time.time()
total_time = end_time - start_time
print(f"Simulation completed in {total_time:.2f} seconds.")