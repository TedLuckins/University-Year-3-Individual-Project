from cgitb import small

import numpy as np
import matplotlib.pyplot as plt


def Euler_Method(Deriv_Func, xyz, dt, *params):
    """
    :param Deriv_Func: function
            The function that computes the derivatives
    :param Intital_Conditions: array-like, shape (3,)
            Initial values of the system (x, y, z)
    :param num_steps: int
            Number of time iterations
    :param dt: float
            Time step of each iteration
    :param params: tuple
            parameters of the system for derivation function
    :return: xyzs: ndarray, shape (num_steps + 1, 3)
            Array of (x, y, z) values at each time step
    """
    return xyz + Deriv_Func(xyz, *params) * dt
    """
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = Initial_Conditions
    results = []

    # Euler integration loop
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + Deriv_Func(xyzs[i], *params) * dt
        if i % (num_steps/out_steps) == 0:
            results.append(xyzs[i])
    return np.array(results)
    """


def Runge_Kutta_Method(Deriv_Func, xyz, dt, *params):
    """
    :param Deriv_Func: function
            The function that computes the derivatives
    :param Intital_Conditions: array-like, shape (3,)
            Initial values of the system (x, y, z)
    :param num_steps: int
            Number of time iterations
    :param dt: float
            Time step of each iteration
    :param params: tuple
            parameters of the system for derivation function
    :return: xyzs: ndarray, shape (num_steps + 1, 3)
            Array of (x, y, z) values at each time step
    """
    k1 = Deriv_Func(xyz, *params)
    k2 = Deriv_Func(xyz + 0.5 * k1 * dt, *params)
    k3 = Deriv_Func(xyz + 0.5 * k2 * dt, *params)
    k4 = Deriv_Func(xyz + k3 * dt, *params)
    return xyz + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0)
    """
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = Initial_Conditions
    results = []
    
    #Runge-Kutta integration loop
    for i in range(num_steps):
        k1 = Deriv_Func(xyzs[i], *params) #Computes the derivative at current point (using Euler method)
        k2 = Deriv_Func(xyzs[i]+ 0.5 * k1 * dt, *params) #Computes the derivative at the midpoint, using k1
        k3 = Deriv_Func(xyzs[i]+ 0.5 * k2 * dt, *params) #COpmputes the derivative at another midpoint, using k2
        k4 = Deriv_Func(xyzs[i] + k3 * dt, *params) #Computes the derivative at the endpoint, using k3

        # Average the k increments with weight (1/6, 1/3, 1/3, 1/6) to update the solution
        xyzs[i + 1] = xyzs[i] + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0)
        if i % (num_steps / out_steps) == 0:
            results.append(xyzs[i])
    return np.array(results)
    """

def Plot_Attractor(xyzs, title):
    """
    A plotting function for attractors
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(*xyzs.T, lw=0.6)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    plt.show()

def Lorenz_Derivatives(xyz, s, r, b):
    """
    Computes the Lorenz attractor derivatives

    :param xyz: array-like, shape (3,)
            Initial condition for the Lorenz system [x0, y0, z0].
    :param s: float, optional
            Parameter of the Lorenz system.
    :param r: float, optional
            Parameter of the Lorenz system.
    :param b: float, optional
    """
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])

def Rossler_Derivatives(xyz, a, b, c):
    """
    Computes the Rossler attractor derivatives

    :param xyz: array-like, shape (3,)
            Initial condition for the Rossler system [x0, y0, z0].
    :param a: float, optional
            Parameter of the Rossler system.
    :param b: float, optional
            Parameter of the Rossler system.
    :param c: float, optional
            Parameter of the Rossler system.
    """
    x, y, z = xyz
    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)
    return np.array([x_dot, y_dot, z_dot])

#Dictionary
Integration_Methods = {
    "Euler" : Euler_Method,
    "Runge-Kutta" : Runge_Kutta_Method
}

def Simulate_Attractor(Step_Func, Deriv_Func, Initial_Conditions, params, dt, num_steps, out_steps, title):
    """
   Simulate and plot the attractor.
    """
    xyz = np.array(Initial_Conditions)
    results = [xyz]

    for i in range(1, num_steps + 1):
        xyz = Step_Func(Deriv_Func, xyz, dt, *params)

        if i % (num_steps // out_steps) == 0 or i == num_steps:
            results.append(xyz)

    results = np.array(results)
    #Plot results
    Plot_Attractor(results, title)
"""
    results = np.array(results)
    Plot_Attractor(results, title)
    #Calls specified functions and methods
    xyzs = Integration_Func(Deriv_Func, Initial_Conditions, num_steps, out_steps, dt, *params)
    """

"""
def Op_dt(Integration_Func, Deriv_Func, Initial_Conditions, params, max_dt, num_steps, accuracy):
    dt = max_dt
    #point = np.array(Initial_Conditions)
    while dt > 10**(-12):
        small_dt = dt * 0.1
        step = Integration_Func(Deriv_Func, Initial_Conditions, num_steps, dt, *params)[1]
        small_step = Integration_Func(Deriv_Func, Initial_Conditions, 10 * num_steps, small_dt, *params)[1]

        calc_accuracy = np.linalg.norm(step - small_step)

        if calc_accuracy < accuracy:
            return print("Optimal dt:", dt)
        else:
            dt = small_dt
    print("Optimal dt was not found in limits")
    return print("Best dt within limts:", dt)
"""

def Op_dt(Integration_Func, Deriv_Func, Initial_Conditions, params, max_dt, num_steps, accuracy):
    """
    Finds the optimal time step (dt) for each integration method for a certain predetermined accuracy
    :param Integration_Func:
    :param Deriv_Func: function
            The function that computes the derivatives
    :param Initial_Conditions: array-like, shape (3,)
            Initial values of the system (x, y, z)
    :param params: tuple
            parameters of the system for derivation function
    :param max_dt: float
            Maximum time step
    :param num_steps: int
            Number of time iterations
    :param accuracy: float
            the maximum absolute error of the final plot for the optimal time step
    :return: float
            Optime time step value
    """
    dt = max_dt
    #point = np.array(Initial_Conditions)
    step = Integration_Func(Deriv_Func, Initial_Conditions, num_steps, dt, *params)[1]
    while dt > 1e-12:
        small_dt = dt * 0.1
        small_step = Integration_Func(Deriv_Func, Initial_Conditions, int(num_steps/dt), small_dt, *params)[1]

        calc_accuracy = np.linalg.norm(step - small_step)

        if calc_accuracy < accuracy:
            return print("Optimal dt:", dt)
        else:
            step = small_step
            dt = small_dt

    print("Optimal dt was not found in limits")
    return print("Best dt within limits:", dt)
"""
Op_dt(
    Integration_Methods["Runge-Kutta"],
    Lorenz_Derivatives,
    [0., 1., 1.05],
    (0.1, 0.1, 14),
    max_dt=0.01,
    num_steps=1000,
    accuracy = 1e-4
)
"""
# Lorenz example from wiki - parameters s=10, r=28, b=2.667
"""
Simulate_Attractor(
    Integration_Methods["Euler"],
    Lorenz_Derivatives,
    [0., 1., 1.05],
    (10, 28, 2.667),
    dt=10**(-4),
    num_steps=1000000,
    out_steps = 1000,
    title="Lorenz Attractor (Euler)"
)
"""

Simulate_Attractor(
    Integration_Methods["Runge-Kutta"],
    Lorenz_Derivatives, 
    [0., 1., 1.05], 
    (10, 28, 2.667), 
    dt=0.0001,
    num_steps=1000000,
    out_steps=1000,
    title="Lorenz Attractor (Runge_Kutta)"
)


# Rossler example from wiki - paparameters a=0.2, b=0.2, c=5.7

"""
Simulate_Attractor(
    Integration_Methods["Euler"],
    Rossler_Derivatives,
    [0., 1., 1.05],
    (0.1, 0.1, 14),
    dt=0.01,
    num_steps=10000,
    title="Rossler Attractor (Euler)"
)
"""
"""
Simulate_Attractor(
    Integration_Methods["Runge-Kutta"],
    Rossler_Derivatives, 
    [0., 1., 1.05], 
    (0.1, 0.1, 14), 
    dt=0.01, 
    num_steps=10000,
    title="Rossler Attractor (Runge_Kutta)"
)
"""


