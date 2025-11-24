# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

def rk3(A, bvector, y0, interval, N):
    """
    Solve the ODE dy/dx = Ay + b(x) using the explicit RK3 method.

    Parameters:
    - A: Matrix A as defined in dy/dx = Ay + b(x).
    - bvector: Function representing the vector b(x).
    - y0: Initial data vector.
    - interval: List [x0, x_end] giving the start and end values.
    - N: Number of steps.

    Returns:
    - x: Locations where the solution is evaluated.
    - y: Numerical solution at the corresponding locations.
    """
    x0, x_end = interval
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N + 1)
    y = np.zeros((len(y0), N + 1))
    y[:, 0] = y0

    for j in range(N):
        yn = y[:, j]
        k1 = h * (A @ yn + bvector(x[j]))
        k2 = h * (A @ (3/4 * yn + 1/4 * (yn + k1)) + bvector(x[j] + h))
        k3 = h * (A @ (1/3 * yn + 2/3 * (yn + k2)) + bvector(x[j] + h))
        y[:, j + 1] = 1/3 * yn + 2/3 * (yn + k3)

    return x, y

def dirk3(A, bvector, y0, interval, N):
    """
    Solve the ODE dy/dx = Ay + b(x) using the implicit DIRK3 algorithm.

    Parameters:
    - A: Matrix A as defined in dy/dx = Ay + b(x).
    - bvector: Function representing the vector b(x).
    - y0: Initial data vector.
    - interval: List [x0, x_end] giving the start and end values.
    - N: Number of steps.

    Returns:
    - x: Locations where the solution is evaluated.
    - y: Numerical solution at the corresponding locations.
    """
    x0, x_end = interval
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N + 1)
    y = np.zeros((len(y0), N + 1))
    y[:, 0] = y0

    mu = 0.5 * (1 - 1/np.sqrt(3))
    nu = 0.5 * (np.sqrt(3) - 1)
    gamma = 1.5 * (3 + np.sqrt(3))
    lam = (3 * (1 + np.sqrt(3))) / (2 * (3 + np.sqrt(3)))

    for j in range(N):
        yn = y[:, j]
        identity_matrix = np.identity(len(yn))
        matrix_eq1 = identity_matrix - h * mu * A
        matrix_eq2 = identity_matrix - h * mu * A - h * nu * A
        matrix_eq3 = identity_matrix - lam * h * A

        # Solve linear systems
        y1 = np.linalg.solve(matrix_eq1, yn + h * mu * bvector(x[j] + h * mu))
        y2 = np.linalg.solve(matrix_eq2, y1 + h * nu * (A @ y1 + bvector(x[j] + h * mu)) + h * mu * bvector(x[j] + h * nu + 2 * h * mu))
        y[:, j + 1] = (1 - lam) * yn + lam * y2 + h * gamma * (A @ y2 + bvector(x[j] + h * nu + 2 * h * mu))

    return x, y


def trivial_b(x):
    """
    Trivial vector function b for task 3.

    Parameters:
    - x: Location.

    Returns:
    - b: Trivial vector b.
    """
    return np.zeros(2)

def exact_solution_a(x, a1, a2, N):
    """
    Exact solution for task 3.

    Parameters:
    - x: Locations where the solution is evaluated.
    - a1, a2: Constants.
    - N: Number of steps.

    Returns:
    - y_exact: Exact solution array.
    """
    y1 = np.exp(-a1 * x)
    y2 = (a1 / (a1 - a2)) * (np.exp(-a2 * x) - np.exp(-a1 * x))

    # Interpolate the exact solution to match the length of numerical solution
    x_exact = np.linspace(x[0], x[-1], N + 1)
    y1_exact = np.interp(x_exact, x, y1)
    y2_exact = np.interp(x_exact, x, y2)

    return x_exact, np.array([y1_exact, y2_exact])

def compute_error(y_num, y_exact, h):
    """
    Compute the 1-norm of the relative error.

    Parameters:
    - y_num: Numerical solution.
    - y_exact: Exact solution.
    -h: Interval/N

    Returns:
    - error: 1-norm of the relative error.
    """
   
    error_array = h*np.abs((y_num - y_exact) / np.maximum(np.abs(y_exact), 1e-15))
    if y_num.ndim == 1:
        return np.sum(error_array)
    else:
        return np.sum(error_array[1:])

def plot_task3():
    a1 = 1000
    a2 = 1
    y0 = np.array([1, 0])
    interval = [0, 0.1]

    N_values = [40 * k for k in range(1, 11)]
    errors = []

    for N in N_values:
        x_rk3, y_rk3 = rk3(np.array([[-a1, 0], [a1, -a2]]), trivial_b, y0, interval, N)
        x_exact, y_exact = exact_solution_a(x_rk3, a1, a2, N)

        error = compute_error(y_rk3, y_exact, (x_rk3[1]-x_rk3[0]))
        errors.append(error)

    # Fit a polynomial curve to the logarithm of errors and step sizes
    log_errors = np.log(errors)
    log_step_sizes = np.log(0.1 / np.array(N_values))
    degree = 3  # Convergence order is 3

    coefficients = np.polyfit(log_step_sizes, log_errors, degree)
    fitted_curve = np.poly1d(coefficients)

    # Plot errors against h and fitted curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.loglog(0.1 / np.array(N_values), errors, marker='o', label='RK3')
    plt.xlabel('Step size (h)')
    plt.ylabel('1-norm of relative error')
    plt.title('Convergence Analysis - RK3')
    plt.loglog(0.1 / np.array(N_values), np.exp(fitted_curve(log_step_sizes)), linestyle='--', color='orange', label=f'Fitted Curve (Order {degree})')
    plt.legend()

    # Plot numerical and exact solutions for y2
    plt.figure(figsize=(12, 6))
    _, y_rk3_high_res = rk3(np.array([[-a1, 0], [a1, -a2]]), trivial_b, y0, interval, 400)
    x_exact, y_exact = exact_solution_a(x_rk3, a1, a2, N)
    plt.subplot(1, 2, 1)
    plt.plot(x_rk3, y_rk3[1, :], label='Numerical Solution (RK3, N=400)')
    plt.plot(x_rk3, y_exact[1], label='Exact Solution')
    plt.xlabel('x')
    plt.ylabel('y2')
    plt.title('Comparison of Numerical and Exact Solutions for y2')
    plt.legend()


def plot_task4():
    A_task4 = np.array([[-1, 0, 0], [-99, -100, 0], [-10098, 9900, -10000]])

    def bvector_task4(x):
        return np.array([np.cos(10 * x) - 10 * np.sin(10 * x),
                         199 * np.cos(10 * x) - 10 * np.sin(10 * x),
                         208 * np.cos(10 * x) + 10000 * np.sin(10 * x)])

    y0_task4 = np.array([0, 1, 0])
    interval_task4 = [0, 1]

    N_values_task4 = [200 * k for k in range(4, 17)]
    errors_task4 = []

    for N in N_values_task4:
        x_dirk3, y_dirk3 = dirk3(A_task4, bvector_task4, y0_task4, interval_task4, N)
        y_exact_task4 = exact_solution_b(x_dirk3)

        error_task4 = compute_error(y_dirk3, y_exact_task4,  (x_dirk3[1] - x_dirk3[0]))
        errors_task4.append(error_task4)

    # Plot errors against h
    plt.figure()
    plt.loglog(1 / np.array(N_values_task4), errors_task4, marker='o', label='DIRK3')
    plt.xlabel('Step size (h)')
    plt.ylabel('1-norm of relative error')
    plt.title('Convergence Analysis - DIRK3')
    plt.legend()
    plt.show()

    # Plot numerical and exact solutions
    _, y_dirk3_high_res = dirk3(A_task4, bvector_task4, y0_task4, interval_task4, 3200)
    plt.figure(figsize=(12, 8))

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x_dirk3, y_dirk3[i, :], label='DIRK3 (N=3200)')
        plt.plot(x_dirk3, y_exact_task4[i, :], label='Exact Solution')
        plt.xlabel('x')
        plt.ylabel(f'y{i + 1}')
        plt.title(f'Comparison of Numerical and Exact Solutions - DIRK3 (Component {i + 1})')
        plt.legend()

    plt.tight_layout()
    plt.show()

def exact_solution_b(x):
    """
    Exact solution for task 4.

    Parameters:
    - x: Location.

    Returns:
    - y_exact: Exact solution matrix.
    """
    y1 = np.cos(10 * x) - np.exp(-x)
    y2 = np.cos(10 * x) + np.exp(-x) - np.exp(-100 * x)
    y3 = np.sin(10 * x) + 2 * np.exp(-x) - np.exp(-100 * x) - np.exp(-10000 * x)
    return np.array([y1, y2, y3])

# Run the plots
plot_task3()
plot_task4()
