import numpy as np
import os


def hypergraph_nonlinear_power_method(
    hypergraph,
    objective_function=lambda x: 1 / x,
    exponent=10,
    norm_parameter=11,
    tolerance=1e-8,
    initial_guess="ones",
    verbose=False,
    max_iterations=200
):
    """
    Compute x with nonnegative entries that maximizes the given objective function 
    over a hypergraph using the Nonlinear Power Method.

    Args:
        hypergraph: The hypergraph object on which to perform the optimization
        objective_function: A function representing the objective to be maximized (default: 1/x)
        exponent: The exponent used in the power iteration (default: 10)
        norm_parameter: The parameter 'p' used in the p-norm calculation (default: 11)
        tolerance: The convergence tolerance for the algorithm (default: 1e-8)
        initial_guess: The initial guess for the solution, either "ones" or a NumPy array (default: "ones")
        verbose: Whether to print progress information (default: False)
        max_iterations: The maximum number of iterations allowed (default: 200)

    Returns:
        The final solution x, the history of x values (x_array), and the history of error values (er_array)
    """

    num_nodes = hypergraph.num_nodes

    if initial_guess == "ones":
        initial_guess = np.ones((num_nodes, 1))

    if verbose:
        print("Nonlinear Power Method for Hypergraph CP:")
        print("-------------------------------")
        print(f"exponent:\t\t{exponent}\nnorm_parameter:\t\t\t{norm_parameter}\ntolerance:\t\t{tolerance}")

    p_prime = norm_parameter / (norm_parameter - 1)
    x_array = initial_guess / np.linalg.norm(initial_guess, p_prime)
    er_array = []

    for k in range(1, max_iterations + 1):
        y = pm_iterator(initial_guess.flatten(), hypergraph, objective_function, exponent)
        y = y / np.linalg.norm(y, p_prime)
        x = y ** (p_prime / norm_parameter)
        x_array = np.hstack((x_array, x))
        er_array.append(np.linalg.norm(x - initial_guess))
        if er_array[-1] < tolerance or np.isnan(er_array[-1]):
            if verbose:
                print(f"Num iter:\t{k}")
            break
        else:
            initial_guess = x.copy()

    return x_array[:, -1], x_array, er_array