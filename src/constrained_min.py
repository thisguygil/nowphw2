import math  # We use math rather than numpy only for the log calculations because it is more efficient for scalar operations
from typing import Callable
import numpy as np
from numpy.typing import NDArray

def interior_pt(
    func: Callable[[NDArray, bool], tuple[float, NDArray, NDArray | None]],
    ineq_constraints: list[Callable[[NDArray, bool], tuple[float, NDArray, NDArray | None]]],
    eq_constraints_mat: NDArray,
    eq_constraints_rhs: NDArray,
    x0: NDArray
) -> tuple[list[NDArray], list[float], list[NDArray], list[float]]:
    """
    Interior point method for constrained minimization, using the log-barrier method.
    
    Args:
        func: The objective function to minimize.
        ineq_constraints: The list of inequality constraints.
        eq_constraints_mat: The matrix of equality constraints.
        eq_constraints_rhs: The right-hand side vector of the equality constraints.
        x0: The starting point for the minimization.
        
    Returns:
        tuple[list[NDArray], list[float], list[NDArray], list[float]]: A tuple containing:
            list[NDArray]: The history of the inner iterations' points.
            list[float]: The history of the inner iterations' objective function values.
            list[NDArray]: The history of the outer iterations' points.
            list[float]: The history of the outer iterations' objective function values.
    """
    t = 1  # Initial value of the barrier parameter
    mu = 10  # Barrier parameter multiplier
    obj_tol = 10e-12  # Tolerance for the objective function change
    epsilon = 10e-10  # Stopping criterion tolerance

    x = x0.copy()  # Starting point for the minimization
    f_x, grad_x, hess_x = func(x, True)  # Evaluate objective function, gradient, and Hessian
    f_x_lb, grad_x_lb, hess_x_lb = log_barrier(x, ineq_constraints)  # Evaluate log-barrier function, gradient, and Hessian

    # Initialize history lists for inner and outer iterations
    inner_x_history = [x.copy()]
    inner_obj_value_history = [f_x]
    outer_x_history = [x.copy()]
    outer_obj_value_history = [f_x]

    # Combine objective function and log-barrier contributions
    f_x = t * f_x + f_x_lb
    grad_x = t * grad_x + grad_x_lb
    hess_x = t * hess_x + hess_x_lb

    while True:
        # Construct the KKT matrix and vector for solving the system of equations
        kkt_matrix, kkt_vector = create_kkt_matrix(grad_x, hess_x, eq_constraints_mat)

        x_prev = x  # Previous iteration point
        f_x_prev = f_x  # Previous objective function value

        iter = 0  # Inner iteration counter
        while True:
            # Stop if the change in x is small
            if iter != 0 and sum(abs(x - x_prev)) < 10e-8:
                break

            # Solve for the search direction
            p = np.linalg.solve(kkt_matrix, kkt_vector)[:x.shape[0]]

            # Stop if the change in the objective function is small
            if 0.5 * ((p.T @ (hess_x @ p)) ** 0.5) ** 2 < obj_tol:
                break

            # Stop if the reduction in the objective function is small
            if iter != 0 and (f_x_prev - f_x) < obj_tol:
                break

            # Perform line search using Wolfe conditions
            alpha = wolfe(func, p, x)

            # Update the previous point and objective function value
            x_prev = x
            f_x_prev = f_x

            # Update the point and evaluate the objective function, gradient, and Hessian
            x = x + alpha * p
            f_x, grad_x, hess_x = func(x, True)
            f_x_lb, grad_x_lb, hess_x_lb = log_barrier(x, ineq_constraints)

            # Store the inner iteration points and objective function values
            inner_x_history.append(x)
            inner_obj_value_history.append(f_x)

            # Combine objective function and log-barrier contributions
            f_x = t * f_x + f_x_lb
            grad_x = t * grad_x + grad_x_lb
            hess_x = t * hess_x + hess_x_lb

            iter += 1

        outer_x_history.append(x)  # Store outer iteration points
        outer_obj_value_history.append((f_x - f_x_lb) / t)  # Store outer iteration objective function values

        if len(ineq_constraints) / t < epsilon:  # Stopping criterion
            break

        # Increase barrier parameter
        t *= mu  
    
    return inner_x_history, inner_obj_value_history, outer_x_history, outer_obj_value_history
        
def create_kkt_matrix(
    grad_x: NDArray,
    hess_x: NDArray,
    eq_constraints_mat: NDArray
) -> NDArray:
    """
    Creates the block matrix for equality constraints.

    Args:
        grad_x: The gradient of the objective function.
        hess_x: The Hessian matrix of the objective function.
        eq_constraints_mat: The matrix of equality constraints.

    Returns:
        NDArray: The block matrix.
    """
    if eq_constraints_mat.size > 0:
        upper = np.concatenate([hess_x, eq_constraints_mat.T], axis=1)
        lower = np.concatenate([eq_constraints_mat, np.zeros((eq_constraints_mat.shape[0], eq_constraints_mat.shape[0]))], axis=1)
        kkt_matrix = np.concatenate([upper, lower], axis=0)
    else:
        kkt_matrix = hess_x

    kkt_vector = np.concatenate([-grad_x, np.zeros(kkt_matrix.shape[0] - len(grad_x))])

    return kkt_matrix, kkt_vector

def log_barrier(
    x: NDArray,
    ineq_constraints: list[Callable[[NDArray, bool], tuple[float, NDArray, NDArray | None]]]
) -> tuple[float, NDArray, NDArray]:
    """
    The log-barrier function for inequality constraints.
    
    Args:
        x: The point at which to evaluate the function.
        ineq_constraints: The list of inequality constraints.
        
    Returns:
        tuple[float, NDArray, NDArray]: A tuple containing:
            float: The value of the log-barrier function at x.
            NDArray: The gradient of the log-barrier function at x.
            NDArray: The Hessian matrix of the log-barrier function at x.
    """
    # Initialize the value, gradient, and Hessian matrix
    value = 0.0
    grad = np.zeros_like(x)
    hess = np.zeros((x.shape[0], x.shape[0]))

    for constraint in ineq_constraints:
        # Evaluate constraint function, gradient, and Hessian
        c_x, grad_x, hess_x = constraint(x, True)

        # Update log-barrier function value
        value -= math.log(-c_x)

        # Compute gradient contribution
        g = grad_x / c_x

        # Update gradient
        grad -= g

        # Update Hessian matrix
        hess -= (hess_x * c_x - np.outer(g, g)) / c_x**2

    return value, grad, hess

def wolfe(
    func: Callable[[NDArray, bool], tuple[float, NDArray, NDArray | None]],
    p: NDArray,
    x: NDArray
) -> float:
    """
    Wolfe conditions for line search.
    
    Args:
        func: The objective function to minimize.
        p: The search direction.
        x: The current point.
        
    Returns:
        float: The step size that satisfies the Wolfe conditions.
    """
    # Wolfe constants
    wolfe_const = 0.01
    backtracking_const = 0.5

    alpha = 1  # Initial step size

    def check_conditions(
        alpha: float
    ) -> bool:
        """
        Checks the Wolfe conditions.
        
        Args:
            alpha: The step size.

        Returns:
            bool: A boolean indicating whether the Wolfe conditions are satisfied.
        """
        sufficient_decrease = func(x + alpha * p)[0] <= func(x)[0] + wolfe_const * alpha * (func(x)[1] @ p)
        return sufficient_decrease

    # Backtracking line search
    while not check_conditions(alpha):
        alpha *= backtracking_const

    return alpha
