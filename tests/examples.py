import numpy as np
from numpy.typing import NDArray

def qp(
    x: NDArray,
    hessian: bool = False
) -> tuple[float, NDArray, NDArray | None]:
    """
    Quadratic programming objective function.
    The problem is to minimize f(x) = x^2 + y^2 + (z + 1)^2.

    Args:
        x: The point at which to evaluate the function.
        hessian: A flag indicating whether to return the Hessian matrix.

    Returns:
        tuple[float, NDArray, NDArray | None]: A tuple containing:
            float: The value of the objective function at x.
            NDArray: The gradient of the objective function at x.
            NDArray | None: The Hessian matrix of the objective function at x, or None if hessian is False.
    """
    f = x[0]**2 + x[1]**2 + (x[2] + 1)**2
    grad = np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])

    if hessian:
        hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    else:
        hess = None
    
    return f, grad.T, hess

def qp_ineq_1(
    x: NDArray,
    hessian: bool = False
) -> tuple[float, NDArray, NDArray | None]:
    """
    Quadratic program inequality constraint 1.
    The constraint is x >= 0.
    
    Args:
        x: The point at which to evaluate the function.
        hessian: A flag indicating whether to return the Hessian matrix.
        
    Returns:
        tuple[float, NDArray, NDArray | None]: A tuple containing:
            float: The value of the constraint at x.
            NDArray: The gradient of the constraint at x.
            NDArray | None: The Hessian matrix of the constraint at x, or None if hessian is False.
    """
    c = -x[0]
    grad = np.array([-1, 0, 0])

    if hessian:
        grad = grad.T
        hess = np.zeros((3, 3))
    else:
        hess = None
    
    return c, grad, hess

def qp_ineq_2(
    x: NDArray,
    hessian: bool = False
) -> tuple[float, NDArray, NDArray | None]:
    """
    Quadratic program inequality constraint 2.
    The constraint is y >= 0.
    
    Args:
        x: The point at which to evaluate the function.
        hessian: A flag indicating whether to return the Hessian matrix.
        
    Returns:
        tuple[float, NDArray, NDArray | None]: A tuple containing:
            float: The value of the constraint at x.
            NDArray: The gradient of the constraint at x.
            NDArray | None: The Hessian matrix of the constraint at x, or None if hessian is False.
    """
    c = -x[1]
    grad = np.array([0, -1, 0])

    if hessian:
        grad = grad.T
        hess = np.zeros((3, 3))
    else:
        hess = None
    
    return c, grad, hess

def qp_ineq_3(
    x: NDArray,
    hessian: bool = False
) -> tuple[float, NDArray, NDArray | None]:
    """
    Quadratic program inequality constraint 3.
    The constraint is z >= 0.
    
    Args:
        x: The point at which to evaluate the function.
        hessian: A flag indicating whether to return the Hessian matrix.
        
    Returns:
        tuple[float, NDArray, NDArray | None]: A tuple containing:
            float: The value of the constraint at x.
            NDArray: The gradient of the constraint at x.
            NDArray | None: The Hessian matrix of the constraint at x, or None if hessian is False.
    """
    c = -x[2]
    grad = np.array([0, 0, -1])

    if hessian:
        grad = grad.T
        hess = np.zeros((3, 3))
    else:
        hess = None
    
    return c, grad, hess

def lp(
    x: NDArray,
    hessian: bool = False
) -> tuple[float, NDArray, NDArray | None]:
    """
    Linear programming objective function.
    The problem is to maximize f(x) = x + y.
    Note that this is a maximization problem, so the objective function is negated.

    Args:
        x: The point at which to evaluate the function.
        hessian: A flag indicating whether to return the Hessian matrix.

    Returns:
        tuple[float, NDArray, NDArray | None]: A tuple containing:
            float: The value of the objective function at x.
            NDArray: The gradient of the objective function at x.
            NDArray | None: The Hessian matrix of the objective function at x, or None if hessian is False.
    """
    f = -x[0] - x[1]
    grad = np.array([-1, -1])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return f, grad.T, hess

def lp_ineq_1(
    x: NDArray,
    hessian: bool = False
) -> tuple[float, NDArray, NDArray | None]:
    """
    Linear program inequality constraint 1.
    The constraint is y >= -x + 1.
    Note that this is a maximization problem, so the constraint function is negated.
    
    Args:
        x: The point at which to evaluate the function.
        hessian: A flag indicating whether to return the Hessian matrix.
        
    Returns:
        tuple[float, NDArray, NDArray | None]: A tuple containing:
            float: The value of the constraint at x.
            NDArray: The gradient of the constraint at x.
            NDArray | None: The Hessian matrix of the constraint at x, or None if hessian is False.
    """
    c = -x[0] - x[1] + 1
    grad = np.array([-1, -1])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return c, grad.T, hess

def lp_ineq_2(
    x: NDArray,
    hessian: bool = False
) -> tuple[float, NDArray, NDArray | None]:
    """
    Linear program inequality constraint 2.
    The constraint is y <= 1.
    Note that this is a maximization problem, so the constraint function is negated.
    
    Args:
        x: The point at which to evaluate the function.
        hessian: A flag indicating whether to return the Hessian matrix.
        
    Returns:
        tuple[float, NDArray, NDArray | None]: A tuple containing:
            float: The value of the constraint at x.
            NDArray: The gradient of the constraint at x.
            NDArray | None: The Hessian matrix of the constraint at x, or None if hessian is False.
    """
    c = x[1] - 1
    grad = np.array([0, 1])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return c, grad.T, hess

def lp_ineq_3(
    x: NDArray,
    hessian: bool = False
) -> tuple[float, NDArray, NDArray | None]:
    """
    Linear program inequality constraint 3.
    The constraint is x <= 2.
    Note that this is a maximization problem, so the constraint function is negated.
    
    Args:
        x: The point at which to evaluate the function.
        hessian: A flag indicating whether to return the Hessian matrix.
        
    Returns:
        tuple[float, NDArray, NDArray | None]: A tuple containing:
            float: The value of the constraint at x.
            NDArray: The gradient of the constraint at x.
            NDArray | None: The Hessian matrix of the constraint at x, or None if hessian is False.
    """
    c = x[0] - 2
    grad = np.array([1, 0])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return c, grad.T, hess

def lp_ineq_4(
    x: NDArray,
    hessian: bool = False
) -> tuple[float, NDArray, NDArray | None]:
    """
    Linear program inequality constraint 4.
    The constraint is y >= 0.
    Note that this is a maximization problem, so the constraint function is negated.
    
    Args:
        x: The point at which to evaluate the function.
        hessian: A flag indicating whether to return the Hessian matrix.
        
    Returns:
        tuple[float, NDArray, NDArray | None]: A tuple containing:
            float: The value of the constraint at x.
            NDArray: The gradient of the constraint at x.
            NDArray | None: The Hessian matrix of the constraint at x, or None if hessian is False.
    """
    c = -x[1]
    grad = np.array([0, -1])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return c, grad.T, hess
