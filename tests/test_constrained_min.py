import unittest
import numpy as np
from src.constrained_min import interior_pt
from src.utils import plot_iterations, plot_feasible_region_qp, plot_feasible_region_lp
from tests.examples import qp, qp_ineq_1, qp_ineq_2, qp_ineq_3, lp, lp_ineq_1, lp_ineq_2, lp_ineq_3, lp_ineq_4

class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        """
        Test the interior point method for a quadratic program.
        
        The quadratic program is:
        min x^2 + y^2 + (z + 1)^2
        s.t. x + y + z = 1
             x >= 0
             y >= 0
             z >= 0
        """
        # Define the problem
        ineq_constraints = [qp_ineq_1, qp_ineq_2, qp_ineq_3]
        eq_constraint_mat = np.array([[1, 1, 1]]).reshape(1, -1)
        eq_constraint_rhs = np.array([1])
        x0 = np.array([0.1, 0.2, 0.7], dtype=np.float64)

        # Solve the problem
        inner_x_history, inner_obj_value_history, _, outer_obj_value_history = interior_pt(qp, ineq_constraints, eq_constraint_mat, eq_constraint_rhs, x0)

        # Print results
        print(f"Point of convergence: {inner_x_history[-1]}")
        print(f"Objective value at point of convergence: {qp(inner_x_history[-1])[0]:.4f} (minimized value)")
        print(f"Sum of variables (should be 1): {inner_x_history[-1][0] + inner_x_history[-1][1] + inner_x_history[-1][2]:.4f}")
        print()

        # Plot results
        plot_iterations("Quadractic", inner_obj_value_history, outer_obj_value_history)
        plot_feasible_region_qp(inner_x_history)

    def test_lp(self):
        """
        Test the interior point method for a linear program.
        
        The linear program is:
        max x + y
        s.t. y >= -x + 1
             y <= 1
             x <= 2
             y >= 0
        """
        # Define the problem
        ineq_constraints = [lp_ineq_1, lp_ineq_2, lp_ineq_3, lp_ineq_4]
        eq_constraint_mat = np.array([])
        eq_constraint_rhs = np.array([])
        x0 = np.array([0.5, 0.75], dtype=np.float64)

        # Solve the problem
        inner_x_history, inner_obj_value_history, _, outer_obj_value_history = interior_pt(lp, ineq_constraints, eq_constraint_mat, eq_constraint_rhs, x0)

        # Convert back to maximization problem
        inner_obj_value_history = [-val for val in inner_obj_value_history]
        outer_obj_value_history = [-val for val in outer_obj_value_history]

        # Print results
        print(f"Point of convergence: {inner_x_history[-1]}")
        print(f"Objective value at point of convergence: {-lp(inner_x_history[-1])[0]:.4f} (maximized value)")
        print()

        # Plot results
        plot_iterations("Linear", inner_obj_value_history, outer_obj_value_history)
        plot_feasible_region_lp(inner_x_history)


if __name__ == '__main__':
    unittest.main()