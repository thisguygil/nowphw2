import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def plot_iterations(
    program_type: str,
    inner_obj_values: list[float],
    outer_obj_values: list[float]
) -> None:
    """
    Plot the objective function values of the inner and outer iterations.
    
    Args:
        program_type: The type of program being solved (e.g., "Quadratic" or "Linear").
        inner_obj_values: The list of objective function values of the inner iterations.
        outer_obj_values: The list of objective function values of the outer iterations.
    """
    fig, ax = plt.subplots()
    ax.plot(range(len(inner_obj_values)), inner_obj_values, label="Inner Values")
    ax.plot(range(len(outer_obj_values)), outer_obj_values, label="Outer Values")

    ax.legend()
    ax.set_title(f"Objective Function Values of {program_type} Program")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Objective Function Value")
    plt.show()

def plot_feasible_region_qp(
    path_points: list[NDArray]
) -> None:
    """
    Plot the feasible region and path for a quadratic programming problem.
    
    Args:
        path_points: The list of points along the path to plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    path_array = np.array(path_points)

    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color="lightgray", alpha=0.5)
    ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], color="k", marker=".", linestyle="--", label="Path")
    ax.scatter(path_array[-1][0], path_array[-1][1], path_array[-1][2], s=50, c="gold", label="Final Candidate")

    ax.set_title("Feasible Region and Path of Quadratic Program")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend()
    ax.view_init(45, 45)
    plt.show()

def plot_feasible_region_lp(
    path_points: list[NDArray]
) -> None:
    """
    Plot the feasible region and path for a linear programming problem.
    
    Args:
        path_points: The list of points along the path to plot.
    """
    x_values = np.linspace(-2, 4, 300)
    x_mesh, y_mesh = np.meshgrid(x_values, x_values)
    plt.imshow(
        ((y_mesh >= -x_mesh + 1) & (y_mesh <= 1) & (x_mesh <= 2) & (y_mesh >= 0)).astype(int),
        extent=(x_mesh.min(), x_mesh.max(), y_mesh.min(), y_mesh.max()),
        origin="lower",
        cmap="Greys",
        alpha=0.3,
    )

    x_plot = np.linspace(0, 4, 2000)
    y1 = -x_plot + 1
    y2 = np.ones(x_plot.size)
    y3 = np.zeros(x_plot.size)

    x_path = [point[0] for point in path_points]
    y_path = [point[1] for point in path_points]
    plt.plot(x_path, y_path, label="Path", color="k", marker=".", linestyle="--")
    plt.scatter(x_path[-1], y_path[-1], label="Final Candidate", color="gold", s=50, zorder=3)

    plt.plot(x_plot, y1, color='b', label="y = -x + 1")
    plt.plot(x_plot, y2, color='g', label="y = 1")
    plt.plot(x_plot, y3, color='r', label="y = 0")
    plt.plot(np.ones(x_plot.size) * 2, x_plot, color='m', label="x = 2")
    
    plt.xlim(0, 3.1)
    plt.ylim(0, 2)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Feasible Region and Path of Linear Program")
    plt.show()
