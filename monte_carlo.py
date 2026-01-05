import numpy as np
import matplotlib.pyplot as plt

def estimate_pi_sphere_in_cube(n_points=10000):
    """
    Estimates the value of Pi using Monte Carlo simulation (Sphere in a Cube)
    as described in Chapter 3 of the project report.
    """
    print(f"Running Monte Carlo Simulation with {n_points} points...")
    
    # 1. Generate random points in a 3D box (cube) ranging from -1 to 1
    # The cube has a side length of 2, so Volume = 8
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    z = np.random.uniform(-1, 1, n_points)

    # 2. Calculate distance from origin (radius)
    radius_squared = x**2 + y**2 + z**2

    # 3. Count points inside the unit sphere (radius <= 1)
    inside_sphere = radius_squared <= 1
    points_inside = np.sum(inside_sphere)

    # 4. Estimate Pi
    # Volume Sphere / Volume Cube = (4/3 * pi * r^3) / (side^3)
    # With r=1, side=2: Ratio = (4/3 * pi) / 8 = pi / 6
    # Therefore: pi = 6 * (Points_Inside / Total_Points)
    pi_estimate = 6 * (points_inside / n_points)
    
    print(f"Estimated value of Pi: {pi_estimate:.4f}")
    
    # 5. Visualization (2D Cross-section)
    plt.figure(figsize=(8, 8))
    plt.scatter(x[inside_sphere], y[inside_sphere], color='blue', s=1, label='Inside Sphere')
    plt.scatter(x[~inside_sphere], y[~inside_sphere], color='red', s=1, label='Outside Sphere')
    plt.title(f"Monte Carlo Simulation: Sphere in Cube (Pi ≈ {pi_estimate:.4f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    estimate_pi_sphere_in_cube()
