import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional plotting
sns.set_style("darkgrid")

def plot_distributions():
    """
    Generates plots for Normal and Uniform distributions 
    as shown in Chapter 2 of the project report.
    """
    
    # --- 1. Normal Distribution (Figure 2.1) ---
    print("Generating Normal Distribution Plot...")
    # Generate 100,000 random data points from a Normal (Gaussian) distribution
    data_normal = np.random.normal(loc=0, scale=1, size=100000)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data_normal, bins=100, kde=True, color='tan')
    plt.title('Normal Distribution', fontsize=15)
    plt.xlabel('Random Values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

    # --- 2. Uniform Distribution (Figure 2.2) ---
    print("Generating Uniform Distribution Plot...")
    # Generate 100,000 random data points from a Uniform distribution
    data_uniform = np.random.uniform(low=0, high=1, size=100000)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data_uniform, bins=50, kde=False, color='dimgray')
    plt.title('Uniform Distribution', fontsize=15)
    plt.xlabel('Random Numbers', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

if __name__ == "__main__":
    plot_distributions()
