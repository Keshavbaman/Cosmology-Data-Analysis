import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. Define the Gaussian Log-Likelihood Function ---
# As described in Chapter 5 of the report: ln L = -0.5 * sum((y - model)^2 / sigma^2)
def neg_log_likelihood(params, x, y, yerr):
    m, c = params
    model = m * x + c
    sigma2 = yerr**2
    return 0.5 * np.sum((y - model)**2 / sigma2 + np.log(2 * np.pi * sigma2))

# --- 2. Generate Synthetic Data (Linear Model y = mx + c) ---
np.random.seed(42)
true_m = 2.5
true_c = 1.0
N = 50
x = np.linspace(0, 10, N)
yerr = 1.5 + 0.5 * np.random.random(N)
y = true_m * x + true_c + np.random.normal(0, yerr)

# --- 3. Minimize the Negative Log-Likelihood (MLE) ---
initial_guess = [1.0, 0.0]
result = minimize(neg_log_likelihood, initial_guess, args=(x, y, yerr))
m_mle, c_mle = result.x

print(f"True Parameters: m={true_m}, c={true_c}")
print(f"MLE Parameters:  m={m_mle:.3f}, c={c_mle:.3f}")

# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=3, label="Observed Data")
plt.plot(x, true_m * x + true_c, "k", alpha=0.3, lw=3, label="True Model")
plt.plot(x, m_mle * x + c_mle, ":r", label="MLE Best Fit")
plt.legend(fontsize=12)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Maximum Likelihood Estimation (MLE)", fontsize=14)
plt.show()
