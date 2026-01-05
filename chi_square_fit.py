import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the Cosmological Model (Hubble Parameter)
# H(z) = H0 * sqrt(Omega_m * (1+z)^3 + Omega_L)
# Assuming Flat Universe: Omega_L = 1 - Omega_m
def hubble_model(z, H0, Omega_m):
    Omega_L = 1 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

# 1. Generate Synthetic Data (mimicking the report's observed data)
np.random.seed(42)
z_obs = np.linspace(0.07, 0.4, 10)  # Redshift z from 0.07 to 0.4
true_H0 = 70.0
true_Omega_m = 0.3
# Add noise to simulate real observations
H_obs = hubble_model(z_obs, true_H0, true_Omega_m) + np.random.normal(0, 5, size=len(z_obs))
H_error = np.ones_like(H_obs) * 5.0  # Error bars

# 2. Chi-Square Fitting
def get_chi_square(params, z, data, errors):
    H0_curr, Om_curr = params
    model_vals = hubble_model(z, H0_curr, Om_curr)
    # Chi-Square Formula: sum((Observed - Expected)^2 / Expected)
    chi2 = np.sum(((data - model_vals) / errors)**2)
    return chi2

# Perform the fit using scipy
popt, pcov = curve_fit(hubble_model, z_obs, H_obs, p0=[68, 0.28], sigma=H_error)
H0_fit, Om_fit = popt

print(f"Best Fit Parameters:\n H0 = {H0_fit:.2f}\n Omega_m = {Om_fit:.2f}")

# 3. Plotting (Recreating the plot from Chapter 4)
plt.figure(figsize=(10, 6))
plt.errorbar(z_obs, H_obs, yerr=H_error, fmt='o', label='Observed data', color='steelblue')

# Plot the best fit line
z_line = np.linspace(0.05, 0.42, 100)
plt.plot(z_line, hubble_model(z_line, *popt), 'r-', label=f'Best fit: $\Omega_m={Om_fit:.2f}$')

plt.xlabel('Redshift z')
plt.ylabel('Hubble parameter H(z)')
plt.title('Chi-Square Fitting of Hubble Parameter')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
