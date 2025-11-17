#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True  


# Constants for equation

rho = 1.225      # kg/m^3 air density (standard value)
g = 9.81         # gravity m/s^2
a = 343.0        # speed of sound (m/s)
M_tip_max = 0.7  # max tip Mach limit
eta_tot = 0.88   # drivetrain efficiency 
# eta_tot converts Pmech at Prop disk to Pelec drawn from battery


# Masses

m_payload = 1.0   # kg
m_frame   = 2.0   # kg
m_batt    = 0.20  # kg
m0 = m_frame + m_payload #The total mass used in the hover equation is m0 +mbatt

# Placeholder CT/CP values 

def CT(D): return 0.09
def CP(D): return 0.018

# Residual: thrust - weight = 0
#The following code defines the hover condition code as outlined in the report
def thrust_residual(n, D, m0, m_batt):
    W = g * (m0 + m_batt)
    return 4 * rho * (n**2) * (D**4) * CT(D) - W
#This is the equation to be solved by NR & Bisection

# Bisection Method

def solve_hover_rpm_bisection(D, m0, m_batt, tol=1e-6, max_iter=200):
#Initial Brackets for RPM    
    n_low = 0.0 # No rotation
    n_high = (M_tip_max * a) / (np.pi * D) # Set by Mach constraint
# So Tip speed
    if thrust_residual(n_high, D, m0, m_batt) < 0:
        return np.nan, None  # Cannot hover

    residuals = []
    for _ in range(max_iter):
        n_mid = 0.5 * (n_low + n_high) # Compute Midpoint
        res = thrust_residual(n_mid, D, m0, m_batt)
        residuals.append(abs(res)) #Store for plotting convergence

        if abs(res) < tol:
            return n_mid, residuals # When residual small enough root is found

        if thrust_residual(n_low, D, m0, m_batt) * res < 0:
            n_high = n_mid
        else:
            n_low = n_mid
        #Applying sign-change method for Bisection
    return n_mid, residuals


# Newton-Raphson Method

def solve_hover_rpm_newton(D, m0, m_batt, n0=2000, tol=1e-6, max_iter=50):
    n = n0 # Initial Guess
    residuals = []

    for _ in range(max_iter):
        f = thrust_residual(n, D, m0, m_batt)
        df = 8 * rho * n * (D**4) * CT(D)  # derivative of Hover Equation

        residuals.append(abs(f)) #Stored for convergence plot

        if abs(f) < tol:
            return n, residuals #Stop when residual small

        n = n - f/df #NR Update

    return n, residuals
    #Loop ends when tolerance met or max iterations hit

# Hover Power Calculation

def hover_power(D, n):
    P_mech_per = rho * (n**3) * (D**5) * CP(D) #Mech P of one Prop
    return 4 * P_mech_per / eta_tot #Electrical power drawn


# Power vs Diameter Sweep

D_vals = np.linspace(0.22, 0.30, 8)
P_vals = []

print("\nHover RPM + Power Results")
print("D (m)\tHover RPM\tPower (W)")
print("---------------------------------------")
# To solve for Hover RPM, Power and tabulate results
for D in D_vals:
    n_bis, _ = solve_hover_rpm_bisection(D, m0, m_batt)
    if np.isnan(n_bis):
        P_vals.append(np.nan)
        print(f"{D:.2f}\tCANNOT HOVER")
    else:
        P = hover_power(D, n_bis)
        P_vals.append(P)
        print(f"{D:.2f}\t{n_bis*60:.2f}\t{P:.2f}")

# Plot Hover Power curve
plt.figure(figsize=(7,5))
plt.plot(D_vals, P_vals, marker='o')
plt.xlabel("Prop Diameter (m)")
plt.ylabel("Hover Electrical Power (W)")
plt.title("Hover Power vs Prop Diameter")
plt.grid(True)


# Convergence Comparison at D_test

D_test = 0.26
n_bis, bis_res = solve_hover_rpm_bisection(D_test, m0, m_batt)
n_new, new_res = solve_hover_rpm_newton(D_test, m0, m_batt, n0=1000)

# Combined Convergence Plot
plt.figure(figsize=(8,6))
plt.semilogy(bis_res, marker='o', label="Bisection Method")
plt.semilogy(new_res, marker='s', color='darkorange', label="Newton–Raphson Method")
plt.xlabel("Iteration")
plt.ylabel("|Residual| (N)")
plt.title(f"Convergence Comparison (D = {D_test} m)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()

# Print Comparison
print("\nComparison at D = 0.26 m")
print("---------------------------------------")
print(f"Bisection:       {n_bis*60:.3f} RPM ({len(bis_res)} iterations)")
print(f"Newton–Raphson:  {n_new*60:.3f} RPM ({len(new_res)} iterations)")
print(f"Difference:      {abs(n_bis-n_new)/n_bis*100:.5f}%")


# Sensitivity Check

diff_list = []
print("\nSensitivity Check Across Diameters")
print("D (m)\tDifference (%)")

for D in D_vals:
    nB, _ = solve_hover_rpm_bisection(D, m0, m_batt)
    nN, _ = solve_hover_rpm_newton(D, m0, m_batt)
    diff = abs(nB-nN)/nB * 100
    diff_list.append(diff)
    print(f"{D:.2f}\t{diff:.5f}")

plt.figure(figsize=(7,5))
plt.plot(D_vals, diff_list, marker='o')
plt.xlabel("Prop Diameter (m)")
plt.ylabel("Difference (%)")
plt.title("Root Method Sensitivity Across Prop Diameters")
plt.grid(True)


# Final show - ALL plots Appear

plt.show()








