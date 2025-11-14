#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# Constants for Equation

rho = 1.225      # kg/m^3 air density
g = 9.81         # gravity m/s^2
a = 343.0        # speed of sound (m/s)
M_tip_max = 0.7  # tip Mach limit (Estimated based on typical values)
eta_tot = 0.88   # drivetrain efficiency


# Prop coefficients (Currently guessed and will be obtained from interpolation)

def CT(D): return 0.09     # placeholder CT
def CP(D): return 0.018    # placeholder CP


# Masses

m_payload = 1.0   # kg
m_frame   = 2.0   # kg
m_batt    = 0.2   # kg
m0 = m_frame + m_payload  # non-battery mass


# Thrust balance residual function: f(n)=0 at hover

def thrust_residual(n, D, m0, m_batt):
    W = g * (m0 + m_batt)
    return 4 * rho * (n**2) * (D**4) * CT(D) - W


# Bisection method to solve for hover RPM (n in rev/s)

def solve_hover_rpm_bisection(D, m0, m_batt, tol=1e-6, max_iter=200):
    n_low = 0.0
    n_high = (M_tip_max * a) / (np.pi * D)  # tip speed limit (rev/s)

    # Check to see if prop can lift weight
    if thrust_residual(n_high, D, m0, m_batt) < 0:
        return np.nan

    for _ in range(int(max_iter)):
        n_mid = 0.5 * (n_low + n_high)
        if abs(thrust_residual(n_mid, D, m0, m_batt)) < tol:
            return n_mid
        if thrust_residual(n_low, D, m0, m_batt) * thrust_residual(n_mid, D, m0, m_batt) < 0:
            n_high = n_mid
        else:
            n_low = n_mid

    return n_mid  # return last mid if tol not fully met


# Power using CP

def hover_power(D, n):
    P_mech_per = rho * (n**3) * (D**5) * CP(D)
    return 4 * P_mech_per / eta_tot


# Sweep prop diameters and compute hover power

D_vals = np.linspace(0.22, 0.30, 20)
P_vals = []

print("\nD (m)\tHover RPM\tPower (W)")
print("---------------------------------------")

for D in D_vals:
    n = solve_hover_rpm_bisection(D, m0, m_batt)
    if np.isnan(n):
        P_vals.append(np.nan)
        print(f"{D:.2f}\tCANNOT HOVER")
    else:
        P = hover_power(D, n)
        P_vals.append(P)
        print(f"{D:.2f}\t{n*60:.0f} rpm\t{P:.1f} W")


# Plot results
# ------------------------------
plt.figure(figsize=(7,5))
plt.plot(D_vals, P_vals, marker='o')
plt.xlabel("Prop Diameter (m)")
plt.ylabel("Hover Electrical Power (W)")
plt.title("Hover Power vs Prop Diameter (Bisection Root-Finding)")
plt.grid(True)
plt.show()


