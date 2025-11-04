import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Constants ---
g = 9.81                # m/s^2
rho = 1.225             # kg/m^3
D = 0.10                # m (propeller diameter)
A = np.pi * (D/2)**2    # m^2
e = 0.75                # efficiency
P0 = 5.0                # W baseline power

# --- Drone & battery ---
m_drone = 1.0           # kg (drone excluding battery)
m_battery = 0.2         # kg
C = 4.0                 # Ah
V_nom = 22.2            # V
E_battery = C * 3600 * V_nom   # J
alpha = m_battery / E_battery  # kg/J


k = (g**1.5) / (e * np.sqrt(2 * rho * A))


def dE_dt(t, E):
    m_total = m_drone + alpha * E  # total mass decreases as E drops
    return -(k * m_total**1.5 + P0)


E0 = E_battery
t_span = (0, 5000)  # seconds, large enough window
sol = solve_ivp(dE_dt, t_span, [E0], max_step=0.5, rtol=1e-6)
E = sol.y[0]
t = sol.t

# Trim until energy ~ 0
mask = E > 0
E = E[mask]
t = t[mask]

# Compute instantaneous power
m_total = m_drone + alpha * E
P_hover = k * m_total**1.5 + P0

# --- Plot results ---
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(t, E/1000, 'b', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('Energy (kJ)')
plt.title('Battery Energy vs Time')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(t, P_hover, 'r', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('Power (W)')
plt.title('Instantaneous Power vs Time')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Estimated flight time until energy depletion: {t[-1]:.1f} seconds ({t[-1]/60:.1f} minutes)")




