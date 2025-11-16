import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Constants ---
#Define Parameters and  Constants used in this case
g = 9.81                # m/s^2
rho = 1.225             # kg/m^3
D = 0.10                # m (propeller diameter)
A = np.pi * (D/2)**2    # m^2 (propeller disc area)
e = 0.75                # battery efficiency
P0 = 5.0                # W (baseline power)

# --- Drone & battery ---
m_drone = 2.0           # kg (drone excluding battery)
m_battery = 0.864         # kg
C = 10.0                 # Ah
V_nom = 21.6            # V
E_battery = C * 3600 * V_nom   # J
alpha = m_battery / E_battery  # kg/J

# --- Add safety parameters ---
discharge_depth = 0.9   # Use only 80% of battery for safety
E_min = E_battery * (1 - discharge_depth)  # Minimum safe energy level

k = (g**1.5) / (e * np.sqrt(2 * rho * A)) # Coefficient k calculation

def dE_dt(t, y):
    """Energy depletion ODE with proper array handling"""
    E = y[0]  # Extract energy from state vector
    m_total = m_drone + alpha * E  # total mass decreases as E drops
    power = k * m_total**1.5 + P0
    return [-power]  # Return as array

def termination_event(t, y):
    """Stop simulation when battery reaches minimum safe level"""
    return y[0] - E_min

# Configure termination event
termination_event.terminal = True  # Stop integration when this triggers
termination_event.direction = -1   # Trigger when crossing from positive to negative

# Run simulation with termination event
E0 = E_battery
t_span = (0, 5000)  # seconds, large enough window
sol = solve_ivp(
    dE_dt, 
    t_span, 
    [E0], 
    events=termination_event,  # Add termination event
    max_step=0.5, 
    rtol=1e-6
)

# Extract results
E = sol.y[0]
t = sol.t

# Get actual flight time from termination event
if sol.t_events[0].size > 0:
    flight_time = sol.t_events[0][0]
    print(f"\n{'='*50}")
    print(f"Simulation stopped at battery safety limit")
    print(f"{'='*50}")
else:
    flight_time = t[-1]
    # Trim arrays if no termination (shouldn't happen with proper parameters)
    mask = E > 0
    E = E[mask]
    t = t[mask]

# Compute instantaneous power and other metrics
m_total = m_drone + alpha * E
P_hover = k * m_total**1.5 + P0

# Calculate statistics
energy_used = E0 - E[-1]
energy_used_percent = (energy_used / E0) * 100
avg_power = np.mean(P_hover)
mass_reduction = (m_total[0] - m_total[-1]) * 1000  # in grams

# --- Enhanced plotting ---
fig = plt.figure(figsize=(14, 8))

# Subplot 1: Energy vs Time
plt.subplot(2, 3, 1)
plt.plot(t, E/1000, 'b', lw=2)
plt.axhline(y=E_min/1000, color='r', linestyle='--', alpha=0.5, label='Min Safe Level')
plt.xlabel('Time (s)')
plt.ylabel('Energy (kJ)')
plt.title('Battery Energy vs Time')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Power vs Time
plt.subplot(2, 3, 2)
plt.plot(t, P_hover, 'r', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('Power (W)')
plt.title('Instantaneous Power vs Time')
plt.grid(True, alpha=0.3)

# Subplot 3: Mass vs Time
plt.subplot(2, 3, 3)
plt.plot(t, m_total*1000, 'g', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('Total Mass (g)')
plt.title('Mass Reduction During Flight')
plt.grid(True, alpha=0.3)

# Subplot 4: Energy Depletion Rate
plt.subplot(2, 3, 4)
dE_dt_values = -P_hover  # Energy depletion rate
plt.plot(t, dE_dt_values, 'orange', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('dE/dt (W)')
plt.title('Energy Depletion Rate')
plt.grid(True, alpha=0.3)

# Subplot 5: State of Charge
plt.subplot(2, 3, 5)
soc = (E / E0) * 100  # State of charge in percentage
plt.plot(t, soc, 'purple', lw=2)
plt.axhline(y=(1-discharge_depth)*100, color='r', linestyle='--', alpha=0.5, label='Min SOC')
plt.xlabel('Time (s)')
plt.ylabel('State of Charge (%)')
plt.title('Battery SOC vs Time')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 6: Current Draw (estimated)
plt.subplot(2, 3, 6)
I_draw = P_hover / V_nom  # Simplified current estimate
plt.plot(t, I_draw, 'cyan', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.title('Estimated Current Draw')
plt.grid(True, alpha=0.3)

plt.suptitle('Drone Energy Model - Enhanced Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Print comprehensive results ---
print(f"\n{'='*50}")
print(f"FLIGHT SIMULATION RESULTS")
print(f"{'='*50}")
print(f"Flight Time:")
print(f"  - Total: {flight_time:.1f} seconds")
print(f"  - Minutes: {flight_time/60:.1f} minutes")
print(f"  - Hours: {flight_time/3600:.2f} hours")

print(f"\nEnergy Analysis:")
print(f"  - Initial energy: {E0/1000:.2f} kJ")
print(f"  - Final energy: {E[-1]/1000:.2f} kJ")
print(f"  - Energy consumed: {energy_used/1000:.2f} kJ ({energy_used_percent:.1f}%)")
print(f"  - Reserve remaining: {E[-1]/1000:.2f} kJ ({(E[-1]/E0)*100:.1f}%)")

print(f"\nPower Analysis:")
print(f"  - Initial power: {P_hover[0]:.1f} W")
print(f"  - Final power: {P_hover[-1]:.1f} W")
print(f"  - Average power: {avg_power:.1f} W")
print(f"  - Power reduction: {P_hover[0] - P_hover[-1]:.1f} W ({(1-P_hover[-1]/P_hover[0])*100:.1f}%)")

print(f"\nMass Analysis:")
print(f"  - Initial total mass: {m_total[0]*1000:.1f} g")
print(f"  - Final total mass: {m_total[-1]*1000:.1f} g")
print(f"  - Mass reduction: {mass_reduction:.1f} g")
print(f"  - Mass reduction %: {(mass_reduction/(m_total[0]*1000))*100:.1f}%")

print(f"\nCurrent Analysis:")
print(f"  - Average current: {np.mean(I_draw):.2f} A")
print(f"  - Peak current: {np.max(I_draw):.2f} A")
print(f"  - C-rate: {np.mean(I_draw)/C:.2f}C")

print(f"\nSafety Parameters:")
print(f"  - Discharge depth limit: {discharge_depth*100:.0f}%")
print(f"  - Min safe energy: {E_min/1000:.2f} kJ")
print(f"  - Safety margin: {(1-discharge_depth)*100:.0f}%")

# --- Additional analysis ---
# Energy efficiency metric
hover_work = m_drone * g * flight_time  # Work done hovering
efficiency = hover_work / energy_used
print(f"\nEfficiency Metrics:")
print(f"  - Hover work done: {hover_work/1000:.1f} kJ")
print(f"  - Energy efficiency: {efficiency*100:.1f}%")
print(f"  - Thrust-to-power: {(m_drone*g)/avg_power:.3f} N/W")

print(f"\n{'='*50}")
print("SIMULATION COMPLETE")

print(f"{'='*50}\n")

