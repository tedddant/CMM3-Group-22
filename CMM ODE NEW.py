#!/usr/bin/env python3
"""
SIMPLIFIED DRONE BATTERY OPTIMIZATION - WINDOWS VERSION
========================================================
FIX: Increased motor power limits to support 3kg drone frame
     No file saving - just displays plots
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================
# PHYSICAL CONSTANTS
# ============================================
g = 9.81                # m/s^2
rho = 1.225             # kg/m^3
P0 = 5.0                # W (baseline electronics power)

# --- Propeller Configuration (10-inch quad) ---
D = 0.254               # m (10 inches diameter)
num_rotors = 4          # Quadcopter
A_single = np.pi * (D/2)**2    # m^2 (single prop area)
A_total = A_single * num_rotors  # m^2 (total rotor area)

# --- Efficiency ---
eta = 0.75              # Overall system efficiency
FM = 0.70               # Figure of merit (rotor efficiency)

# --- Drone ---
m_drone = 3.0           # kg (drone excluding battery) - YOUR ORIGINAL VALUE

# --- Battery ---
V_nom = 21.6            # V (6S LiPo)
energy_density = 250    # Wh/kg 
energy_density_J = energy_density * 3600  # J/kg
discharge_depth = 0.9   # Use 90% of battery

# --- Power Limits ---
# FIXED: Increased to support 3kg drone
P_MAX_CONTINUOUS = 650.0   # W (4 x 162W motors continuous - e.g., 2216 880kv motors)
P_MAX_SURGE = 800.0        # W (surge capability)

# ============================================
# POWER COEFFICIENT
# ============================================
k_theoretical = 1.0 / np.sqrt(2 * rho * A_total)
k_with_losses = k_theoretical / (eta * FM)

print(f"=" * 60)
print("DRONE BATTERY OPTIMIZATION - WINDOWS VERSION")
print(f"=" * 60)
print(f"\nSystem Parameters:")
print(f"  Drone mass: {m_drone} kg (YOUR ORIGINAL VALUE)")
print(f"  Propellers: {num_rotors} × {D*39.37:.0f} inch")
print(f"  Total rotor area: {A_total:.4f} m²")
print(f"  Power coefficient k: {k_with_losses:.4f}")
print(f"  Motor limit: {P_MAX_CONTINUOUS:.0f}W continuous, {P_MAX_SURGE:.0f}W surge")
print(f"  Required motors: ~{P_MAX_CONTINUOUS/4:.0f}W per motor")

def calculate_power(m_total):
    """
    Calculate power required for hover with safety margin.
    P = k * (Weight)^1.5 + P_electronics
    """
    weight = m_total * g
    
    # Basic hover power from momentum theory with losses
    P_hover = k_with_losses * (weight**1.5)
    
    # Add profile drag (12% typical)
    P_with_profile = P_hover * 1.12
    
    # Add control margin (15% for stability)
    P_with_margin = P_with_profile * 1.15
    
    # Add electronics
    P_total = P_with_margin + P0
    
    return P_total

def simulate_flight(C_mAh):
    """
    Simulate flight for given battery capacity in mAh.
    Mass remains constant throughout flight.
    """
    C_Ah = C_mAh / 1000  # Convert to Ah
    
    # Calculate battery mass from capacity
    E_battery_Wh = C_Ah * V_nom
    m_battery = E_battery_Wh / energy_density
    
    # Total mass (constant during flight)
    m_total = m_drone + m_battery
    
    # Calculate required power
    P_required = calculate_power(m_total)
    
    # Check viability
    if P_required > P_MAX_SURGE:
        return {
            'C_mAh': C_mAh,
            'm_battery': m_battery,
            'm_total': m_total,
            'P_required': P_required,
            'flight_time': 0,
            'viable': False
        }
    
    # Battery energy
    E_battery_J = C_Ah * 3600 * V_nom  # Total energy in Joules
    E_min = E_battery_J * (1 - discharge_depth)  # Min safe level
    
    # ODE: dE/dt = -P (constant power draw)
    def dE_dt(t, y):
        return [-P_required]
    
    # Stop at minimum battery level
    def termination_event(t, y):
        return y[0] - E_min
    
    termination_event.terminal = True
    termination_event.direction = -1
    
    # Run simulation
    E0 = E_battery_J
    t_span = (0, 10000)
    
    sol = solve_ivp(
        dE_dt, 
        t_span, 
        [E0], 
        events=termination_event,
        max_step=1.0,
        rtol=1e-6
    )
    
    # Get flight time
    if sol.t_events[0].size > 0:
        flight_time = sol.t_events[0][0]
    else:
        flight_time = 0
    
    return {
        'C_mAh': C_mAh,
        'm_battery': m_battery,
        'm_total': m_total,
        'P_required': P_required,
        'flight_time': flight_time,
        'viable': P_required <= P_MAX_SURGE and flight_time > 0
    }

# ============================================
# OPTIMIZATION LOOP
# ============================================
print(f"\n{'='*60}")
print("BATTERY OPTIMIZATION SWEEP")
print(f"{'='*60}")

# Test range: 2000 to 20000 mAh
C_range_mAh = np.linspace(2000, 20000, 40)
results = []

print(f"\n{'Battery':<10} {'Mass':<10} {'Power':<10} {'Time':<10} {'Status'}")
print("-" * 50)

for C_mAh in C_range_mAh:
    result = simulate_flight(C_mAh)
    results.append(result)
    
    status = "✓" if result['viable'] else "✗"
    print(f"{result['C_mAh']:>7.0f} mAh {result['m_battery']*1000:>7.0f} g "
          f"{result['P_required']:>8.1f} W {result['flight_time']/60:>7.1f} min  {status}")

# Find optimal
viable_results = [r for r in results if r['viable']]

if viable_results:
    flight_times = [r['flight_time'] for r in viable_results]
    optimal_idx = np.argmax(flight_times)
    optimal = viable_results[optimal_idx]
    
    print(f"\n{'='*60}")
    print("OPTIMAL CONFIGURATION FOUND")
    print(f"{'='*60}")
    print(f"\nOptimal Battery: {optimal['C_mAh']:.0f} mAh")
    print(f"   Battery mass: {optimal['m_battery']*1000:.0f} g")
    print(f"   Total mass: {optimal['m_total']:.2f} kg")
    print(f"   Power required: {optimal['P_required']:.1f} W")
    print(f"   Flight time: {optimal['flight_time']/60:.1f} minutes")
    print(f"   C-rate: {optimal['P_required']/(optimal['C_mAh']/1000*V_nom):.2f}C")
    
    # ============================================
    # VISUALIZATION - WITH EXPLANATION
    # ============================================
    print(f"\n{'='*60}")
    print("WHY GRAPHS DROP TO ZERO")
    print(f"{'='*60}")
    print(f"\nNotice the graphs show values, then suddenly drop to ZERO?")
    print(f"This happens when batteries get too large:")
    print(f"  • Larger battery = more mass")
    print(f"  • More mass = more power needed (scales as mass^1.5)")
    print(f"  • When power exceeds {P_MAX_SURGE}W surge limit → NOT VIABLE")
    print(f"  • Non-viable configs have flight_time = 0")
    print(f"\nExample from your results:")
    last_viable = viable_results[-1]
    first_nonviable = next((r for r in results if not r['viable']), None)
    if first_nonviable:
        print(f"  Last viable:  {last_viable['C_mAh']:.0f} mAh → {last_viable['P_required']:.1f}W ✓")
        print(f"  First too big: {first_nonviable['C_mAh']:.0f} mAh → {first_nonviable['P_required']:.1f}W ✗ EXCEEDS LIMIT!")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    C_all = np.array([r['C_mAh'] for r in results])
    time_all = np.array([r['flight_time']/60 for r in results])
    power_all = np.array([r['P_required'] for r in results])
    mass_all = np.array([r['m_total'] for r in results])
    viable_mask = np.array([r['viable'] for r in results])
    
    # 1. Flight time vs capacity
    ax = axes[0, 0]
    ax.plot(C_all[viable_mask], time_all[viable_mask], 'b-', lw=2, label='Viable')
    ax.plot(C_all[~viable_mask], time_all[~viable_mask], 'r--', lw=1, alpha=0.3, label='Too heavy (0 min)')
    ax.plot(optimal['C_mAh'], optimal['flight_time']/60, 'g*', 
            markersize=25, label=f"Optimal: {optimal['C_mAh']:.0f} mAh", zorder=5)
    ax.set_xlabel('Battery Capacity (mAh)', fontsize=11)
    ax.set_ylabel('Flight Time (minutes)', fontsize=11)
    ax.set_title('Flight Time vs Battery Size\n(Drops to zero when power exceeds limit)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=-1)
    
    # 2. Power requirements
    ax = axes[0, 1]
    ax.plot(C_all, power_all, 'orange', lw=2)
    ax.axhline(y=P_MAX_CONTINUOUS, color='yellow', linestyle='--', 
               alpha=0.7, label=f'Continuous: {P_MAX_CONTINUOUS:.0f}W', lw=2)
    ax.axhline(y=P_MAX_SURGE, color='red', linestyle='-', 
               alpha=0.8, label=f'Surge Limit: {P_MAX_SURGE:.0f}W', lw=3)
    ax.plot(optimal['C_mAh'], optimal['P_required'], 'g*', markersize=20, zorder=5)
    
    # Shade the "too heavy" region
    ax.axhspan(P_MAX_SURGE, max(power_all)*1.1, alpha=0.1, color='red', 
               label='Non-viable zone')
    
    ax.set_xlabel('Battery Capacity (mAh)', fontsize=11)
    ax.set_ylabel('Power Required (W)', fontsize=11)
    ax.set_title('Power vs Battery Size\n(Red zone = exceeds motor capability)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Total mass
    ax = axes[1, 0]
    ax.plot(C_all[viable_mask], mass_all[viable_mask], 'g-', lw=2, label='Viable')
    ax.plot(C_all[~viable_mask], mass_all[~viable_mask], 'r--', lw=2, alpha=0.5, label='Too heavy')
    ax.plot(optimal['C_mAh'], optimal['m_total'], 'g*', markersize=20, zorder=5)
    ax.set_xlabel('Battery Capacity (mAh)', fontsize=11)
    ax.set_ylabel('Total Mass (kg)', fontsize=11)
    ax.set_title('Total Drone Mass', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Efficiency (flight time per battery mass) - only for viable
    ax = axes[1, 1]
    battery_mass_g = np.array([r['m_battery']*1000 for r in results])
    efficiency = np.zeros_like(time_all)
    efficiency[viable_mask] = time_all[viable_mask] / (battery_mass_g[viable_mask]/1000)
    
    ax.plot(C_all[viable_mask], efficiency[viable_mask], 'purple', lw=2, label='Viable')
    ax.plot(C_all[~viable_mask], efficiency[~viable_mask], 'r--', lw=1, 
            alpha=0.3, label='Non-viable (0)')
    
    opt_eff = optimal['flight_time']/60 / optimal['m_battery']
    ax.plot(optimal['C_mAh'], opt_eff, 'g*', markersize=20, zorder=5)
    ax.set_xlabel('Battery Capacity (mAh)', fontsize=11)
    ax.set_ylabel('Flight Time / Battery Mass (min/kg)', fontsize=11)
    ax.set_title('Mass Efficiency\n(Zero when config is non-viable)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=-1)
    
    plt.suptitle('Drone Battery Optimization - 3kg Drone with 800W Motors', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print(f"\n Displaying plots... (close window to continue)")
    plt.show()
    
else:
    print("\n No viable configurations found!")
    print("Check motor power limits or drone mass.")

print(f"\n{'='*60}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*60}")
print(f"\n KEY INSIGHT:")
print(f"   Batteries larger than {viable_results[-1]['C_mAh']:.0f} mAh are TOO HEAVY")
print(f"   They make the drone require more than {P_MAX_SURGE:.0f}W")
print(f"   That's why flight time drops to ZERO in the graphs!")
print(f"\n   Power scales as mass^1.5, so weight penalty is severe!")
print(f"{'='*60}\n")