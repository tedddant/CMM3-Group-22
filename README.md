# Drone optimisation project
simulates a drone hovering to find optimum propeller diameter and motor speed to maximise flight time.

## Setup 
`conda env create -f environment.yml`
`conda activate cmm3`

## Run
`python src/simulate.py`

## Numerical methods:
### Root finding

This script determines the required motor speed (RPM) for the drone to hover by solving the thrust–weight balance equation using numerical root finding. Both the Bisection and Newton–Raphson methods are implemented to ensure accuracy and solver independence.

The code outputs:
- A table of hover RPM needed for different propeller diameters and its relevant plot  
- A sensitivity analysis of the two root finding methods and a graphical output  
- A convergence comparison between the two methods and a graphical output  

The results are used to select the optimal propeller diameter for the final drone design. Values for $C_T$ and $C_P$ were obtained from regression/interpolation and can be modified along with physical variables such as the masses and propeller diameters for which root finding is carried out.


### ODEs
This script performs a dynamic flight simulation for a quadcopter in a steady hover using an Ordinary Differential Equation (ODE) solver. Its primary function is to accurately predict the maximum achievable flight time based on battery and drone parameters. 

Some Key features include:
- Dynamic Simulation: Uses scipy.integrate.solve_ivp to model continuous energy change, providing a smooth, realistic profile over time.
- Optimization Sweep: The core of the script performs a sweep across varying battery capacities ($\text{mAh}$) to find the optimal size that maximizes flight endurance.
- Critical Constraint: It includes a Maximum Surge Power limit, which acts as a hard physical constraint, causing flight time to abruptly drop to zero for excessively heavy (oversized) batteries.
- Safety Termination: Automatically stops the simulation when the battery reaches the defined minimum safe energy level, ensuring the calculated flight time is safe and achievable.

The script prints a numerical summary of the optimal configuration and generates a plot showing the energy-vs-weight trade-off curve

To run momdify the constants in the script (m_battery, C, D, P_MAX_SURGE, etc.) for your specific drone configuration. It currely uses constraints from the Root Finding system.


### Regression and interpolation
To run the Regression and interpolation code, ensure the 'database' folder is present in the same folder as the .py file.
The code should output 3 figures, one showing original data, one showing interpolated data and one showing the original data with regression curves fitted. The code also outputs progress updates and the equations of each regression curve to the terminal.
Note: the names of each CSV relate to the diameter and pitch of the propeller in inches, since the data comes from an american source. E.g. a file with the title 11X7.csv is the test data from an 11 inch diameter propeller with a pitch of 7 inches.

