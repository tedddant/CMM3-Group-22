# Drone optimisation project
simulates a drone hovering to find optimum propeller diameter and motor speed to maximise flight time.

## Setup 
`conda env create -f environment.yml`
`conda activate cmm3`

## Run
`python src/simulate.py`

## Numerical methods:
### Root finding

### ODEs
This script performs a dynamic flight simulation for a quadcopter in a steady hover using an Ordinary Differential Equation (ODE) solver. Its primary function is to accurately predict the maximum achievable flight time based on battery and drone parameters. 

Some Key features include:
-Dynamic Simulation: Uses scipy.integrate.solve_ivp to model continuous energy change, providing a smooth, realistic profile over time.
-Safety Termination: Automatically stops the simulation when the battery reaches the defined minimum safe energy level (E_min), ensuring the calculated flight time is safe and achievable.

The script prints a numerical summary and generates a 5 panel figure plot.

To run Momdify the constants in the script (m_battery, C, D, etc.) for your specific drone configuration.


### Regression and interpolation
To run the Regression and interpolation code, ensure the 'database' folder is present in the same folder as the .py file.
The code should output 3 figures, one showing original data, one showing interpolated data and one showing the original data with regression curves fitted. The code also outputs progress updates and the equations of each regression curve to the terminal.
Note: the names of each CSV relate to the diameter and pitch of the propeller in inches, since the data comes from an american source. E.g. a file with the title 11X7.csv is the test data from an 11 inch diameter propeller with a pitch of 7 inches.

