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

### Regression and interpolation
To run the Regression and interpolation code, ensure the 'database' folder is present in the same folder as the .py file.
The code should output 3 figures, one showing original data, one showing interpolated data and one showing the original data with regression curves fitted. The code also outputs progress updates and the equations of each regression curve to the terminal.
Note: the names of each CSV relate to the diameter and pitch of the propeller in inches, since the data comes from an american source. E.g. a file with the title 11X7.csv is the test data from an 11 inch diameter propeller with a pitch of 7 inches.

