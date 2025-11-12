#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:11:57 2025

@author: teddant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d

# Enable interactive plotting mode so multiple plots can appear without blocking
plt.ion()

def load_data (csv_files=None, csv_directory=None):
    # Get CSV files
    if csv_files is None and csv_directory is None:
        csv_directory = '.'  # Current directory if nothing specified
    
    if csv_files is None:
        if csv_directory:
            csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    
    if not csv_files:
        print("No CSV files found!")
    
    dataframes = {} #create dictionary of dataframes
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files):
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Extract filename 
            filename = os.path.basename(csv_file)
            df_name = os.path.splitext(filename)[0] # remove extension
            df_name = 'df_' + df_name
            # Add to dictionary
            dataframes [df_name] = df
            print(f'Created DataFrame: {df_name}')
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    return dataframes

slowflyer = load_data(csv_directory='database/slowflyer/')
print(type(slowflyer))
print(slowflyer['df_11X7'].head()) #check to see if a dataframe is accessible and is formatted correctly

def plot_all(data_dict, title_suffix="", regression_results=None, show_lines=True):
   
    """
    Plot all DataFrames on CT vs RPM and CP vs RPM plots
    
    Parameters:
    - data_dict: Dictionary of DataFrames
    - title_suffix: String to add to plot titles
    - regression_results: Optional dictionary of regression results to plot fitted lines
    - show_lines: Whether to show lines connecting data points (default True)
    """
    if not data_dict:
        print("No data to plot!")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors for different propellers
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    # Plot each DataFrame
    for i, (df_name, df) in enumerate(data_dict.items()):
        # Clean the name for legend (remove 'df_' prefix)
        legend_name = df_name.replace('df_', '')
        
        # Determine column names (handle case sensitivity)
        rpm_col = 'RPM' if 'RPM' in df.columns else 'rpm'
        ct_col = 'CT' if 'CT' in df.columns else 'ct'
        cp_col = 'CP' if 'CP' in df.columns else 'cp'
        
        # Check if required columns exist
        if rpm_col not in df.columns:
            print(f"Warning: RPM column not found in {df_name}")
            continue
        
        # Plot CT vs RPM
        if ct_col in df.columns:
            ax1.scatter(df[rpm_col], df[ct_col], 
                       color=colors[i], label=legend_name, alpha=0.7, s=30)
            if show_lines:
                ax1.plot(df[rpm_col], df[ct_col], 
                        color=colors[i], alpha=0.5, linewidth=1)
        else:
            print(f"Warning: CT column not found in {df_name}")
        
        # Plot CP vs RPM
        if cp_col in df.columns:
            ax2.scatter(df[rpm_col], df[cp_col], 
                       color=colors[i], label=legend_name, alpha=0.7, s=30)
            if show_lines:
                ax2.plot(df[rpm_col], df[cp_col], 
                        color=colors[i], alpha=0.5, linewidth=1)
        else:
            print(f"Warning: CP column not found in {df_name}")
        
        # Plot fitted regression lines if available
        if regression_results and df_name in regression_results:
            rpm_range = np.linspace(df[rpm_col].min(), df[rpm_col].max(), 100)
            
            # Plot fitted CT line
            if 'CT' in regression_results[df_name]:
                ct_poly = regression_results[df_name]['CT']['poly']
                ct_fit = ct_poly(rpm_range)
                ax1.plot(rpm_range, ct_fit, 
                        color=colors[i], linewidth=2.5, linestyle='--', 
                        label=f"{legend_name} (fit)", alpha=0.8)
            
            # Plot fitted CP line
            if 'CP' in regression_results[df_name]:
                cp_poly = regression_results[df_name]['CP']['poly']
                cp_fit = cp_poly(rpm_range)
                ax2.plot(rpm_range, cp_fit, 
                        color=colors[i], linewidth=2.5, linestyle='--', 
                        label=f"{legend_name} (fit)", alpha=0.8)
    
    # Customize CT vs RPM plot
    ax1.set_xlabel('RPM')
    ax1.set_ylabel('CT')
    ax1.set_title(f'CT vs RPM for All Propellers {title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Customize CP vs RPM plot
    ax2.set_xlabel('RPM')
    ax2.set_ylabel('CP')
    ax2.set_title(f'CP vs RPM for All Propellers {title_suffix}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_interpolated_dataframes(propeller_dict, rpm_min=None, rpm_max=None, num_points=100):
    """
    Create a new DataFrame with interpolated values for a specific propeller
    
    Parameters:
    - propeller_dict: Dictionary of original DataFrames
    - rpm_min: Minimum RPM for interpolation range (defaults to data min)
    - rpm_max: Maximum RPM for interpolation range (defaults to data max)
    - num_points: Number of points to interpolate
    """
    interpolated_dict = {}
    
    for i, (df_name, df) in enumerate(propeller_dict.items()):
        print(f"Processing {df_name}...")
        try:
            # Extract the relevant columns
            rpm_original = df['RPM'].values
            ct_original = df['CT'].values
            cp_original = df['CP'].values
            
            # Determine interpolation range for THIS propeller (not global)
            current_rpm_min = rpm_original.min()
            current_rpm_max = rpm_original.max()
            
            # Create fine RPM range for interpolation (based on THIS propeller's data)
            rpm_interp = np.linspace(current_rpm_min, current_rpm_max, 50)
            
            # Create interpolation functions
            ct_interp_func = interp1d(rpm_original, ct_original, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            cp_interp_func = interp1d(rpm_original, cp_original, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            
            # Calculate interpolated values
            ct_interp = ct_interp_func(rpm_interp)
            cp_interp = cp_interp_func(rpm_interp)
            
            # Create new DataFrame
            df_interpolated = pd.DataFrame({
                'RPM': rpm_interp,
                'CT': ct_interp,
                'CP': cp_interp,
            })
            
            # Add calculated columns
            df_interpolated['n_RPS'] = df_interpolated['RPM'] / 60
            df_interpolated['efficiency'] = df_interpolated['CT'] / df_interpolated['CP']
            
            # Store in dictionary with same naming convention as original data (df_*)
            # Remove '_interpolated' suffix so plot_all() can work with it
            interpolated_dict[df_name] = df_interpolated
            print(f"✅ Successfully created interpolated DataFrame for {df_name} with {len(df_interpolated)} points")
            
        except Exception as e:
            print(f"❌ Error processing {df_name}: {e}")
    return interpolated_dict


def perform_regression(data_dict, poly_degree=2):
    """
    Perform polynomial regression on each propeller's data
    
    Parameters:
    - data_dict: Dictionary of DataFrames with RPM, CT, CP columns
    - poly_degree: Degree of polynomial (2=quadratic, 3=cubic, etc.)
    
    Returns:
    - regression_results: Dictionary with coefficients and R² values for each propeller
    """
    regression_results = {}
    
    for df_name, df in data_dict.items():
        print(f"\nPerforming regression on {df_name}...")
        try:
            # Extract RPM and performance columns
            rpm = df['RPM'].values
            ct = df['CT'].values
            cp = df['CP'].values
            
            # Fit polynomial for CT vs RPM
            ct_coeffs = np.polyfit(rpm, ct, poly_degree)
            ct_poly = np.poly1d(ct_coeffs)
            ct_fit = ct_poly(rpm)
            ct_r2 = 1 - (np.sum((ct - ct_fit)**2) / np.sum((ct - np.mean(ct))**2))
            
            # Fit polynomial for CP vs RPM
            cp_coeffs = np.polyfit(rpm, cp, poly_degree)
            cp_poly = np.poly1d(cp_coeffs)
            cp_fit = cp_poly(rpm)
            cp_r2 = 1 - (np.sum((cp - cp_fit)**2) / np.sum((cp - np.mean(cp))**2))
            
            # Store results
            regression_results[df_name] = {
                'CT': {
                    'coefficients': ct_coeffs,
                    'poly': ct_poly,
                    'r_squared': ct_r2,
                    'equation': _format_polynomial(ct_coeffs, 'RPM')
                },
                'CP': {
                    'coefficients': cp_coeffs,
                    'poly': cp_poly,
                    'r_squared': cp_r2,
                    'equation': _format_polynomial(cp_coeffs, 'RPM')
                }
            }
            
            print(f"✅ {df_name}:")
            print(f"   CT equation: {regression_results[df_name]['CT']['equation']}")
            print(f"   CT R²: {ct_r2:.6f}")
            print(f"   CP equation: {regression_results[df_name]['CP']['equation']}")
            print(f"   CP R²: {cp_r2:.6f}")
            
        except Exception as e:
            print(f"❌ Error performing regression on {df_name}: {e}")
    
    return regression_results


def _format_polynomial(coeffs, var_name='x'):
    """
    Format polynomial coefficients into a readable equation string
    
    Parameters:
    - coeffs: Numpy array of polynomial coefficients (highest degree first)
    - var_name: Name of the variable (default 'x')
    
    Returns:
    - Formatted equation string
    """
    terms = []
    degree = len(coeffs) - 1
    
    for i, coeff in enumerate(coeffs):
        power = degree - i
        
        if abs(coeff) < 1e-10:  # Skip near-zero coefficients
            continue
        
        # Format coefficient
        if power == 0:
            term = f"{coeff:.6f}"
        elif power == 1:
            term = f"{coeff:.6f}*{var_name}"
        else:
            term = f"{coeff:.6f}*{var_name}^{power}"
        
        terms.append(term)
    
    if not terms:
        return "0"
    
    # Join with + or -
    equation = terms[0]
    for term in terms[1:]:
        if term[0] == '-':
            equation += f" {term}"
        else:
            equation += f" + {term}"
    
    return equation

# executing the plotting:

# Plot original data
print("\n=== Plotting Original Data ===")
plot_all(slowflyer, "(Original Data)")

# Create interpolated dataframes
print("\n=== Creating Interpolated Data ===")
interpolated_dict = create_interpolated_dataframes(slowflyer)

# Plot interpolated data
print("\n=== Plotting Interpolated Data ===")
plot_all(interpolated_dict, "(Interpolated Data)")

# Perform polynomial regression on original data
print("\n=== Performing Polynomial Regression ===")
regression_results = perform_regression(slowflyer, poly_degree=2)

# Plot original data WITH fitted regression lines
print("\n=== Plotting Original Data with Regression Fits ===")
plot_all(slowflyer, "(with Regression Fits)", regression_results=regression_results, show_lines=False)

# Keep plots open and visible
print("\nPlots displayed! Close the windows to exit.")
plt.show(block=True)