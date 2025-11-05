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

slowflyer = load_data(csv_directory ='/Users/teddant/Documents/uni/year 3 /Computational Methods and Modelling/group project/database/slowflyer/')
print(type(slowflyer))
print (slowflyer['df_11X7'].head()) #check to see if a dataframe is accessible and is formatted correctly

def plot_all (slowflyer):
   
    """
    Plot all DataFrames on CT vs RPM and CP vs RPM plots
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
            ax1.plot(df[rpm_col], df[ct_col], 
                    color=colors[i], alpha=0.5, linewidth=1)
        else:
            print(f"Warning: CT column not found in {df_name}")
        
        # Plot CP vs RPM
        if cp_col in df.columns:
            ax2.scatter(df[rpm_col], df[cp_col], 
                       color=colors[i], label=legend_name, alpha=0.7, s=30)
            ax2.plot(df[rpm_col], df[cp_col], 
                    color=colors[i], alpha=0.5, linewidth=1)
        else:
            print(f"Warning: CP column not found in {df_name}")
    
    # Customize CT vs RPM plot
    ax1.set_xlabel('RPM')
    ax1.set_ylabel('CT')
    ax1.set_title('CT vs RPM for All Propellers')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Customize CP vs RPM plot
    ax2.set_xlabel('RPM')
    ax2.set_ylabel('CP')
    ax2.set_title('CP vs RPM for All Propellers')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return fig

# executing the plotting:
    
if __name__ == "__main__":
    # Load data
    data_dict = load_data(csv_directory='/Users/teddant/Documents/uni/year 3 /Computational Methods and Modelling/group project/database/slowflyer/')
    
    # Print info about loaded DataFrames
    print(f"\nLoaded {len(data_dict)} DataFrames:")
    for name, df in data_dict.items():
        print(f"  {name}: {df.shape}, Columns: {df.columns.tolist()}")
    
    # Plot all DataFrames
    if data_dict:
        plot_all(data_dict)
        
def create_interpolated_dataframes(propeller_dict, rpm_min=None, rpm_max=None, num_points=100):
    """
    Create a new DataFrame with interpolated values for a specific propeller
    
    Parameters:
    - original_df: DataFrame with original data
    - propeller_name: Name of the propeller (for identification)
    - rpm_min: Minimum RPM for interpolation range (defaults to data min)
    - rpm_max: Maximum RPM for interpolation range (defaults to data max)
    - num_points: Number of points to interpolate
    """
    interpolated_dict = {}
    
    for i, (df_name, df)in enumerate(propeller_dict.items()):
            print(f"Processing {df_name}...")
            try:
                    # Extract the relevant columns
                    rpm_original = df['RPM'].values
                    ct_original = df['CT'].values
                    cp_original = df['CP'].values
                    
                    # Determine interpolation range
                    if rpm_min is None:
                        rpm_min = rpm_original.min()
                    if rpm_max is None:
                        rpm_max = rpm_original.max()
                    
                    # Create fine RPM range for interpolation
                    rpm_interp = np.linspace(rpm_min, rpm_max, 50)
                    
                    # Create interpolation functions
                    ct_interp_func = interp1d(rpm_original, ct_original, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    cp_interp_func = interp1d(rpm_original, cp_original, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    
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
               
                    interpolated_name = f"{df_name}_interpolated"
                    interpolated_dict[interpolated_name] = df_interpolated
                    print(f"✅ Successfully created interpolated DataFrame for {df_name} with {len(df_interpolated)} points")
                    
                    
            except Exception as e:
                print(f"❌ Error processing {df_name}: {e}")
    return interpolated_dict

interpolated_data = create_interpolated_dataframes(slowflyer)

#check original dataframes are the same
for name, df in slowflyer.items():
    print(f"\n--- {name} ---")
    print(df.head(3))  # First 3 rows
    
    
# check interpolated dataframes
    print("\n--- INTERPOLATED RESULTS ---")
for name, df in interpolated_data.items():
    print(f"\n{name}:")
    print(df.head())
    print(f"Shape: {df.shape}")