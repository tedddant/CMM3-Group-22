#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:31:07 2025

@author: teddant
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_rpm_vs_ct_cp(csv_files=None, csv_directory=None):
    """
    Plot RPM vs CT and RPM vs CP from multiple CSV files
    
    Parameters:
    csv_files: list of specific CSV file paths to plot
    csv_directory: directory path containing CSV files (will plot all CSVs in directory)
    """
    
    # Get CSV files
    if csv_files is None and csv_directory is None:
        csv_directory = '.'  # Current directory if nothing specified
    
    if csv_files is None:
        if csv_directory:
            csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors for different lines
    colors = plt.cm.tab10(range(len(csv_files)))
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files):
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Extract filename for legend
            filename = os.path.basename(csv_file)
            
            # Check if required columns exist
            if 'RPM' not in df.columns:
                print(f"Warning: 'rpm' column not found in {filename}")
                continue
            
            # Plot RPM vs CT (if CT column exists)
            if 'ct' in df.columns:
                ax1.scatter(df['RPM'], df['ct'], 
                        color=colors[i], label=filename, s = 10)
            elif 'CT' in df.columns:
                ax1.scatter(df['RPM'], df['CT'], 
                        color=colors[i], label=filename, s = 10)
            else:
                print(f"Warning: CT column not found in {filename}")
            
            # Plot RPM vs CP (if CP column exists)
            if 'cp' in df.columns:
                ax2.scatter(df['RPM'], df['cp'], 
                        color=colors[i], label=filename, s = 10)
            elif 'CP' in df.columns:
                ax2.scatter(df['RPM'], df['CP'], 
                        color=colors[i], label=filename, s = 10)
            else:
                print(f"Warning: CP column not found in {filename}")
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Customize RPM vs CT plot
    ax1.set_xlabel('RPM')
    ax1.set_ylabel('CT')
    ax1.set_title('RPM vs CT')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Customize RPM vs CP plot
    ax2.set_xlabel('RPM')
    ax2.set_ylabel('CP')
    ax2.set_title('RPM vs CP')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return fig

# Usage examples:

# Option 1: Plot all CSV files in current directory
# plot_rpm_vs_ct_cp(csv_directory='/database/slowflyer/.')

# Option 2: Plot specific CSV files
# plot_rpm_vs_ct_cp(csv_files=['data1.csv', 'data2.csv', 'data3.csv'])

# Option 3: Plot all CSV files in a specific directory
plot_rpm_vs_ct_cp(csv_directory ='/Users/teddant/Documents/uni/year 3 /Computational Methods and Modelling/group project/database/slowflyer/')