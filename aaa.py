import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D # For 3D plots
from scipy.stats import linregress # Still useful for underlying concepts, even if not directly plotted in 3D

# --- Configuration ---
# Adjust these paths to where your CSV files are located
TEST_DIR_PATH = "015_test_015_FastSlow_Plastic_Kdesign2/data/csv" # Replace with the actual test directory name
RED_CSV_PATH = f"{TEST_DIR_PATH}/red_intensity_timeseries.csv"
GREEN_CSV_PATH = f"{TEST_DIR_PATH}/green_intensity_timeseries.csv"
BLUE_CSV_PATH = f"{TEST_DIR_PATH}/blue_intensity_timeseries.csv"

# --- Function to load and prepare data ---
def load_and_prepare_data(file_path):
    """Loads a CSV, extracts pathlength columns, and calculates I0."""
    df = pd.read_csv(file_path)

    # Get pathlength columns (excluding 'elapsed_seconds')
    pathlength_cols = [col for col in df.columns if col.startswith('pathlength_')]

    # Convert pathlength column values from string to float
    # We will use these for plotting
    pathlength_values = np.array([float(col.replace('pathlength_', '')) for col in pathlength_cols])

    # I0 is the first row's intensity for each pathlength
    # Make sure to handle potential NaNs if the first row could have them
    I0 = df.iloc[0][pathlength_cols].values

    return df, pathlength_cols, pathlength_values, I0

# --- Load data for all colors ---
print(f"Loading data from: {RED_CSV_PATH}")
red_df, red_pathlength_cols, red_pathlength_values, I0_red = load_and_prepare_data(RED_CSV_PATH)
print(f"Loading data from: {GREEN_CSV_PATH}")
green_df, green_pathlength_cols, green_pathlength_values, I0_green = load_and_prepare_data(GREEN_CSV_PATH)
print(f"Loading data from: {BLUE_CSV_PATH}")
blue_df, blue_pathlength_cols, blue_pathlength_values, I0_blue = load_and_prepare_data(BLUE_CSV_PATH)

print("\nData Loading Complete. Starting Analysis...")

# --- 1. 3D Plot of Pathlength, Intensity, and Time ---

def create_3d_plot(df, pathlength_cols, pathlength_values, color_name):
    """Creates a 3D plot for a given color channel."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for 3D plot
    # Create meshgrid for pathlengths (X) and elapsed_seconds (Y)
    X, Y = np.meshgrid(pathlength_values, df['elapsed_seconds'].values)

    # Z will be the intensity values
    Z = df[pathlength_cols].values

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Path Length (mm)')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_zlabel('Intensity')
    ax.set_title(f'{color_name} Intensity: Path Length vs. Time')
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Intensity')
    plt.tight_layout()

print("\nGenerating 3D plots...")
create_3d_plot(red_df, red_pathlength_cols, red_pathlength_values, 'Red')
create_3d_plot(green_df, green_pathlength_cols, green_pathlength_values, 'Green')
create_3d_plot(blue_df, blue_pathlength_cols, blue_pathlength_values, 'Blue')
plt.show()

# --- 2. Plots of Column Lengths with Most Variance in Intensity Over Time ---

def plot_most_variance_column(df, pathlength_cols, color_name):
    """
    Identifies the pathlength column with the most variance over time
    and plots its intensity vs. time.
    """
    # Calculate variance for each pathlength column over time
    # Skip 'elapsed_seconds' column
    intensity_variances = df[pathlength_cols].var()

    # Find the column name with the maximum variance
    most_variant_column = intensity_variances.idxmax()
    max_variance_value = intensity_variances.max()

    print(f"\n{color_name}: Column with most variance is '{most_variant_column}' (Variance: {max_variance_value:.2f})")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='elapsed_seconds', y=most_variant_column, data=df)
    plt.title(f'{color_name} Intensity over Time for Most Variant Path Length ({float(most_variant_column.replace("pathlength_", "")):.2f} mm)')
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.tight_layout()

print("\nGenerating plots for most variant path lengths...")
plot_most_variance_column(red_df, red_pathlength_cols, 'Red')
plot_most_variance_column(green_df, green_pathlength_cols, 'Green')
plot_most_variance_column(blue_df, blue_pathlength_cols, 'Blue')
plt.show()

# --- (Optional) Beer-Lambert Law Application for Absorption Coefficient Time Series ---
# It calculates the absorption coefficient over time for each color.

print("\nCalculating Absorption Coefficients (Beer-Lambert Law)...")
def calculate_absorption_coefficients(df, pathlength_cols, pathlength_values, I0_values, color_name):
    alpha_time_series = []
    elapsed_times = df['elapsed_seconds'].values

    for i in range(len(df)):
        current_I = df.iloc[i][pathlength_cols].values

        # Calculate ln(I/I0). Handle cases where I or I0 might be zero or negative by setting to NaN
        # Also, handle cases where I0 is zero to prevent division by zero
        log_ratio = np.full_like(current_I, np.nan, dtype=float)
        # Only calculate if I0 is not zero and I is not zero/negative
        valid_log_calc_indices = (I0_values > 0) & (current_I > 0)
        log_ratio[valid_log_calc_indices] = np.log(current_I[valid_log_calc_indices] / I0_values[valid_log_calc_indices])


        # Filter out NaNs for regression
        # A valid point for regression needs a valid log_ratio and a valid pathlength
        valid_indices_for_fit = ~np.isnan(log_ratio) & ~np.isnan(pathlength_values)

        if np.sum(valid_indices_for_fit) >= 2: # Need at least 2 points for a linear fit
            slope, intercept, r_value, p_value, std_err = linregress(
                pathlength_values[valid_indices_for_fit], log_ratio[valid_indices_for_fit]
            )
            alpha_time_series.append(-slope) # alpha = -slope from ln(I/I0) = -alpha * b
        else:
            alpha_time_series.append(np.nan) # Not enough valid points for a fit

    print(f"  {color_name} absorption coefficient calculation complete.")
    return pd.Series(alpha_time_series, index=elapsed_times, name=f'{color_name}_alpha')

red_alpha_series = calculate_absorption_coefficients(red_df, red_pathlength_cols, red_pathlength_values, I0_red, 'Red')
green_alpha_series = calculate_absorption_coefficients(green_df, green_pathlength_cols, green_pathlength_values, I0_green, 'Green')
blue_alpha_series = calculate_absorption_coefficients(blue_df, blue_pathlength_cols, blue_pathlength_values, I0_blue, 'Blue')

# Plotting the absorption coefficients over time
plt.figure(figsize=(10, 6))
plt.plot(red_alpha_series.index, red_alpha_series.values, label='Red Absorption Coefficient')
plt.plot(green_alpha_series.index, green_alpha_series.values, label='Green Absorption Coefficient')
plt.plot(blue_alpha_series.index, blue_alpha_series.values, label='Blue Absorption Coefficient')
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Absorption Coefficient (alpha)')
plt.title('Calculated Absorption Coefficients Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def plot_absorption_most_variant(red_df, green_df, blue_df, red_I0, green_I0, blue_I0):
    """
    Plots absorption ln(I/I0) over time for the most variant path length of each color.
    Includes path length information in legend and prints details.
    """
    plt.figure(figsize=(10, 6))
    
    # Process each color
    for df, I0, color, name in [(red_df, red_I0, 'red', 'Red'), 
                               (green_df, green_I0, 'green', 'Green'), 
                               (blue_df, blue_I0, 'blue', 'Blue')]:
        
        # Find most variant column
        pathlength_cols = [col for col in df.columns if col.startswith('pathlength_')]
        intensity_variances = df[pathlength_cols].var()
        most_variant_col = intensity_variances.idxmax()
        path_length = float(most_variant_col.replace('pathlength_', ''))
        
        # Calculate absorption ln(I/I0)
        I = df[most_variant_col].values
        I0_val = I0[pathlength_cols.index(most_variant_col)]
        absorption = np.log10(I0_val/I)
        
        # Plot with path length in legend
        plt.plot(df['elapsed_seconds'], absorption, 
                label=f'{name} ({path_length:.2f} mm)',
                color=color)
        
        # Print information
        print(f"{name}: Most variant path length = {path_length:.2f} mm")
    
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Absorption ln(I/I0)')
    plt.title('Absorption Over Time for Most Variant Path Lengths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Use the function
print("\nGenerating absorption plot for most variant path lengths...")
plot_absorption_most_variant(
    red_df, green_df, blue_df,
    I0_red, I0_green, I0_blue
)


print("\nAnalysis complete. Check your plots.")

