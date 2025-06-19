#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 9th, 2025

This script demonstrates how to use all the functions defined for the soil
based mass balance approach (see Suhrhoff et al. 2025) and demonstrates that
the model is internally consistent. 

The script has three main parts:
    1.  Generation of example data.
    2.  Application of the soil mass balance approach.
    3.  Plotting of results

In the first part an example dataset is generated based on assumed deployment 
parameters. Some of these parameters such as feedstock application amount, 
dissolution fraction, etc., are only required for the SOMBA_start and SOMBA_end 
functions used to estimate pre- and post- weathering soil-feedstock mix
composition based on deployment parameters. 
These parameters may not be necessary when using the soil mass balance
framework to estimate rock powder dissolution fractions from post-weathering 
samples. 

The second part of the script uses the generated data frame to sequentially 
call the soil mass balance approach functions defined in the SOMBA.py file
which are derived in Suhrhoff et al. (2025). 
The functions being called are:
    1.  SOMBA_start: calculates pre-weathering soil-feedstock mix composition
        based on deployment parameters
    2.  SOMBA_end: calculates post-weathering soil-feedstock mix composition
        based on the pre-weathering composition (calculated from SOMBA_start)
        as well as assumed rock powder dissolution fraction.
    3.  SOMBA_tau: Calculates the rock powder dissolution fraction based on
        deployment data (baseline soil, feedstock, and post-weathering soil-
        feedstock mix composition)
    4.  SOMBA_tau_meta: Calculates the rock powder dissolution just like 
        SOMBA_tau, but exports additional metadata such as endmember 
        contributions as well as detected feedstock amount.
    5.  In the end all data is exported.

The third and last part creates two plots to demonstrate the internal 
consistency of this framework: 
    1.  Calculated dissolution fractions as well as pre-weathering soil
        compositions from the SOMBA functions are equivalent to the values
        assumed to generate post-weathering data.
    2.  The detected feedstock amount is as expected, i.e. equivalent to the 
        application amount if mixing depth is smaller or equal than the 
        sampling depth (or lower by sampling depth / mixing  depth if
        sampling depth is smaller than the mixing depth). Ideally, sampling 
        and mixing depth should be equivalent.

@author: Τim Jesper Suhrhoff
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SOMBA import SOMBA_tau, SOMBA_tau_meta, SOMBA_start, SOMBA_end
from matplotlib import rcParams


# Update font settings globally
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'font.weight': 'light',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

######################### generate example dataframe ##########################

soil_j_molkg = 0.5              # soil base cation (j) conc
soil_i_molkg = 0.05             # soil immobile element (i) conc
feedstock_j_molkg = 4           # feedstock [j]
feedstock_i_molkg = 0.25        # feedstock [i]


# generate random sample compositions
# Base values
params = {
    'soil_j_molkg': soil_j_molkg,           # soil base cation (j) conc
    'soil_i_molkg': soil_i_molkg,           # soil immobile element (i) conc
    'feedstock_j_molkg': feedstock_j_molkg,  # feedstock [j]
    'feedstock_i_molkg': feedstock_i_molkg,  # feedstock [i]
    'rho_soil_tm3': 1.42,                   # density soil
    'rho_feedstock_tm3': 2.9 * 0.63         # density rock powder
}

# Relative standard deviation
SD = 0.05


# Generate dataframe based on these distributions
df = pd.DataFrame({
    key: np.random.normal(loc=val, scale=val * SD, size=10)
    for key, val in params.items()
})

# Add columns for mixed layer and sampling depth, for the sake of this example
# they are set to not be the same though ideally they should be.
mixing_depth_m = 0.20
sampling_depth_m = 0.10


# Adjust mixing depth if it's less than sampling depth. For the sake of the
# data generation here this is necessary because else the mass mixing ratio
# of the mixed layer is assumed to be applicable to the entire sampled volume
# causing an inflated estimate of the amount of feedstock that was added.
if mixing_depth_m < sampling_depth_m:
    mixing_depth_m = sampling_depth_m

df['d_mixed_layer_m'] = mixing_depth_m
df['d_sample_m'] = sampling_depth_m


# Note that running the SOMBA_tau_meta function also requires the sampled soil
# layer volume; hence this is calculated here from the sampling depth.
df['v_sampled_layer_m3ha'] = 100 * 100 * df['d_sample_m']


# Because in this example we are also creating pre- and post-weathering
# compositions based on the functions SOMBA_start and SOMBA_end rather than
# measured sample data, we need to also generate the respective parameters

# Application rate (t/ha) and cation mass transfer coefficient ranges
a_min, a_max = 50, 250
tau_min, tau_max = 0.1, 0.9

# Add the new columns with uniformly sampled values:
df['a_assumed_tha'] = np.random.uniform(a_min, a_max, size=len(df))
df['tau_assumed'] = np.random.uniform(tau_min, tau_max, size=len(df))

# calculate soil and feedstock volumes as well as feedstock-soil-mass mixing
# ratio as the latter is required for the SOMBA functions

# Calculate initial feedstock volume:
df['v_feedstock_m3ha'] = df['a_assumed_tha'] / df['rho_feedstock_tm3']

# Calculate initial soil volume from mixed layer volume and feedstock volume:
df['v_soil_m3ha'] = 100 * 100 * df['d_mixed_layer_m'] - df['v_feedstock_m3ha']

# Calculate calculate the mass of initial soil:
df['m_soil_tha'] = df['v_soil_m3ha'] * df['rho_soil_tm3']

# Calculate mass mixing ratio rm:
df['rm'] = df['a_assumed_tha'] / \
    (df['a_assumed_tha'] + df['m_soil_tha'])


####################### example use of SOMBA functions ########################
# Compute pre-weathering composition
df = SOMBA_start(
    df,
    soil_j_col_molkg='soil_j_molkg',
    feedstock_j_col_molkg='feedstock_j_molkg',
    r_m_t0='rm',
    j_output_col_molkg='pre_weathering_mix_j_molkg',
    soil_i_col_molkg='soil_i_molkg',
    feedstock_i_col_molkg='feedstock_i_molkg',
    i_output_col_molkg='pre_weathering_mix_i_molkg'
)

# Compute post-weathering composition
df = SOMBA_end(
    df,
    soil_j_col_molkg='soil_j_molkg',
    F_d_col='tau_assumed',
    output_post_weathering_j_col_molkg='post_weathering_mix_j_molkg',
    v_feedstock_initial_m3ha_col='v_feedstock_m3ha',
    v_soil_initial_col_m3ha='v_soil_m3ha',
    rho_feedstock_col_tm3='rho_feedstock_tm3',
    rho_soil_col_tm3='rho_soil_tm3',
    feedstock_i_col_molkg='feedstock_i_molkg',
    feedstock_j_col_molkg='feedstock_j_molkg',
    soil_i_col_molkg='soil_i_molkg',
    output_v_feedstock_end_col_m3ha='v_feedstock_end_m3ha',
    output_v_soil_end_col_m3ha='v_soil_end_m3ha',
    output_post_weathering_i_col_molkg='post_weathering_mix_i_molkg'
)

# Compute tau from SOMBA_tau function
df = SOMBA_tau(
    data=df,
    soil_j_col_molkg='soil_j_molkg',
    feedstock_j_col_molkg='feedstock_j_molkg',
    soil_i_col_molkg='soil_i_molkg',
    feedstock_i_col_molkg='feedstock_i_molkg',
    post_weathering_j_col_molkg='post_weathering_mix_j_molkg',
    post_weathering_i_col_molkg='post_weathering_mix_i_molkg',
    rho_soil_col_tm3='rho_soil_tm3',
    rho_feedstock_col_tm3='rho_feedstock_tm3',
    output_tau_SOMBA_col='tau_SOMBA'
)


# Compute tau from SOMBA in addition to additional metadata based on function
# SOMBA_tau_meta

df = SOMBA_tau_meta(
    data=df,
    soil_j_col_molkg='soil_j_molkg',
    feedstock_j_col_molkg='feedstock_j_molkg',
    soil_i_col_molkg='soil_i_molkg',
    feedstock_i_col_molkg='feedstock_i_molkg',
    post_weathering_j_col_molkg='post_weathering_mix_j_molkg',
    post_weathering_i_col_molkg='post_weathering_mix_i_molkg',
    rho_soil_col_tm3='rho_soil_tm3',
    rho_feedstock_col_tm3='rho_feedstock_tm3',
    v_sampled_layer_col_m3='v_sampled_layer_m3ha',
    output_diss_feedstock_i_col_molkg='diss_feedstock_i_molkg',
    output_Xs_soil_contribution_col='Xs_soil_contribution',
    output_Xf_feed_contribution_col='Xf_feed_contribution',
    output_Xd_diss_feed_contribution_col='Xwf_weath_feed_contribution',
    output_tau_SOMBA_col='tau_SOMBA_meta',
    output_application_col_tha='application_amount_output_tha',
    output_pre_weathered_mix_i_col_molkg='pre_weathering_mix_output_i_molkg',
    output_pre_weathered_mix_j_col_molkg='pre_weathering_mix_output_j_molkg'
)

# Note that the detected application amount will not be the same as the
# assumed application amount if the mixing depth and sampling depth are not
# the same.

# Now we export all the data generated in this model verification script for
# potential futher use.
df.to_csv("verification_data_plots/model_verification_data.csv", index=False)


#################### plotting to visualize example data #######################
# Define colors from the Oslo 10 Scientific Colour Map (Crameri, 2020)
colors = {
    "black": "#010101",
    "oslo_dark": "#0D1B29",
    "oslo_med_dark": "#133251",
    "oslo_blue": "#658AC7",
    "oslo_gray": "#AAB6CA",
    "oslo_light": "#D4D6DB",
    "white": "#FFFFFF"
}


# Plotting of example data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot for the left subplot (ax1)
# Plot the composition of assumed baseline soi and feedstock composition.
# This could reflect sample mean for deployment data.

ax1.scatter(
    soil_i_molkg, soil_j_molkg,
    color=colors['oslo_dark'], edgecolors=colors['white'],
    label='soil baseline', s=150, linewidth=1.5, zorder=6, marker='o'
)

ax1.scatter(
    feedstock_i_molkg, feedstock_j_molkg,
    color=colors['oslo_blue'], edgecolors=colors['white'],
    label='feedstock', s=150, linewidth=1.5, zorder=6, marker='D'
)

# Draw a single "Mixing line" between the baseline soil and feedstock
# compositions.
ax1.plot(
    [soil_i_molkg, feedstock_i_molkg],
    [soil_j_molkg, feedstock_j_molkg],
    color=colors['oslo_gray'], linestyle='-',
    linewidth=2, label='mixing line', zorder=1
)

# Plot data for sample realizations
# baseline soils
ax1.scatter(
    df['soil_i_molkg'], df['soil_j_molkg'],
    color=colors['oslo_dark'], edgecolors=colors['white'],
    label='baseline soil samples', s=75, linewidth=1.5, zorder=3
)

# feedstock composition
ax1.scatter(
    df['feedstock_i_molkg'], df['feedstock_j_molkg'],
    color=colors['oslo_blue'], edgecolors=colors['white'],
    label='feedstock samples', s=75, linewidth=1.5, zorder=3, marker='D'
)

# pre-weathering composition
ax1.scatter(
    df['pre_weathering_mix_i_molkg'], df['pre_weathering_mix_j_molkg'],
    color=colors['oslo_light'], edgecolors=colors['oslo_blue'],
    label='pre-weathering composition', s=75, linewidth=1.5, zorder=4,
    marker='s'
)

# post-weathering composition
ax1.scatter(
    df['post_weathering_mix_i_molkg'], df['post_weathering_mix_j_molkg'],
    color=colors['oslo_blue'], edgecolors=colors['oslo_med_dark'],
    label='post-weathering composition', s=75, linewidth=1.5, zorder=4,
    marker='v'
)


# Draw the mixing line between each baseline soil-feedstock pair as well as
# between pre- and post-weathering samples
for i in range(len(df)):
    # Line between soil and feedstock
    x_vals = [df.loc[i, 'soil_i_molkg'], df.loc[i, 'feedstock_i_molkg']]
    y_vals = [df.loc[i, 'soil_j_molkg'], df.loc[i, 'feedstock_j_molkg']]
    ax1.plot(x_vals, y_vals, color=colors['oslo_gray'],
             linestyle='--', linewidth=1, zorder=2)

    # Line between pre-weathering and post-weathering compositions
    x_vals_weathering = [df.loc[i, 'pre_weathering_mix_i_molkg'],
                         df.loc[i, 'post_weathering_mix_i_molkg']]
    y_vals_weathering = [df.loc[i, 'pre_weathering_mix_j_molkg'],
                         df.loc[i, 'post_weathering_mix_j_molkg']]
    ax1.plot(x_vals_weathering, y_vals_weathering, color=colors['oslo_gray'],
             linestyle='-.', linewidth=1.5, zorder=3)

# Descriptions for the first subplot
ax1.set_xlabel(
    "immobile element concentration [i] [mol kg$^{-1}$]")
ax1.set_ylabel("base cation concentration [j] [mol kg$^{-1}$]")
ax1.set_title("a) Demonstration of SOMBA framework")
ax1.legend()
ax1.tick_params(axis='both', direction='in', length=6)


# Plot for the second subplot (ax2) with adjusted axis limits
# Plot data for sample realizations (same as in ax1)
ax2.scatter(
    soil_i_molkg, soil_j_molkg,
    color=colors['oslo_dark'], edgecolors=colors['white'],
    label='soil baseline', s=150, linewidth=1.5, zorder=6, marker='o'
)

ax2.scatter(
    feedstock_i_molkg, feedstock_j_molkg,
    color=colors['oslo_blue'], edgecolors=colors['white'],
    label='feedstock', s=150, linewidth=1.5, zorder=6, marker='h'
)

# Draw a single "Mixing line" between the baseline soil and feedstock
# compositions.
ax2.plot(
    [soil_i_molkg, feedstock_i_molkg],
    [soil_j_molkg, feedstock_j_molkg],
    color=colors['oslo_gray'], linestyle='-',
    linewidth=2, label='mixing line', zorder=1
)

# Plot data for sample realizations (same as in ax1)
ax2.scatter(
    df['soil_i_molkg'], df['soil_j_molkg'],
    color=colors['oslo_dark'], edgecolors=colors['white'],
    label='soil samples', s=75, linewidth=1.5, zorder=3
)

ax2.scatter(
    df['feedstock_i_molkg'], df['feedstock_j_molkg'],
    color=colors['oslo_blue'], edgecolors=colors['white'],
    label='feedstock samples', s=75, linewidth=1.5, zorder=3, marker='h'
)

ax2.scatter(
    df['pre_weathering_mix_i_molkg'], df['pre_weathering_mix_j_molkg'],
    color=colors['oslo_light'], edgecolors=colors['oslo_blue'],
    label='pre-weathering composition', s=75, linewidth=1.5, zorder=4,
    marker='s'
)

ax2.scatter(
    df['post_weathering_mix_i_molkg'], df['post_weathering_mix_j_molkg'],
    color=colors['oslo_blue'], edgecolors=colors['oslo_med_dark'],
    label='post-weathering composition', s=75, linewidth=1.5, zorder=4,
    marker='v'
)

# Draw the mixing line between each baseline soil-feedstock pair as well as
# between pre- and post-weathering samples
for i in range(len(df)):
    # Line between soil and feedstock
    x_vals = [df.loc[i, 'soil_i_molkg'], df.loc[i, 'feedstock_i_molkg']]
    y_vals = [df.loc[i, 'soil_j_molkg'], df.loc[i, 'feedstock_j_molkg']]
    ax2.plot(x_vals, y_vals, color=colors['oslo_gray'],
             linestyle='--', linewidth=1, zorder=2)

    # Line between pre-weathering and post-weathering compositions
    x_vals_weathering = [df.loc[i, 'pre_weathering_mix_i_molkg'],
                         df.loc[i, 'post_weathering_mix_i_molkg']]
    y_vals_weathering = [df.loc[i, 'pre_weathering_mix_j_molkg'],
                         df.loc[i, 'post_weathering_mix_j_molkg']]
    ax2.plot(x_vals_weathering, y_vals_weathering, color=colors['oslo_gray'],
             linestyle='-.', linewidth=1.5, zorder=3)

# Calculate min and max values for axis limits before setting them
x_min = min(soil_i_molkg, min(df['post_weathering_mix_i_molkg'])) * 0.9
x_max = max(soil_i_molkg, max(df['post_weathering_mix_i_molkg'])) * 1.1

y_min = min(soil_j_molkg, min(df['pre_weathering_mix_j_molkg'])) * 0.9
y_max = max(soil_j_molkg, max(df['pre_weathering_mix_j_molkg'])) * 1.1

# Set axis limits for the second subplot using calculated min/max values
ax2.set_xlim([x_min, x_max])
ax2.set_ylim([y_min, y_max])

# Descriptions for the second subplot
ax2.set_xlabel(
    "immobile element concentration [i] [mol kg$^{-1}$]")
ax2.set_ylabel("base cation concentration [j] [mol kg$^{-1}$]")
ax2.set_title("b) zoom")
ax2.tick_params(axis='both', direction='in', length=6)

plt.tight_layout()

plt.savefig('verification_data_plots/SOMBA_example_demonstration_1.pdf')

plt.show()


# Create a second figure to compare assumed dissolution fractions, calculated
# preweathering compositions and application amounts with the values derived
# from the soil mass balance framework.

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(18, 6))  # Updated to have 3 subplots

# First subplot: comparison of dissolution fractions
ax1.scatter(
    df['tau_assumed'], df['tau_SOMBA'],
    color=colors['oslo_blue'], edgecolors=colors['white'],
    label='samples', s=75, linewidth=1.5
)
ax1.set_xlabel('true/assumed τ$_j$ value for SOMBA_end, []')
ax1.set_ylabel('τ$_j$ value estimated from SOMBA_tau, []')
ax1.set_title(
    'a) Comparison of estimated and assumed dissolution fractions (τ$_j$)')

# Adding the 1:1 line (y = x) for the first subplot
ax1.plot([min(df['tau_assumed']), max(df['tau_assumed'])],
         [min(df['tau_assumed']), max(df['tau_assumed'])],
         color='gray', linestyle='--', linewidth=1, label='1:1 line')

ax1.legend()
ax1.tick_params(axis='both', direction='in', length=6)

# Second subplot: comparison of pre-weathering mix sample compositions
ax2.scatter(
    df['pre_weathering_mix_j_molkg'], df['pre_weathering_mix_output_j_molkg'],
    color=colors['oslo_dark'], edgecolors=colors['white'],
    label='samples', s=75, linewidth=1.5
)
ax2.set_xlabel(
    'calculated pre-weathering [j] from SOMBA_start [mol kg$^{-1}$]')
ax2.set_ylabel(
    'pre-weathering [j] estimated from SOMBA_tau_meta [mol kg$^{-1}$]')
ax2.set_title(
    'b) Comparison of calculated and estimated base cation concentrations')

# Adding the 1:1 line (y = x) for the second subplot
ax2.plot([min(df['pre_weathering_mix_j_molkg']),
          max(df['pre_weathering_mix_j_molkg'])],
         [min(df['pre_weathering_mix_j_molkg']),
          max(df['pre_weathering_mix_j_molkg'])],
         color='gray', linestyle='--', linewidth=1, label='1:1 line')

ax2.legend()
ax2.tick_params(axis='both', direction='in', length=6)

# Third subplot: comparison of detected and assumed feedstock application
# amounts
ax3.scatter(
    df['a_assumed_tha'], df['application_amount_output_tha'],
    color=colors['oslo_light'], edgecolors=colors['white'],
    label='samples', s=75, linewidth=1.5
)

# Adding the 1:1 line (y = x) for the third subplot
ax3.plot([min(df['a_assumed_tha']), max(df['a_assumed_tha'])],
         [min(df['a_assumed_tha']), max(df['a_assumed_tha'])],
         color='gray', linestyle='--', linewidth=1, label='1:1 line')

# Only add the expected correlation line if mixing depth is not equal to
# sampling depth
if mixing_depth_m != sampling_depth_m:
    # Find the minimum and maximum values of a_assumed_tha
    min_x = df['a_assumed_tha'].min()
    max_x = df['a_assumed_tha'].max()

    # Calculate the corresponding expected y-values
    min_y = min_x * (sampling_depth_m / mixing_depth_m)
    max_y = max_x * (sampling_depth_m / mixing_depth_m)

    # Plot the expected correlation line
    ax3.plot([min_x, max_x], [min_y, max_y], color=colors['oslo_gray'],
             linestyle='-.', linewidth=1,
             label='expected correlation (based on sampling and mixing '
             'depth mismatch)')

ax3.set_xlabel('Assumed application amount [tha$^{-1}$]')
ax3.set_ylabel(
    'Application amount estimated from SOMBA_tau_meta [tha$^{-1}$]')
ax3.set_title(
    'c) Comparison of detected and assumed application amounts')
ax3.legend()
ax3.tick_params(axis='both', direction='in', length=6)

# Adjust layout for the three subplots
plt.tight_layout()

# Save the figure as PDF
plt.savefig('verification_data_plots/SOMBA_example_demonstration_2.pdf')

# Show the plot
plt.show()
