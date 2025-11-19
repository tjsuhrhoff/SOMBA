#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soil-Based Mass Balance (SOMBA) – Demonstration of Internal Consistency
for the Full Three-Endmember Framework
==============================================================================

This script demonstrates the *non-simplified* Soil-Based Mass Balance (SOMBA)
framework for Enhanced Weathering (EW) systems (Suhrhoff et al., 2025), in
which the **weathered feedstock** (wf) is treated as an explicit third
compositional and physical endmember. The script verifies that the forward and
inverse solutions of the full SOMBA formulation are internally consistent and
illustrates how the third endmember modifies mixing geometry in (i, j)
composition space.

The workflow proceeds in four steps:

    1.  Generation of example data
        Synthetic soil and feedstock endmember compositions and densities are
        generated with realistic variability. Deployment parameters such as the
        feedstock application rate and dissolution fraction (τ) are specified.

    2.  Definition of explicit weathered-feedstock properties
        The weathered feedstock is assigned independent base-cation and
        immobile-element concentrations ([j]_wf, [i]_wf) and density (ρ_wf),
        relaxing the simplifying assumptions:
            [j]_wf = [j]_s    and    ρ_wf = ρ_s.
        This enables realistic modeling of dissolution, hydration, and porosity
        changes during weathering.

    3.  Application of the non-simplified SOMBA formulation
        The following functions from SOMBA.py are applied:
            • SOMBA_start:
                Computes the pre-weathering soil–feedstock mixture composition.
            • SOMBA_end:
                Computes the post-weathering composition while conserving mass
                among soil, feedstock, and weathered feedstock endmembers.
            • SOMBA_tau_meta:
                Recovers the dissolution fraction (τ), endmember contributions,
                and inferred feedstock application amount from the post-
                weathering composition and known endmember definitions.

        Agreement between assumed and recovered τ and application amount
        demonstrates algebraic and numerical internal consistency.

    4.  Visualization of results
        Two figures illustrate:
            (a) Mixing topology in (i, j) composition space when wf is explicit.
            (b) Assumed vs. recovered τ and application amounts.

Reference
---------
Suhrhoff, T. J., et al. (2025). Environmental Science and Technology. An 
updated framework and signal-to-noise analysis of soil mass balance approaches 
for quantifying enhanced weathering on managed lands.
https://doi.org/10.1021/acs.est.5c08303

Author
------
Tim Jesper Suhrhoff, October 11 2025
"""

from SOMBA import SOMBA_start, SOMBA_end, SOMBA_tau_meta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# ===========================================================================
# 1. Generation of Example Dataset
# ===========================================================================
# In this section, a synthetic dataset is constructed to demonstrate the full
# (three-endmember) SOMBA framework under controlled conditions. The values
# chosen represent plausible soil, feedstock, and deployment parameters.
# Moderate variation is introduced to mimic natural field heterogeneity and
# analytical noise.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Define representative soil and feedstock compositions
# ---------------------------------------------------------------------------
# j = weathering-sensitive base cation (e.g., Ca + Mg proxy)
# i = immobile reference element used to track mass mixing (e.g., Ti, Zr)
#
# These values represent the characteristic mean compositions of the baseline
# soil and the applied silicate feedstock prior to weathering. In field
# applications, they would be determined from laboratory analyses.
soil_j_molkg = 0.5
soil_i_molkg = 0.05
feedstock_j_molkg = 4.0
feedstock_i_molkg = 0.25

# ---------------------------------------------------------------------------
# Define base values for generating synthetic deployment replicates
# ---------------------------------------------------------------------------
# The parameters below specify the mean composition and bulk density of each
# endmember. These represent field-scale baseline properties and are varied
# below to generate realistic sample-to-sample variability.
params = {
    "soil_j_molkg": soil_j_molkg,
    "soil_i_molkg": soil_i_molkg,
    "feedstock_j_molkg": feedstock_j_molkg,
    "feedstock_i_molkg": feedstock_i_molkg,
    "rho_soil_tm3": 1.42,            # bulk soil density [t m⁻³]
    # effective bulk density of rock powder [t m⁻³]
    "rho_feedstock_tm3": 2.9 * 0.63
}

# Relative standard deviation used to introduce natural variability
# (e.g., analytical scatter, field heterogeneity).
SD = 0.05

# Generate a synthetic dataset (n = 10 deployment scenarios)
df = pd.DataFrame({
    key: np.random.normal(loc=val, scale=val * SD, size=10)
    for key, val in params.items()
})

# ---------------------------------------------------------------------------
# Define mixing and sampling depths (m)
# ---------------------------------------------------------------------------
# The mixing depth is the depth to which feedstock is incorporated during
# deployment. The sampling depth is the depth from which soil cores are taken.
#
# Differences between these depths directly modulate the detected feedstock
# signal in the sampled layer.
mixing_depth_m = 0.20
sampling_depth_m = 0.10

# Ensure mixing depth ≥ sampling depth to avoid artificial inflation of
# detected feedstock amounts in the model evaluation.
if mixing_depth_m < sampling_depth_m:
    mixing_depth_m = sampling_depth_m

df["d_mixed_layer_m"] = mixing_depth_m
df["d_sample_m"] = sampling_depth_m

# Compute the sampled soil volume [m³ ha⁻¹], required for feedstock detection
df["v_sampled_layer_m3ha"] = 100 * 100 * df["d_sample_m"]

# ---------------------------------------------------------------------------
# Generate deployment parameters (application rate and dissolution fraction)
# ---------------------------------------------------------------------------
# Application amount (a) is expressed in metric tonnes of rock powder per hectare.
# Dissolution fraction (τ) is the mass fraction of feedstock that dissolves
# during weathering in the mixed layer.
a_min, a_max = 50, 250        # application range [t ha⁻¹]
tau_min, tau_max = 0.1, 0.9   # dissolution fraction range [–]

df["a_assumed_tha"] = np.random.uniform(a_min, a_max, size=len(df))
df["tau_assumed"] = np.random.uniform(tau_min, tau_max, size=len(df))

# ---------------------------------------------------------------------------
# Compute soil and feedstock volumes and pre-weathering mass mixing ratio
# ---------------------------------------------------------------------------
# Feedstock volume in the mixed layer [m³ ha⁻¹]
df["v_feedstock_m3ha"] = df["a_assumed_tha"] / df["rho_feedstock_tm3"]

# Soil volume in the mixed layer [m³ ha⁻¹]
df["v_soil_m3ha"] = (100 * 100 * df["d_mixed_layer_m"]) - \
    df["v_feedstock_m3ha"]

# Mass of soil in the mixed layer [t ha⁻¹]
df["m_soil_tha"] = df["v_soil_m3ha"] * df["rho_soil_tm3"]

# Mass mixing ratio (unitless): fraction of feedstock mass in the mixed layer
df["rm"] = df["a_assumed_tha"] / (df["a_assumed_tha"] + df["m_soil_tha"])


# ---------------------------------------------------------------------------
# Define weathered feedstock properties (third endmember)
# ---------------------------------------------------------------------------
# In the simplified SOMBA formulation, weathered feedstock is not represented
# explicitly: its compositional effects are absorbed into soil volume changes.
# Here, we introduce the weathered feedstock (wf) as a true third endmember.

# Base cation concentration of weathered feedstock [j]_wf [mol kg⁻¹]
# Weathering generally lowers [j] relative to pristine feedstock; for
# demonstration, we sample slight enrichment around the soil value.
df["wf_j_molkg"] = df["soil_j_molkg"] * np.random.uniform(1.0, 1.2, len(df))

# Density of weathered feedstock ρ_wf [t m⁻³]
# Weathered material is often more porous and water-bearing, thus less dense.
# As for mobile concentration, we here simulate densities that are slightly
# enrighed compared to the soil composition.
df["rho_wf_tm3"] = df["rho_soil_tm3"] * np.random.uniform(1.0, 1.2, len(df))

# Immobile element concentration of the weathered feedstock [i]_wf [mol kg⁻¹]
# Immobile elements introduced from feedstock remain in the system and are
# redistributed among soil and weathered material. A conservative-tracer
# mass balance determines [i]_wf:
df["wf_i_molkg"] = (
    (df["rho_soil_tm3"] / df["rho_wf_tm3"]) * df["soil_i_molkg"] +
    (df["rho_feedstock_tm3"] / df["rho_wf_tm3"]) * df["feedstock_i_molkg"]
)


# ===========================================================================
# 2. Application of the SOMBA Functions (Non-Simplified Framework)
# ===========================================================================
# In this section, the full three-endmember SOMBA formulation is applied to the
# synthetic dataset. The workflow follows the same forward–inverse pairing used
# in the simplified demonstration, but now explicitly includes the **weathered
# feedstock** endmember (wf) in both forward and inverse calculations.
#
# Workflow summary:
#   (a) SOMBA_start computes the **pre-weathering** soil–feedstock mixture
#       composition from deployment parameters and endmember compositions.
#
#   (b) SOMBA_end simulates **post-weathering** composition and volume changes
#       using the *non-simplified* framework, where the weathered feedstock is
#       a chemically and physically distinct third endmember.
#
#   (c) SOMBA_tau_meta applies an inverse, matrix-based mass balance solution
#       to recover:
#           • dissolution fraction (τⱼ)
#           • endmember contributions (X_s, X_f, X_wf)
#           • detected feedstock input (a_detected)
#           • reconstructed pre-weathering composition
#
# Under ideal (noise-free) conditions, the recovered parameters should match the
# assumed input parameters. Agreement confirms **internal algebraic consistency**
# of the complete three-endmember SOMBA formulation.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# (a) Compute the pre-weathering soil–feedstock mixture composition
# ---------------------------------------------------------------------------
df = SOMBA_start(
    df,
    soil_j_col_molkg="soil_j_molkg",
    feedstock_j_col_molkg="feedstock_j_molkg",
    r_m_t0="rm",
    j_output_col_molkg="pre_weathering_mix_j_molkg",
    soil_i_col_molkg="soil_i_molkg",
    feedstock_i_col_molkg="feedstock_i_molkg",
    i_output_col_molkg="pre_weathering_mix_i_molkg"
)

# ---------------------------------------------------------------------------
# (b) Compute the post-weathering composition using the full three-endmember
#     model, which explicitly includes the weathered feedstock.
# ---------------------------------------------------------------------------
df = SOMBA_end(
    df,
    soil_j_col_molkg="soil_j_molkg",
    soil_i_col_molkg="soil_i_molkg",
    feedstock_j_col_molkg="feedstock_j_molkg",
    feedstock_i_col_molkg="feedstock_i_molkg",
    wf_j_col_molkg="wf_j_molkg",
    wf_i_col_molkg="wf_i_molkg",
    F_d_col="tau_assumed",                       # assumed dissolution fraction τ
    v_feedstock_initial_m3ha_col="v_feedstock_m3ha",
    v_soil_initial_m3ha_col="v_soil_m3ha",
    rho_soil_tm3_col="rho_soil_tm3",
    rho_feedstock_tm3_col="rho_feedstock_tm3",
    rho_wf_tm3_col="rho_wf_tm3",
    output_v_feedstock_end_m3ha_col="v_feedstock_end_m3ha",
    output_v_wf_end_m3ha_col="v_wf_end_m3ha",
    output_v_soil_end_m3ha_col="v_soil_end_m3ha",
    output_post_weathering_j_molkg_col="post_weathering_mix_j_molkg",
    output_post_weathering_i_molkg_col="post_weathering_mix_i_molkg"
)

# ---------------------------------------------------------------------------
# (c) Recover dissolution fraction (τⱼ), endmember contributions, and inferred
#     application amount via the full matrix-based inverse solution.
# ---------------------------------------------------------------------------
df = SOMBA_tau_meta(
    data=df,
    soil_j_col_molkg="soil_j_molkg",
    feedstock_j_col_molkg="feedstock_j_molkg",
    wf_j_col_molkg="wf_j_molkg",
    soil_i_col_molkg="soil_i_molkg",
    feedstock_i_col_molkg="feedstock_i_molkg",
    wf_i_col_molkg="wf_i_molkg",
    post_weathering_j_col_molkg="post_weathering_mix_j_molkg",
    post_weathering_i_col_molkg="post_weathering_mix_i_molkg",
    rho_soil_col_tm3="rho_soil_tm3",
    rho_feedstock_col_tm3="rho_feedstock_tm3",
    rho_wf_col_tm3="rho_wf_tm3",
    v_sampled_layer_col_m3="v_sampled_layer_m3ha",
    output_Xs_soil_contribution_col="Xs_soil_contribution",
    output_Xf_feed_contribution_col="Xf_feed_contribution",
    output_Xwf_weathered_feed_contribution_col="Xwf_weath_feed_contribution",
    output_tau_SOMBA_col="tau_SOMBA_meta_non_simplified",
    output_application_col_tha="application_amount_output_tha",
    output_pre_weathered_mix_i_col_molkg="pre_weathering_mix_output_i_molkg",
    output_pre_weathered_mix_j_col_molkg="pre_weathering_mix_output_j_molkg"
)

# ---------------------------------------------------------------------------
# Export full verification dataset for transparency and reproducibility
# ---------------------------------------------------------------------------
df.to_csv(
    "verification_data_plots/model_verification_data.csv",
    index=False
)


# ===========================================================================
# 3. Visualization of Results (Non-Simplified Framework)
# ===========================================================================
# This section generates two figures that demonstrate the internal consistency
# of the SOMBA framework when the weathered feedstock is explicitly included
# as a third compositional and physical endmember.
#
# Figure 1:
#   Plots soil, feedstock, weathered feedstock, and pre-/post-weathering
#   mixture compositions in (i, j) composition space. The trajectories between:
#       • soil → feedstock (initial mixing), and
#       • pre-weathering → post-weathering (dissolution evolution),
#   illustrate that the non-simplified SOMBA preserves physically valid,
#   mass-conserving mixing relationships among all three endmembers.
#
# Figure 2:
#   Compares assumed (input) vs. recovered (model-estimated) values for:
#       • dissolution fraction (τ),
#       • pre-weathering mixture composition [j],
#       • feedstock application amount.
#   Agreement between these quantities demonstrates algebraic closure of the
#   forward (SOMBA_start → SOMBA_end) and inverse (SOMBA_tau_meta) workflow.
#
#   Differences in the detected feedstock amount arise when the sampling depth
#   is smaller than the mixing depth, because only the sampled portion of the
#   mixed layer is analyzed. This scaling effect:
#
#         a_detected  ≈  a_true × (sampling depth / mixing depth)
#
#   emphasizes the importance of aligning sampling and mixing depths in
#   field deployments if the goal is also to quantitatively estimate initial
#   application amounts.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Define color palette (Oslo scientific colour map; Crameri, 2020)
# ---------------------------------------------------------------------------
colors = {
    "black": "#010101",
    "oslo_dark": "#0D1B29",
    "oslo_med_dark": "#133251",
    "oslo_blue": "#658AC7",
    "oslo_gray": "#AAB6CA",
    "oslo_light": "#D4D6DB",
    "white": "#FFFFFF"
}

# ---------------------------------------------------------------------------
# Figure 1 — Mixing-space representation including weathered feedstock
# ---------------------------------------------------------------------------
# This figure illustrates:
#   (a) Baseline soil and pristine feedstock endmembers,
#   (b) Explicit weathered-feedstock endmember,
#   (c) Pre- and post-weathering mixture compositions, and
#   (d) Elemental mixing and weathering trajectories.
# Together, these demonstrate the geometric structure of three-endmember
# SOMBA mixing in (i, j) space.
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# === Subplot (a): full overview of endmember and mix geometry ===

# Plot the *average baseline* soil and feedstock compositions
ax1.scatter(
    soil_i_molkg, soil_j_molkg,
    color=colors['oslo_dark'], edgecolors=colors['white'],
    label='average soil baseline', s=150, linewidth=1.5, zorder=6, marker='o'
)
ax1.scatter(
    feedstock_i_molkg, feedstock_j_molkg,
    color=colors['oslo_blue'], edgecolors=colors['white'],
    label='average feedstock baseline', s=150, linewidth=1.5, zorder=6, marker='D'
)

# Add the *average weathered feedstock endmember*
ax1.scatter(
    df['wf_i_molkg'].mean(), df['wf_j_molkg'].mean(),
    color=colors['oslo_gray'], edgecolors=colors['white'],
    label='average weathered feedstock', s=120, linewidth=1.5, zorder=5, marker='^'
)

# Single reference line between soil and feedstock
ax1.plot(
    [soil_i_molkg, feedstock_i_molkg],
    [soil_j_molkg, feedstock_j_molkg],
    color=colors['oslo_gray'], linestyle='-', linewidth=2,
    label='mixing line (avg)', zorder=1
)

# === Individual endmembers ===
ax1.scatter(
    df['soil_i_molkg'], df['soil_j_molkg'],
    color=colors['oslo_dark'], edgecolors=colors['white'],
    label='individual soil samples', s=75, linewidth=1.5, zorder=3
)
ax1.scatter(
    df['feedstock_i_molkg'], df['feedstock_j_molkg'],
    color=colors['oslo_blue'], edgecolors=colors['white'],
    label='individual feedstock samples', s=75, linewidth=1.5, zorder=3, marker='D'
)
ax1.scatter(
    df['wf_i_molkg'], df['wf_j_molkg'],
    color=colors['oslo_gray'], edgecolors=colors['white'],
    label='weathered feedstock (wf)', s=65, linewidth=1.0, zorder=2, marker='^'
)

# === Pre- and post-weathering compositions ===
ax1.scatter(
    df['pre_weathering_mix_i_molkg'], df['pre_weathering_mix_j_molkg'],
    color=colors['oslo_light'], edgecolors=colors['oslo_blue'],
    label='pre-weathering composition', s=75, linewidth=1.5, zorder=4, marker='s'
)
ax1.scatter(
    df['post_weathering_mix_i_molkg'], df['post_weathering_mix_j_molkg'],
    color=colors['oslo_blue'], edgecolors=colors['oslo_med_dark'],
    label='post-weathering composition', s=75, linewidth=1.5, zorder=4, marker='v'
)

# === Mixing trajectories (soils → feedstock, pre → post) ===
for i in range(len(df)):
    ax1.plot(
        [df.loc[i, 'soil_i_molkg'], df.loc[i, 'feedstock_i_molkg']],
        [df.loc[i, 'soil_j_molkg'], df.loc[i, 'feedstock_j_molkg']],
        color=colors['oslo_gray'], linestyle='--', linewidth=1, zorder=2
    )
    ax1.plot(
        [df.loc[i, 'pre_weathering_mix_i_molkg'],
         df.loc[i, 'post_weathering_mix_i_molkg']],
        [df.loc[i, 'pre_weathering_mix_j_molkg'],
         df.loc[i, 'post_weathering_mix_j_molkg']],
        color=colors['oslo_gray'], linestyle='-.', linewidth=1.5, zorder=3
    )

ax1.set_xlabel("immobile element concentration [i] [mol kg$^{-1}$]")
ax1.set_ylabel("base cation concentration [j] [mol kg$^{-1}$]")
ax1.set_title("a) Three-endmember SOMBA framework demonstration")
ax1.legend()
ax1.tick_params(axis='both', direction='in', length=6)

# === Subplot (b): zoomed compositional view ===
ax2.scatter(
    soil_i_molkg, soil_j_molkg,
    color=colors['oslo_dark'], edgecolors=colors['white'],
    label='avg soil baseline', s=150, linewidth=1.5, zorder=6, marker='o'
)
ax2.scatter(
    feedstock_i_molkg, feedstock_j_molkg,
    color=colors['oslo_blue'], edgecolors=colors['white'],
    label='avg feedstock baseline', s=150, linewidth=1.5, zorder=6, marker='h'
)
ax2.plot(
    [soil_i_molkg, feedstock_i_molkg],
    [soil_j_molkg, feedstock_j_molkg],
    color=colors['oslo_gray'], linestyle='-', linewidth=2, zorder=1
)
ax2.scatter(
    df['soil_i_molkg'], df['soil_j_molkg'],
    color=colors['oslo_dark'], edgecolors=colors['white'], s=75, linewidth=1.5
)
ax2.scatter(
    df['feedstock_i_molkg'], df['feedstock_j_molkg'],
    color=colors['oslo_blue'], edgecolors=colors['white'], s=75, linewidth=1.5,
    marker='h'
)
ax2.scatter(
    df['wf_i_molkg'], df['wf_j_molkg'],
    color=colors['oslo_gray'], edgecolors=colors['white'], s=65, linewidth=1.0,
    marker='^'
)
ax2.scatter(
    df['pre_weathering_mix_i_molkg'], df['pre_weathering_mix_j_molkg'],
    color=colors['oslo_light'], edgecolors=colors['oslo_blue'],
    s=75, linewidth=1.5, marker='s'
)
ax2.scatter(
    df['post_weathering_mix_i_molkg'], df['post_weathering_mix_j_molkg'],
    color=colors['oslo_blue'], edgecolors=colors['oslo_med_dark'],
    s=75, linewidth=1.5, marker='v'
)

for i in range(len(df)):
    ax2.plot(
        [df.loc[i, 'soil_i_molkg'], df.loc[i, 'feedstock_i_molkg']],
        [df.loc[i, 'soil_j_molkg'], df.loc[i, 'feedstock_j_molkg']],
        color=colors['oslo_gray'], linestyle='--', linewidth=1, zorder=2
    )
    ax2.plot(
        [df.loc[i, 'pre_weathering_mix_i_molkg'],
         df.loc[i, 'post_weathering_mix_i_molkg']],
        [df.loc[i, 'pre_weathering_mix_j_molkg'],
         df.loc[i, 'post_weathering_mix_j_molkg']],
        color=colors['oslo_gray'], linestyle='-.', linewidth=1.5, zorder=3
    )

# Axis scaling
x_min = min(soil_i_molkg, df['post_weathering_mix_i_molkg'].min()) * 0.9
x_max = max(soil_i_molkg, df['post_weathering_mix_i_molkg'].max()) * 1.1
y_min = min(soil_j_molkg, df['pre_weathering_mix_j_molkg'].min()) * 0.9
y_max = max(soil_j_molkg, df['pre_weathering_mix_j_molkg'].max()) * 1.1
ax2.set_xlim([x_min, x_max])
ax2.set_ylim([y_min, y_max])

ax2.set_xlabel("immobile element concentration [i] [mol kg$^{-1}$]")
ax2.set_ylabel("base cation concentration [j] [mol kg$^{-1}$]")
ax2.set_title("b) zoomed compositional space")
ax2.tick_params(axis='both', direction='in', length=6)

plt.tight_layout()
plt.savefig('verification_data_plots/SOMBA_example_demonstration_1.pdf')
plt.show()

# ---------------------------------------------------------------------------
# Figure 2 — Comparison of Input and Recovered Values
# ---------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# (a) Dissolution fraction comparison
ax1.scatter(df['tau_assumed'], df['tau_SOMBA_meta_non_simplified'],
            color=colors['oslo_blue'], edgecolors=colors['white'],
            label='samples', s=75, linewidth=1.5)
ax1.plot([df['tau_assumed'].min(), df['tau_assumed'].max()],
         [df['tau_assumed'].min(), df['tau_assumed'].max()],
         color='gray', linestyle='--', linewidth=1, label='1:1 line')
ax1.set_xlabel('True/assumed τ$_j$ value []')
ax1.set_ylabel('τ$_j$ estimated from SOMBA (non-simplified) []')
ax1.set_title('a) Comparison of assumed and estimated dissolution fractions')
ax1.legend()
ax1.tick_params(axis='both', direction='in', length=6)

# (b) Pre-weathering [j] composition comparison
ax2.scatter(df['pre_weathering_mix_j_molkg'],
            df['pre_weathering_mix_output_j_molkg'],
            color=colors['oslo_dark'], edgecolors=colors['white'],
            label='samples', s=75, linewidth=1.5)
ax2.plot([df['pre_weathering_mix_j_molkg'].min(),
          df['pre_weathering_mix_j_molkg'].max()],
         [df['pre_weathering_mix_j_molkg'].min(),
          df['pre_weathering_mix_j_molkg'].max()],
         color='gray', linestyle='--', linewidth=1, label='1:1 line')
ax2.set_xlabel('[j]_pre from SOMBA_start [mol kg$^{-1}$]')
ax2.set_ylabel('[j]_pre from SOMBA_tau_meta_non_simplified [mol kg$^{-1}$]')
ax2.set_title(
    'b) Comparison of calculated and reconstructed pre-weathering [j]')
ax2.legend()
ax2.tick_params(axis='both', direction='in', length=6)

# (c) Comparison of detected vs. assumed application amounts
ax3.scatter(df['a_assumed_tha'], df['application_amount_output_tha'],
            color=colors['oslo_light'], edgecolors=colors['white'],
            label='samples', s=75, linewidth=1.5)
ax3.plot([df['a_assumed_tha'].min(), df['a_assumed_tha'].max()],
         [df['a_assumed_tha'].min(), df['a_assumed_tha'].max()],
         color='gray', linestyle='--', linewidth=1, label='1:1 line')

# Expected mismatch line if mixing depth ≠ sampling depth
if mixing_depth_m != sampling_depth_m:
    min_x = df['a_assumed_tha'].min()
    max_x = df['a_assumed_tha'].max()
    min_y = min_x * (sampling_depth_m / mixing_depth_m)
    max_y = max_x * (sampling_depth_m / mixing_depth_m)
    ax3.plot([min_x, max_x], [min_y, max_y],
             color=colors['oslo_gray'], linestyle='-.', linewidth=1,
             label=('expected correlation (based on sampling/mixing '
                    'depth mismatch)'))

ax3.set_xlabel('Assumed application amount [t ha$^{-1}$]')
ax3.set_ylabel('Application amount estimated from SOMBA [t ha$^{-1}$]')
ax3.set_title('c) Comparison of detected and assumed application amounts')
ax3.legend()
ax3.tick_params(axis='both', direction='in', length=6)

plt.tight_layout()
plt.savefig('verification_data_plots/SOMBA_example_demonstration_2.pdf')
plt.show()
