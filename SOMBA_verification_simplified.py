#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soil-Based Mass Balance (SOMBA) – Demonstration of internal consistency for
the simplified framework
==============================================================================

This script demonstrates the use of the simplified soil-based mass balance
approach for Enhanced Weathering (EW) systems (Suhrhoff et al., 2025),
and verifies that the framework is internally consistent. The script performs
three sequential steps:

    1.  Generation of example data
        An example dataset is constructed based on assumed deployment
        parameters. These parameters (e.g., feedstock application amount and
        dissolution fraction) are required when using the forward SOMBA
        functions (SOMBA_start and SOMBA_end_simplified), which estimate pre-
        and post-weathering soil–feedstock compositions from deployment inputs.
        These parameters are *not* needed when SOMBA is used in its inverse
        form to infer dissolution fractions from measured post-weathering data.

    2.  Application of the soil mass balance approach
        The generated dataset is processed using the functions implemented in
        SOMBA.py (Suhrhoff et al., 2025):
            • SOMBA_start:
                Computes the pre-weathering soil–feedstock mixture composition
                based on soil and feedstock endmembers and the mixing ratio.
            • SOMBA_end_simplified:
                Computes the post-weathering mixture composition based on the
                pre-weathering composition and the assumed dissolution fraction
                of the feedstock.
            • SOMBA_tau_simplified:
                Recovers the dissolution fraction from the post-weathering
                composition, together with soil and feedstock endmembers.
            • SOMBA_tau_meta_simplified:
                Computes the same dissolution fraction as SOMBA_tau_simplified,
                but additionally returns endmember contributions and the amount
                of feedstock detected in the sampled volume.

        The sequence demonstrates that the SOMBA functions recover the true 
        dissolution fraction and feedstock amount.

    3.  Plotting of results
        Two figures illustrate internal consistency:
            (a)  Pre- and post-weathering compositions fall along physically
                 valid mixing trajectories in (i, j) composition space.
            (b)  Estimated dissolution fractions and feedstock application
                 amounts match the assumed input values. Deviations in detected
                 feedstock amount follow from differences between sampling and
                 mixing depths. Ideally, these depths should be matched in field
                 deployments to minimize sampling bias.
                 
Reference
---------
Suhrhoff, T. J., et al. (2025). Environmental Science and Technology. An 
updated framework and signal-to-noise analysis of soil mass balance approaches 
for quantifying enhanced weathering on managed lands.
https://doi.org/10.1021/acs.est.5c08303


Author
------
Tim Jesper Suhrhoff, October 31 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from SOMBA import (
    SOMBA_tau_simplified,
    SOMBA_tau_meta_simplified,
    SOMBA_start,
    SOMBA_end_simplified
)

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
# In this section, a synthetic dataset is constructed to demonstrate the SOMBA
# framework under controlled conditions. The values chosen represent plausible
# soil and silicate feedstock compositions and deployment parameters. Moderate
# variation is introduced to mimic natural field heterogeneity and analytical
# uncertainty.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Define representative soil and feedstock compositions
# ---------------------------------------------------------------------------
# j = weathering-sensitive major cation (e.g., Ca + Mg proxy)
# i = immobile reference element used to track mass mixing (e.g., Ti, Zr)
#
# These values represent the characteristic mean compositions of the baseline
# soil and the applied silicate feedstock before deployment. In practice, these
# would be determined from laboratory measurements of site-specific materials.
soil_j_molkg = 0.5
soil_i_molkg = 0.05
feedstock_j_molkg = 4.0
feedstock_i_molkg = 0.25

# ---------------------------------------------------------------------------
# Define base values for generating synthetic deployment replicates
# ---------------------------------------------------------------------------
# The parameters below define the mean composition and bulk density of each
# endmember. These values represent site-scale properties and are varied below
# to generate realistic sample-to-sample variability.
params = {
    "soil_j_molkg": soil_j_molkg,
    "soil_i_molkg": soil_i_molkg,
    "feedstock_j_molkg": feedstock_j_molkg,
    "feedstock_i_molkg": feedstock_i_molkg,
    "rho_soil_tm3": 1.42,            # bulk soil density [t m⁻³]
    "rho_feedstock_tm3": 2.9 * 0.63  # effective feedstock density [t m⁻³]
}

# Relative standard deviation used to introduce realistic natural variation
# across samples (e.g., field heterogeneity, analytical noise).
SD = 0.05

# Generate a synthetic dataset (here: n = 10 samples / deployment scenarios)
df = pd.DataFrame({
    key: np.random.normal(loc=val, scale=val * SD, size=10)
    for key, val in params.items()
})

# ---------------------------------------------------------------------------
# Define mixing and sampling depths (m)
# ---------------------------------------------------------------------------
# The mixing depth represents the depth over which the feedstock is incorporated
# into the soil following field application. The sampling depth represents the
# depth from which soil samples are collected for analysis.
#
# These depths are not always equal in field campaigns; differences between them
# directly influence the amount of feedstock detectable in the sampled profile.
mixing_depth_m = 0.20
sampling_depth_m = 0.10

# Ensure that the mixing depth is not smaller than the sampling depth.
# If mixing depth < sampling depth, the feedstock would be incorrectly assumed
# to be mixed throughout the full sampled volume, which would artificially
# inflate detected feedstock amounts in the model evaluation.
if mixing_depth_m < sampling_depth_m:
    mixing_depth_m = sampling_depth_m

df["d_mixed_layer_m"] = mixing_depth_m
df["d_sample_m"] = sampling_depth_m

# Compute the sampled soil layer volume [m³ ha⁻¹], required for computing
# feedstock detectability in SOMBA_tau_meta_simplified.
df["v_sampled_layer_m3ha"] = 100 * 100 * df["d_sample_m"]

# ---------------------------------------------------------------------------
# Generate deployment parameters for the forward SOMBA demonstration
# ---------------------------------------------------------------------------
# Application amount (a) is expressed in metric tonnes of feedstock per hectare.
# Dissolution fraction (τ) is the fraction of feedstock mass that dissolves
# during weathering in the soil environment.
a_min, a_max = 50, 250        # feedstock application range [t ha⁻¹]
tau_min, tau_max = 0.1, 0.9   # dissolution fraction range [-]

df["a_assumed_tha"] = np.random.uniform(a_min, a_max, size=len(df))
df["tau_assumed"] = np.random.uniform(tau_min, tau_max, size=len(df))

# ---------------------------------------------------------------------------
# Compute soil and feedstock volumes and mixing ratio
# ---------------------------------------------------------------------------
# Feedstock volume in the mixed layer [m³ ha⁻¹]
df["v_feedstock_m3ha"] = df["a_assumed_tha"] / df["rho_feedstock_tm3"]

# Soil volume present in the mixed layer [m³ ha⁻¹]
df["v_soil_m3ha"] = 100 * 100 * df["d_mixed_layer_m"] - df["v_feedstock_m3ha"]

# Mass of soil in the mixed layer [t ha⁻¹]
df["m_soil_tha"] = df["v_soil_m3ha"] * df["rho_soil_tm3"]

# Mass mixing ratio (unitless) representing the fraction of feedstock in the
# total mixed layer mass. This is the key parameter linking deployment to
# pre-weathering mixture composition.
df["rm"] = df["a_assumed_tha"] / (df["a_assumed_tha"] + df["m_soil_tha"])


# ===========================================================================
# 2. Application of the SOMBA functions
# ===========================================================================
# The SOMBA framework is applied in a forward–inverse pairing:
#   (a) SOMBA_start constructs the pre-weathering soil–feedstock mixture from
#       deployment parameters and endmember compositions.
#   (b) SOMBA_end_simplified simulates weathering by applying the assumed
#       dissolution fraction τ, yielding a post-weathering composition.
#   (c) SOMBA_tau_simplified and SOMBA_tau_meta_simplified recover
#       τ and the feedstock contribution from the post-weathering mixture.
# Under ideal (noise-free) conditions, the recovered values should match the
# assumed input values, demonstrating internal consistency.
# ---------------------------------------------------------------------------

# Compute the pre-weathering soil–feedstock mixture composition
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

# Compute the post-weathering mixture composition after dissolution
df = SOMBA_end_simplified(
    df,
    soil_j_col_molkg="soil_j_molkg",
    F_d_col="tau_assumed",  # assumed dissolution fraction τ
    output_post_weathering_j_col_molkg="post_weathering_mix_j_molkg",
    v_feedstock_initial_m3ha_col="v_feedstock_m3ha",
    v_soil_initial_col_m3ha="v_soil_m3ha",
    rho_feedstock_col_tm3="rho_feedstock_tm3",
    rho_soil_col_tm3="rho_soil_tm3",
    feedstock_i_col_molkg="feedstock_i_molkg",
    feedstock_j_col_molkg="feedstock_j_molkg",
    soil_i_col_molkg="soil_i_molkg",
    output_v_feedstock_end_col_m3ha="v_feedstock_end_m3ha",
    output_v_soil_end_col_m3ha="v_soil_end_m3ha",
    output_post_weathering_i_col_molkg="post_weathering_mix_i_molkg"
)

# ---------------------------------------------------------------------------
# Recovering dissolution fraction τ from post-weathering composition
# ---------------------------------------------------------------------------
# SOMBA_tau_simplified returns only the estimated dissolution fraction τ.
df = SOMBA_tau_simplified(
    data=df,
    soil_j_col_molkg="soil_j_molkg",
    feedstock_j_col_molkg="feedstock_j_molkg",
    soil_i_col_molkg="soil_i_molkg",
    feedstock_i_col_molkg="feedstock_i_molkg",
    post_weathering_j_col_molkg="post_weathering_mix_j_molkg",
    post_weathering_i_col_molkg="post_weathering_mix_i_molkg",
    rho_soil_col_tm3="rho_soil_tm3",
    rho_feedstock_col_tm3="rho_feedstock_tm3",
    output_tau_SOMBA_col="tau_SOMBA"
)

# ---------------------------------------------------------------------------
# Recovering τ together with endmember contributions and detected application
# ---------------------------------------------------------------------------
# SOMBA_tau_meta_simplified provides:
#   • X_s, X_f, X_wf     (proportional mixing contributions)
#   • an inferred application amount
#   • reconstructed pre-weathering mixture composition
df = SOMBA_tau_meta_simplified(
    data=df,
    soil_j_col_molkg="soil_j_molkg",
    feedstock_j_col_molkg="feedstock_j_molkg",
    soil_i_col_molkg="soil_i_molkg",
    feedstock_i_col_molkg="feedstock_i_molkg",
    post_weathering_j_col_molkg="post_weathering_mix_j_molkg",
    post_weathering_i_col_molkg="post_weathering_mix_i_molkg",
    rho_soil_col_tm3="rho_soil_tm3",
    rho_feedstock_col_tm3="rho_feedstock_tm3",
    v_sampled_layer_col_m3="v_sampled_layer_m3ha",
    output_diss_feedstock_i_col_molkg="diss_feedstock_i_molkg",
    output_Xs_soil_contribution_col="Xs_soil_contribution",
    output_Xf_feed_contribution_col="Xf_feed_contribution",
    output_Xwf_weathered_feed_contribution_col="Xwf_weath_feed_contribution",
    output_tau_SOMBA_col="tau_SOMBA_meta",
    output_application_col_tha="application_amount_output_tha",
    output_pre_weathered_mix_i_col_molkg="pre_weathering_mix_output_i_molkg",
    output_pre_weathered_mix_j_col_molkg="pre_weathering_mix_output_j_molkg"
)

# ---------------------------------------------------------------------------
# Interpretation note:
# If mixing depth > sampling depth, the detected feedstock amount differs from
# the applied feedstock amount. Specifically, detection scales with the ratio
# (sampling depth / mixing depth). Matching these depths in field deployments
# improves sensitivity and interpretability.
# ---------------------------------------------------------------------------

# Export full verification dataset for plotting or supplementary material
df.to_csv(
    "verification_data_plots/model_verification_data_simplified.csv",
    index=False
)

# ===========================================================================
# 3. Visualization of Results
# ===========================================================================
# This section generates two figures that illustrate the internal consistency
# of the SOMBA framework using the synthetic dataset constructed above.
#
# Figure 1:
#   Plots pre- and post-weathering compositions in (i, j) concentration space.
#   The trajectories between:
#       • baseline soil → feedstock,
#       • pre-weathering → post-weathering compositions,
#   demonstrate that the modeled mass balance follows physically valid mixing
#   relationships.
#
# Figure 2:
#   Compares assumed (input) and recovered (model-estimated) values of:
#       • dissolution fraction (τ),
#       • pre-weathering mixture composition,
#       • feedstock application amount.
#   Agreement between these quantities demonstrates internal model consistency.
#   Differences in detected application amounts reflect the ratio between the
#   sampling depth and the mixing depth, emphasizing the importance of matching
#   these depths in field deployments.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Define color palette (Oslo 10 Scientific Colour Map; Crameri, 2020)
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
# Figure 1 — Mixing-space representation of soil, feedstock, and mixtures
# ---------------------------------------------------------------------------
# This figure illustrates:
#   (a) Baseline soil and feedstock compositions,
#   (b) Pre-weathering and post-weathering mixture compositions, and
#   (c) Mixing trajectories before and after dissolution.
# Together, these plots demonstrate that the SOMBA framework preserves physically
# valid mixing relationships in (i, j) composition space.
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Baseline soil composition (single point and sample realizations)
ax1.scatter(
    soil_i_molkg, soil_j_molkg,
    color=colors["oslo_dark"], edgecolors=colors["white"],
    label="soil baseline", s=150, linewidth=1.5, zorder=6, marker="o"
)

# Baseline feedstock composition
ax1.scatter(
    feedstock_i_molkg, feedstock_j_molkg,
    color=colors["oslo_blue"], edgecolors=colors["white"],
    label="feedstock", s=150, linewidth=1.5, zorder=6, marker="D"
)

# Idealized mixing line between soil and feedstock endmembers
ax1.plot(
    [soil_i_molkg, feedstock_i_molkg],
    [soil_j_molkg, feedstock_j_molkg],
    color=colors["oslo_gray"], linestyle="-", linewidth=2,
    label="mixing line", zorder=1
)

# Sample realizations of baseline soil
ax1.scatter(
    df["soil_i_molkg"], df["soil_j_molkg"],
    color=colors["oslo_dark"], edgecolors=colors["white"],
    label="baseline soil samples", s=75, linewidth=1.5, zorder=3
)

# Sample realizations of feedstock
ax1.scatter(
    df["feedstock_i_molkg"], df["feedstock_j_molkg"],
    color=colors["oslo_blue"], edgecolors=colors["white"],
    label="feedstock samples", s=75, linewidth=1.5, zorder=3, marker="D"
)

# Pre-weathering mixture compositions
ax1.scatter(
    df["pre_weathering_mix_i_molkg"], df["pre_weathering_mix_j_molkg"],
    color=colors["oslo_light"], edgecolors=colors["oslo_blue"],
    label="pre-weathering composition", s=75, linewidth=1.5, zorder=4,
    marker="s"
)

# Post-weathering mixture compositions
ax1.scatter(
    df["post_weathering_mix_i_molkg"], df["post_weathering_mix_j_molkg"],
    color=colors["oslo_blue"], edgecolors=colors["oslo_med_dark"],
    label="post-weathering composition", s=75, linewidth=1.5, zorder=4,
    marker="v"
)

# Plot mixing trajectories for each realization
for i in range(len(df)):
    # Soil → feedstock mixing direction
    ax1.plot(
        [df.loc[i, "soil_i_molkg"], df.loc[i, "feedstock_i_molkg"]],
        [df.loc[i, "soil_j_molkg"], df.loc[i, "feedstock_j_molkg"]],
        color=colors["oslo_gray"], linestyle="--", linewidth=1, zorder=2
    )

    # Pre-weathering → post-weathering (dissolution trajectory)
    ax1.plot(
        [df.loc[i, "pre_weathering_mix_i_molkg"],
         df.loc[i, "post_weathering_mix_i_molkg"]],
        [df.loc[i, "pre_weathering_mix_j_molkg"],
         df.loc[i, "post_weathering_mix_j_molkg"]],
        color=colors["oslo_gray"], linestyle="-.", linewidth=1.5, zorder=3
    )

ax1.set_xlabel("immobile element concentration [i] [mol kg$^{-1}$]")
ax1.set_ylabel("base cation concentration [j] [mol kg$^{-1}$]")
ax1.set_title("a) Demonstration of SOMBA framework")
ax1.legend()
ax1.tick_params(axis="both", direction="in", length=6)

# ---------------------------------------------------------------------------
# Right panel (zoomed view of the same compositional space)
# ---------------------------------------------------------------------------
ax2.scatter(
    soil_i_molkg, soil_j_molkg,
    color=colors["oslo_dark"], edgecolors=colors["white"],
    label="soil baseline", s=150, linewidth=1.5, zorder=6, marker="o"
)

ax2.scatter(
    feedstock_i_molkg, feedstock_j_molkg,
    color=colors["oslo_blue"], edgecolors=colors["white"],
    label="feedstock", s=150, linewidth=1.5, zorder=6, marker="h"
)

ax2.plot(
    [soil_i_molkg, feedstock_i_molkg],
    [soil_j_molkg, feedstock_j_molkg],
    color=colors["oslo_gray"], linestyle="-", linewidth=2, zorder=1,
    label="mixing line"
)

ax2.scatter(
    df["soil_i_molkg"], df["soil_j_molkg"],
    color=colors["oslo_dark"], edgecolors=colors["white"],
    label="soil samples", s=75, linewidth=1.5, zorder=3
)

ax2.scatter(
    df["feedstock_i_molkg"], df["feedstock_j_molkg"],
    color=colors["oslo_blue"], edgecolors=colors["white"],
    label="feedstock samples", s=75, linewidth=1.5, zorder=3, marker="h"
)

ax2.scatter(
    df["pre_weathering_mix_i_molkg"], df["pre_weathering_mix_j_molkg"],
    color=colors["oslo_light"], edgecolors=colors["oslo_blue"],
    label="pre-weathering composition", s=75, linewidth=1.5, zorder=4,
    marker="s"
)

ax2.scatter(
    df["post_weathering_mix_i_molkg"], df["post_weathering_mix_j_molkg"],
    color=colors["oslo_blue"], edgecolors=colors["oslo_med_dark"],
    label="post-weathering composition", s=75, linewidth=1.5, zorder=4,
    marker="v"
)

for i in range(len(df)):
    ax2.plot(
        [df.loc[i, "soil_i_molkg"], df.loc[i, "feedstock_i_molkg"]],
        [df.loc[i, "soil_j_molkg"], df.loc[i, "feedstock_j_molkg"]],
        color=colors["oslo_gray"], linestyle="--", linewidth=1, zorder=2
    )
    ax2.plot(
        [df.loc[i, "pre_weathering_mix_i_molkg"],
         df.loc[i, "post_weathering_mix_i_molkg"]],
        [df.loc[i, "pre_weathering_mix_j_molkg"],
         df.loc[i, "post_weathering_mix_j_molkg"]],
        color=colors["oslo_gray"], linestyle="-.", linewidth=1.5, zorder=3
    )

# Axis scaling for clearer visualization
x_min = min(soil_i_molkg, df["post_weathering_mix_i_molkg"].min()) * 0.9
x_max = max(soil_i_molkg, df["post_weathering_mix_i_molkg"].max()) * 1.1
y_min = min(soil_j_molkg, df["pre_weathering_mix_j_molkg"].min()) * 0.9
y_max = max(soil_j_molkg, df["pre_weathering_mix_j_molkg"].max()) * 1.1

ax2.set_xlim([x_min, x_max])
ax2.set_ylim([y_min, y_max])

ax2.set_xlabel("immobile element concentration [i] [mol kg$^{-1}$]")
ax2.set_ylabel("base cation concentration [j] [mol kg$^{-1}$]")
ax2.set_title("b) zoom")
ax2.tick_params(axis="both", direction="in", length=6)

plt.tight_layout()

plt.savefig(
    "verification_data_plots/SOMBA_example_demonstration_1_simplified.pdf"
)

plt.show()

# ---------------------------------------------------------------------------
# Figure 2 — Comparison of Input and Recovered Values
# ---------------------------------------------------------------------------
# This figure compares:
#   (a) assumed vs. recovered dissolution fraction (τ),
#   (b) calculated vs. recovered pre-weathering base cation concentration,
#   (c) assumed vs. inferred feedstock application amount.
# Agreement between these quantities confirms internal model consistency.
# ---------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# (a) Dissolution fraction comparison
ax1.scatter(
    df["tau_assumed"], df["tau_SOMBA"],
    color=colors["oslo_blue"], edgecolors=colors["white"],
    label="samples", s=75, linewidth=1.5
)
ax1.set_xlabel("assumed dissolution fraction τ$_j$")
ax1.set_ylabel("estimated dissolution fraction τ$_j$")
ax1.set_title("a) Dissolution fraction comparison")
ax1.plot(
    [df["tau_assumed"].min(), df["tau_assumed"].max()],
    [df["tau_assumed"].min(), df["tau_assumed"].max()],
    color="gray", linestyle="--", linewidth=1, label="1:1 line"
)
ax1.legend()
ax1.tick_params(axis="both", direction="in", length=6)

# (b) Pre-weathering composition comparison (base cation concentration j)
ax2.scatter(
    df["pre_weathering_mix_j_molkg"], df["pre_weathering_mix_output_j_molkg"],
    color=colors["oslo_dark"], edgecolors=colors["white"],
    label="samples", s=75, linewidth=1.5
)
ax2.set_xlabel("pre-weathering [j] from SOMBA_start [mol kg$^{-1}$]")
ax2.set_ylabel(
    "pre-weathering [j] from SOMBA_tau_meta_simplified [mol kg$^{-1}$]")
ax2.set_title("b) Base cation concentration comparison")
ax2.plot(
    [df["pre_weathering_mix_j_molkg"].min(),
     df["pre_weathering_mix_j_molkg"].max()],
    [df["pre_weathering_mix_j_molkg"].min(),
     df["pre_weathering_mix_j_molkg"].max()],
    color="gray", linestyle="--", linewidth=1, label="1:1 line"
)
ax2.legend()
ax2.tick_params(axis="both", direction="in", length=6)

# (c) Application amount comparison
ax3.scatter(
    df["a_assumed_tha"], df["application_amount_output_tha"],
    color=colors["oslo_light"], edgecolors=colors["white"],
    label="samples", s=75, linewidth=1.5
)

ax3.plot(
    [df["a_assumed_tha"].min(), df["a_assumed_tha"].max()],
    [df["a_assumed_tha"].min(), df["a_assumed_tha"].max()],
    color="gray", linestyle="--", linewidth=1, label="1:1 line"
)

# Expected sampling-depth correction if sampling depth != mixing depth
if mixing_depth_m != sampling_depth_m:
    min_x = df["a_assumed_tha"].min()
    max_x = df["a_assumed_tha"].max()
    ax3.plot(
        [min_x, max_x],
        [min_x * (sampling_depth_m / mixing_depth_m),
         max_x * (sampling_depth_m / mixing_depth_m)],
        color=colors["oslo_gray"], linestyle="-.", linewidth=1,
        label="expected scaling (depth mismatch)"
    )

ax3.set_xlabel("assumed application amount [t ha$^{-1}$]")
ax3.set_ylabel(
    "estimated application amount [t ha$^{-1}$]"
)
ax3.set_title("c) Application amount comparison")
ax3.legend()
ax3.tick_params(axis="both", direction="in", length=6)

plt.tight_layout()

plt.savefig(
    "verification_data_plots/SOMBA_example_demonstration_2_simplified.pdf"
)

plt.show()
