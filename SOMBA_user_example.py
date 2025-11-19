#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soil-Based Mass Balance (SOMBA) Demonstration Script
====================================================

This script demonstrates how to apply two explicit (non simplified) 
mass-balance solvers for Enhanced Rock Weathering (ERW) field data:

    (1) SOMBA_tau
        Computes only the dissolution fraction τ_SOMBA and appends it to the
        input DataFrame.

    (2) SOMBA_tau_meta
        Computes τ together with endmember contributions (soil, fresh
        feedstock, weathered feedstock residue) and auxiliary diagnostic
        variables. This version is useful for interpretation, plotting, and
        uncertainty propagation in downstream workflows.

Scientific Context
------------------
The Soil-Based Mass Balance Approach (SOMBA) quantifies how much of an applied
silicate feedstock has dissolved in the soil over a given time interval. The
method relies on mixing relationships between:

    • the original soil,
    • the freshly added feedstock,
    • the weathered feedstock residue remaining in the soil.


Important Guidance for Users
----------------------------
• A numerically reasonable τ value does *not* guarantee meaningful inference.
  Researchers must verify that the observed mixture composition lies inside the
  physically admissible ternary mixing domain and that inferred contributions
  remain non-negative.

• τ should be computed from representative *sample population means*, not by
  averaging τ values computed sample-by-sample, because the system is not
  self-averaging.

• Uncertainty propagation (e.g., Monte Carlo) is strongly recommended to assess
  robustness of the signal.

Input Data Requirements
-----------------------
The input CSV must contain composition and density values for each endmember,
along with the measured post-weathering mixture composition and sampled layer
volume. Units must be consistent:

    Concentrations:   mol kg⁻¹
    Densities:        t m⁻³
    Sampled volume:   m³ ha⁻¹

If column names differ, adjust them in the function calls below.

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

import numpy as np
import pandas as pd

# Import SOMBA solvers (ensure these functions are available in the environment)
from SOMBA import SOMBA_tau, SOMBA_tau_meta


# ============================================================================
# Load input dataset
# ============================================================================
# The example dataset bundled with this script contains:
#   soil_j_molkg, feedstock_j_molkg, wf_j_molkg,
#   soil_i_molkg, feedstock_i_molkg, wf_i_molkg,
#   rho_soil_tm3, rho_feedstock_tm3, rho_wf_tm3,
#   v_sampled_layer_m3ha,
#   post_weathering_mix_j_molkg, post_weathering_mix_i_molkg
#
# Modify the file path below as needed for your workflow.
# ============================================================================
df = pd.read_csv("user_example_data/example_data.csv")


# ============================================================================
# Example 1 — Compute τ only (compact output)
# ============================================================================
# This call adds a single τ column (`tau_SOMBA`) to the DataFrame. Use this
# option if only the dissolution fraction is required.
# ============================================================================
df = SOMBA_tau(
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
    output_tau_SOMBA_col="tau_SOMBA"
)


# ============================================================================
# Example 2 — Compute τ with endmember contributions and meta variables
# ============================================================================
# This version is recommended for scientific interpretation and uncertainty
# analysis, as it provides X_s, X_f, X_wf, reconstructed pre-weathering
# compositions, and reconstructed ropck application amount for diagnostic
# evaluation.
# ============================================================================
df = SOMBA_tau_meta(
    data=df,
    # === Composition inputs (mol kg-1) ===
    soil_j_col_molkg="soil_j_molkg",
    feedstock_j_col_molkg="feedstock_j_molkg",
    wf_j_col_molkg="wf_j_molkg",
    soil_i_col_molkg="soil_i_molkg",
    feedstock_i_col_molkg="feedstock_i_molkg",
    wf_i_col_molkg="wf_i_molkg",
    post_weathering_j_col_molkg="post_weathering_mix_j_molkg",
    post_weathering_i_col_molkg="post_weathering_mix_i_molkg",
    # === Physical parameters ===
    rho_soil_col_tm3="rho_soil_tm3",
    rho_feedstock_col_tm3="rho_feedstock_tm3",
    rho_wf_col_tm3="rho_wf_tm3",
    v_sampled_layer_col_m3="v_sampled_layer_m3ha",
    # === Outputs (choose names suitable for your workflow) ===
    output_Xs_soil_contribution_col="X_s",
    output_Xf_feed_contribution_col="X_f",
    output_Xwf_weathered_feed_contribution_col=("X_wf"),
    output_tau_SOMBA_col="tau_SOMBA_meta",
    output_application_col_tha="application_amount_output_tha",
    output_pre_weathered_mix_i_col_molkg=("pre_weathering_mix_i_molkg"),
    output_pre_weathered_mix_j_col_molkg=("pre_weathering_mix_j_molkg")
)


# ============================================================================
# Export results
# ============================================================================
df.to_csv("user_example_data/example_output_data.csv", index=False)
