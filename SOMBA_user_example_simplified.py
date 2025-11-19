#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soil-Based Mass Balance (SOMBA) Demonstration Script — Simplified Case
======================================================================

This script demonstrates how to apply the simplified soil-based mass balance
solvers for Enhanced Rock Weathering (ERW) field datasets:

    (1) SOMBA_tau_simplified
        Computes only the dissolution fraction τ_SOMBA and appends it to the
        input DataFrame.

    (2) SOMBA_tau_meta_simplified
        Computes τ together with endmember contribution fractions and selected
        diagnostic outputs. This version is useful for interpretation, plotting,
        and uncertainty analysis.

Scientific Context
------------------
The simplified SOMBA formulation is a constrained version of the three-endmember
soil mass balance model described in Suhrhoff et al. (2025). Simplification is
achieved by assuming that the weathered feedstock residue shares selected
properties (e.g., mobile element composition and bulk density) with the
original soil, reducing the number of free parameters while preserving the
physical interpretation of the dissolution fraction τ.

Important Guidance for Users
----------------------------
• A numerically reasonable τ value does not guarantee physical validity. Users
  must confirm that inferred contributions are non-negative and that the
  measured mixture lies within the admissible mixing domain.

• Due to non-self-averaging behavior, τ should be computed from representative
  *sample population means*. Averaging τ values computed on individual samples
  is not appropriate and can bias results, especially when some measurements lie
  outside the mixing triangle.

• Uncertainty propagation (e.g., Monte Carlo) is recommended to test robustness.

Input Data Requirements
-----------------------
The input CSV must supply endmember compositions, densities, and the observed
post-weathering mixture. Units must be consistent:

    Concentrations:   mol kg⁻¹
    Densities:        t m⁻³
    Sampled volume:   m³ ha⁻¹

Adjust column names in the function calls below if working with custom data.

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

import pandas as pd

# Import simplified SOMBA solvers
from SOMBA import (
    SOMBA_tau_simplified,
    SOMBA_tau_meta_simplified
)

# ============================================================================
# Load input dataset
# ============================================================================
df = pd.read_csv("user_example_data/example_data_simplified.csv")


# ============================================================================
# Example 1 — Compute τ only (compact output)
# ============================================================================
# This call appends a single column (tau_SOMBA) to the DataFrame. Use this
# function when only the dissolution fraction is required.
# ============================================================================
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


# ============================================================================
# Example 2 — Compute τ with endmember contributions and diagnostic variables
# ============================================================================
# This version provides τ_SOMBA along with inferred contributions (X_s, X_f,
# X_wf), reconstructed pre-weathering mixture compositions, and estimated
# rock application amount as inferred from post-weathering sample composition.
# This is typically the recommended workflow for scientific interpretation and
# downstream sensitivity or uncertainty analysis.
# ============================================================================
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
    # define names for output columns
    output_diss_feedstock_i_col_molkg="diss_feedstock_i_molkg",
    output_Xs_soil_contribution_col="X_s",
    output_Xf_feed_contribution_col="X_f",
    output_Xwf_weathered_feed_contribution_col="X_wf",
    output_tau_SOMBA_col="tau_SOMBA_meta",
    output_application_col_tha="application_amount_output_tha",
    output_pre_weathered_mix_i_col_molkg="pre_weathering_mix_i_molkg",
    output_pre_weathered_mix_j_col_molkg="pre_weathering_mix_j_molkg"
)


# ============================================================================
# Export results
# ============================================================================
df.to_csv("user_example_data/example_output_data_simplified.csv", index=False)
