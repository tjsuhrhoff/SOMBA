#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soil Mass Balance (SOMBA) Framework — Core Computational Functions
==================================================================

This module provides a set of functions implementing the Soil-Based Mass Balance
(SOMBA) framework for quantifying feedstock dissolution and compositional
changes in enhanced rock weathering (ERW) deployments. All functions operate
within a three-endmember system consisting of:
    • baseline soil (s)
    • fresh feedstock (f)
    • weathered feedstock residue (wf)

Different functions correspond to either the general three-endmember case or
a constrained version in which the weathered feedstock residue is assumed to
share selected properties with the baseline soil (e.g., mobile element
composition and bulk density). 

Function overview
-----------------
1. SOMBA_start
   Computes the initial (t = 0) soil–feedstock mixture composition immediately
   after deployment from baseline soil composition, feedstock composition, and
   applied mass ratio.

2. SOMBA_end_simplified
   Computes post-weathering mixture composition under the simplifying
   assumptions:
       [j]_wf = [j]_s and ρ_wf = ρ_s.
   This reduces the number of unknowns but still represents a three-endmember
   soil–feedstock–weathered-feedstock system.

3. SOMBA_end
   Computes post-weathering composition in the fully generalized three-endmember
   case, allowing [j]_wf ≠ [j]_s and ρ_wf ≠ ρ_s.

4. SOMBA_tau_simplified and SOMBA_tau_meta_simplified
   Estimate the dissolution fraction τ and, optionally additional derived
   meta data such as estimated applicatoin amount and initial soil composition
   for the constrained endmember system described above.

5. SOMBA_tau and SOMBA_tau_meta
   Same as for 4, but for the non-simplified case where the weathered feedstock
   endmember is defined explicitly.

Units and assumptions
---------------------
• Tracer concentrations ([j], [i]) should be supplied in consistent units,
  ideally mol kg-1.
• Bulk densities (ρ_s, ρ_f, ρ_wf): t m-3.
• Volumes (v_s, v_f, v_wf): m³ ha-1 or equivalent per-area reference.
• τ is dimensionless and physically ranges from 0 to 1 under well-resolved
  mixing geometry.
• Functions return modified pandas DataFrames without altering the original.

Reference
---------
Suhrhoff, T. J., et al. (2025). Environmental Science and Technology. An 
updated framework and signal-to-noise analysis of soil mass balance approaches 
for quantifying enhanced weathering on managed lands.
https://doi.org/10.1021/acs.est.5c08303

Author
------
Tim Jesper Suhrhoff
"""

import pandas as pd
import numpy as np


def SOMBA_start(
    data,
    soil_j_col_molkg,
    feedstock_j_col_molkg,
    r_m_t0,
    j_output_col_molkg,
    soil_i_col_molkg,
    feedstock_i_col_molkg,
    i_output_col_molkg
):
    """
    Compute the initial (t = 0) composition of the soil–feedstock mixture
    following deployment. The calculation uses a simple mass mixing ratio
    between baseline soil and feedstock, applied separately to a mobile
    tracer (j) and an immobile tracer (i). The function assumes that both
    tracers are provided in consistent concentration units (e.g., mol/kg)
    across soil and feedstock.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing the required soil and feedstock composition
        columns as well as the mixing ratio at t = 0.
    soil_j_col_molkg : str
        Column containing the soil concentration of the mobile tracer j.
    feedstock_j_col_molkg : str
        Column containing the feedstock concentration of tracer j.
    r_m_t0 : str
        Column containing the feedstock mass mixing ratio before weathering.
    j_output_col_molkg : str
        Name of the output column to store the computed mixed tracer j
        concentration.
    soil_i_col_molkg : str
        Column containing the soil concentration of the immobile tracer i.
    feedstock_i_col_molkg : str
        Column containing the feedstock concentration of tracer i.
    i_output_col_molkg : str
        Name of the output column to store the computed mixed tracer i
        concentration.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame containing two additional columns:
        the initial mixed concentrations of tracer j and tracer i.
    """

    # Copy input to avoid modifying the original data
    processed_data = data.copy()

    # ------------------------------------------------------------------
    # 1. Check that all required input columns are present
    # ------------------------------------------------------------------
    required_cols = [
        soil_j_col_molkg,
        feedstock_j_col_molkg,
        r_m_t0,
        soil_i_col_molkg,
        feedstock_i_col_molkg
    ]
    missing = [col for col in required_cols if col not in processed_data.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the input data: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # 2. Convert input columns to numeric and record any new NaN values.
    #    This safeguard helps detect non-numeric entries such as values
    #    below detection limit recorded as strings.
    # ------------------------------------------------------------------
    before_nan = processed_data[required_cols].isna().sum()
    processed_data[required_cols] = processed_data[required_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )
    after_nan = processed_data[required_cols].isna().sum()

    new_nans = (after_nan - before_nan)
    newly_introduced = new_nans[new_nans > 0]

    if not newly_introduced.empty:
        # Report which columns gained NaNs through coercion
        print(
            "Warning: Non-numeric values were converted to NaN in the "
            "following columns:\n"
            + "\n".join(f"  {col}: {count} values"
                        for col, count in newly_introduced.items())
        )

    # ------------------------------------------------------------------
    # 3. Compute the initial mixed concentrations for tracers j and i.
    #    The mixing ratio r_m_t0 controls the relative contribution of
    #    feedstock (r_m_t0) and soil (1 - r_m_t0).
    # ------------------------------------------------------------------
    processed_data[j_output_col_molkg] = (
        processed_data[feedstock_j_col_molkg] * processed_data[r_m_t0]
        + processed_data[soil_j_col_molkg] * (1.0 - processed_data[r_m_t0])
    )

    processed_data[i_output_col_molkg] = (
        processed_data[feedstock_i_col_molkg] * processed_data[r_m_t0]
        + processed_data[soil_i_col_molkg] * (1.0 - processed_data[r_m_t0])
    )

    return processed_data


def SOMBA_end_simplified(
    data,
    soil_j_col_molkg,
    F_d_col,
    output_post_weathering_j_col_molkg,
    v_feedstock_initial_m3ha_col,
    v_soil_initial_col_m3ha,
    rho_feedstock_col_tm3,
    rho_soil_col_tm3,
    feedstock_i_col_molkg,
    feedstock_j_col_molkg,
    soil_i_col_molkg,
    output_v_feedstock_end_col_m3ha,
    output_v_soil_end_col_m3ha,
    output_post_weathering_i_col_molkg
):
    """
    Compute the post-weathering composition of a soil–feedstock mixture under
    the simplifying assumption that the weathered feedstock residue has the
    same bulk density and mobile-element composition as the soil
    matrix. The function calculates final volumes of remaining feedstock and
    soil, and the resulting concentrations of a mobile tracer (j) and an
    immobile tracer (i) in the mixed system.

    All tracer concentrations should be supplied in consistent units,
    preferably mol kg-1, and densities in t m-3.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing composition, density, and deployment data.
    soil_j_col_molkg : str
        Column name for the soil concentration of tracer j.
    F_d_col : str
        Column name for the assumed feedstock dissolution fraction (dimensionless).
    output_post_weathering_j_col_molkg : str
        Output column name for the post-weathering concentration of tracer j.
    v_feedstock_initial_m3ha_col : str
        Column name for the initial feedstock volume (m3 ha-1).
    v_soil_initial_col_m3ha : str
        Column name for the initial soil volume within the mixed layer (m3 ha-1).
    rho_feedstock_col_tm3 : str
        Column name for feedstock bulk density (t m-3).
    rho_soil_col_tm3 : str
        Column name for soil bulk density (t m-3).
    feedstock_i_col_molkg : str
        Column name for the feedstock concentration of tracer i.
    feedstock_j_col_molkg : str
        Column name for the feedstock concentration of tracer j.
    soil_i_col_molkg : str
        Column name for the soil concentration of tracer i.
    output_v_feedstock_end_col_m3ha : str
        Output column name for post-weathering feedstock volume (m3 ha-1).
    output_v_soil_end_col_m3ha : str
        Output column name for post-weathering soil volume (m3 ha-1).
    output_post_weathering_i_col_molkg : str
        Output column name for post-weathering concentration of tracer i.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with added columns describing final
        volumes and post-weathering tracer concentrations.
    """

    # Create a working copy to avoid modifying the original dataframe
    processed_data = data.copy()

    # ------------------------------------------------------------------
    # 1. Verify that all required input columns are present
    # ------------------------------------------------------------------
    required_cols = [
        soil_j_col_molkg,
        F_d_col,
        v_feedstock_initial_m3ha_col,
        v_soil_initial_col_m3ha,
        rho_feedstock_col_tm3,
        rho_soil_col_tm3,
        feedstock_i_col_molkg,
        feedstock_j_col_molkg,
        soil_i_col_molkg
    ]
    missing = [col for col in required_cols if col not in processed_data.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the input data: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # 2. Convert required columns to numeric and report newly introduced NaNs
    # ------------------------------------------------------------------
    before_nan = processed_data[required_cols].isna().sum()
    processed_data[required_cols] = processed_data[required_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    after_nan = processed_data[required_cols].isna().sum()

    newly_introduced = (after_nan - before_nan)
    newly_introduced = newly_introduced[newly_introduced > 0]

    if not newly_introduced.empty:
        print(
            "Warning: Non-numeric values were converted to NaN in the "
            "following columns:\n"
            + "\n".join(f"  {col}: {count} values"
                        for col, count in newly_introduced.items())
        )

    # ------------------------------------------------------------------
    # 3. Compute final feedstock and soil volumes after weathering
    # ------------------------------------------------------------------
    processed_data[output_v_feedstock_end_col_m3ha] = (
        processed_data[v_feedstock_initial_m3ha_col]
        * (1.0 - processed_data[F_d_col])
    )

    processed_data[output_v_soil_end_col_m3ha] = (
        processed_data[v_soil_initial_col_m3ha]
        + (processed_data[v_feedstock_initial_m3ha_col]
           - processed_data[output_v_feedstock_end_col_m3ha])
    )

    # ------------------------------------------------------------------
    # 4. Compute post-weathering concentrations of tracer j
    # ------------------------------------------------------------------
    denominator_j = (
        processed_data[rho_soil_col_tm3] *
        processed_data[output_v_soil_end_col_m3ha]
        + processed_data[rho_feedstock_col_tm3] *
        processed_data[output_v_feedstock_end_col_m3ha]
    )

    processed_data[output_post_weathering_j_col_molkg] = (
        (processed_data[rho_soil_col_tm3]
         * processed_data[output_v_soil_end_col_m3ha]
         * processed_data[soil_j_col_molkg]
         + processed_data[rho_feedstock_col_tm3]
         * processed_data[output_v_feedstock_end_col_m3ha]
         * processed_data[feedstock_j_col_molkg])
        / denominator_j
    )

    # ------------------------------------------------------------------
    # 5. Compute post-weathering concentrations of tracer i
    # ------------------------------------------------------------------
    processed_data[output_post_weathering_i_col_molkg] = (
        (processed_data[rho_soil_col_tm3]
         * processed_data[output_v_soil_end_col_m3ha]
         * processed_data[soil_i_col_molkg]
         + processed_data[rho_feedstock_col_tm3]
         * processed_data[v_feedstock_initial_m3ha_col]
         * processed_data[feedstock_i_col_molkg])
        / denominator_j
    )

    return processed_data


def SOMBA_end(
    data,
    soil_j_col_molkg,
    soil_i_col_molkg,
    feedstock_j_col_molkg,
    feedstock_i_col_molkg,
    wf_j_col_molkg,
    wf_i_col_molkg,
    F_d_col,
    v_feedstock_initial_m3ha_col,
    v_soil_initial_m3ha_col,
    rho_soil_tm3_col,
    rho_feedstock_tm3_col,
    rho_wf_tm3_col,
    output_v_feedstock_end_m3ha_col,
    output_v_wf_end_m3ha_col,
    output_v_soil_end_m3ha_col,
    output_post_weathering_j_molkg_col,
    output_post_weathering_i_molkg_col
):
    """
    Compute the post-weathering composition of a soil–feedstock layer using
    a fully general three-endmember mass balance. The system consists of:

        • Soil (s)
        • Remaining fresh feedstock (f)
        • Weathered feedstock residue (wf)

    The dissolution fraction (F_d_col) determines the volumetric partition
    between fresh and weathered feedstock. Mobile (j) and immobile (i) tracer
    concentrations are then computed using density-weighted mass balance.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing tracer concentrations, densities, and
        deployment volumes.
    soil_j_col_molkg, soil_i_col_molkg : str
        Column names for soil tracer concentrations j and i.
    feedstock_j_col_molkg, feedstock_i_col_molkg : str
        Column names for fresh feedstock tracer concentrations.
    wf_j_col_molkg, wf_i_col_molkg : str
        Column names for the weathered feedstock tracer concentrations.
    F_d_col : str
        Column containing the dissolution fraction (dimensionless).
    v_feedstock_initial_m3ha_col, v_soil_initial_m3ha_col : str
        Columns containing initial feedstock and soil volumes (m3 ha-1).
    rho_soil_tm3_col, rho_feedstock_tm3_col, rho_wf_tm3_col : str
        Columns containing bulk densities (t m-3) of soil, fresh feedstock,
        and weathered feedstock.
    output_* : str
        Names of output columns for final volumes and post-weathering
        concentrations.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with added columns describing post-
        weathering component volumes and tracer concentrations.
    """

    # Create a working copy to avoid modifying the original dataframe
    df = data.copy()

    # ------------------------------------------------------------------
    # 1. Verify required columns exist
    # ------------------------------------------------------------------
    required_cols = [
        soil_j_col_molkg, soil_i_col_molkg,
        feedstock_j_col_molkg, feedstock_i_col_molkg,
        wf_j_col_molkg, wf_i_col_molkg,
        F_d_col,
        v_feedstock_initial_m3ha_col, v_soil_initial_m3ha_col,
        rho_soil_tm3_col, rho_feedstock_tm3_col, rho_wf_tm3_col
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the input data: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # 2. Convert to numeric and report newly introduced NaNs
    # ------------------------------------------------------------------
    before_nan = df[required_cols].isna().sum()
    df[required_cols] = df[required_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    after_nan = df[required_cols].isna().sum()

    newly_introduced = (after_nan - before_nan)
    newly_introduced = newly_introduced[newly_introduced > 0]

    if not newly_introduced.empty:
        print(
            "Warning: Non-numeric values were converted to NaN in the "
            "following columns:\n"
            + "\n".join(f"  {col}: {count} values"
                        for col, count in newly_introduced.items())
        )

    # ------------------------------------------------------------------
    # 3. Compute post-weathering endmember volumes
    # ------------------------------------------------------------------
    tau = df[F_d_col]                      # dissolution fraction
    v_f_t0 = df[v_feedstock_initial_m3ha_col]
    v_s_t0 = df[v_soil_initial_m3ha_col]

    v_f_tn = v_f_t0 * (1.0 - tau)          # remaining fresh feedstock
    v_wf_tn = v_f_t0 - v_f_tn              # weathered feedstock volume
    v_s_tn = v_s_t0                        # soil volume remains unchanged

    df[output_v_feedstock_end_m3ha_col] = v_f_tn
    df[output_v_wf_end_m3ha_col] = v_wf_tn
    df[output_v_soil_end_m3ha_col] = v_s_tn

    # ------------------------------------------------------------------
    # 4. Compute total mixture mass (density × volume)
    # ------------------------------------------------------------------
    rho_s = df[rho_soil_tm3_col]
    rho_f = df[rho_feedstock_tm3_col]
    rho_wf = df[rho_wf_tm3_col]

    mass_mix = (rho_s * v_s_tn) + (rho_f * v_f_tn) + (rho_wf * v_wf_tn)

    # ------------------------------------------------------------------
    # 5. Compute post-weathering tracer concentrations (j and i)
    # ------------------------------------------------------------------
    num_j = (
        rho_s * v_s_tn * df[soil_j_col_molkg]
        + rho_f * v_f_tn * df[feedstock_j_col_molkg]
        + rho_wf * v_wf_tn * df[wf_j_col_molkg]
    )
    num_i = (
        rho_s * v_s_tn * df[soil_i_col_molkg]
        + rho_f * v_f_tn * df[feedstock_i_col_molkg]
        + rho_wf * v_wf_tn * df[wf_i_col_molkg]
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        df[output_post_weathering_j_molkg_col] = num_j / mass_mix
        df[output_post_weathering_i_molkg_col] = num_i / mass_mix

    return df


def SOMBA_tau_simplified(
    data,
    soil_j_col_molkg,
    feedstock_j_col_molkg,
    soil_i_col_molkg,
    feedstock_i_col_molkg,
    post_weathering_j_col_molkg,
    post_weathering_i_col_molkg,
    rho_soil_col_tm3,
    rho_feedstock_col_tm3,
    output_tau_SOMBA_col
):
    """
    Compute the dissolution fraction τ (base-cation mass transfer coefficient)
    under the simplified SOMBA framework. The simplified case assumes that
    the weathered feedstock residue shares both the mobile-element
    composition and density of the soil after weathering.

    Two tracers are used:
        • j – a mobile/base-cation element (e.g., Ca, Mg)
        • i – an immobile element (e.g., Ti, Zr)

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing all required composition and density columns.
    soil_j_col_molkg : str
        Column name for soil concentration of the mobile tracer j [mol kg⁻¹].
    feedstock_j_col_molkg : str
        Column name for feedstock concentration of tracer j [mol kg⁻¹].
    soil_i_col_molkg : str
        Column name for soil concentration of the immobile tracer i [mol kg⁻¹].
    feedstock_i_col_molkg : str
        Column name for feedstock concentration of tracer i [mol kg⁻¹].
    post_weathering_j_col_molkg : str
        Column name for post-weathering concentration of tracer j [mol kg⁻¹].
    post_weathering_i_col_molkg : str
        Column name for post-weathering concentration of tracer i [mol kg⁻¹].
    rho_soil_col_tm3 : str
        Column name for soil bulk density [t m⁻³].
    rho_feedstock_col_tm3 : str
        Column name for feedstock bulk density [t m⁻³].
    output_tau_SOMBA_col : str
        Name of the output column storing the computed dissolution fraction τ.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame containing one additional column with
        the computed τ values.
    """

    # Copy input to avoid modifying the original dataframe
    processed_data = data.copy()

    # ------------------------------------------------------------------
    # 1. Verify that all required input columns are present
    # ------------------------------------------------------------------
    required_cols = [
        soil_j_col_molkg,
        feedstock_j_col_molkg,
        soil_i_col_molkg,
        feedstock_i_col_molkg,
        post_weathering_j_col_molkg,
        post_weathering_i_col_molkg,
        rho_soil_col_tm3,
        rho_feedstock_col_tm3
    ]
    missing = [col for col in required_cols
               if col not in processed_data.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the input data: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # 2. Convert required columns to numeric and report any new NaN values
    # ------------------------------------------------------------------
    before_nan = processed_data[required_cols].isna().sum()
    processed_data[required_cols] = processed_data[required_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )
    after_nan = processed_data[required_cols].isna().sum()

    new_nans = (after_nan - before_nan)
    newly_introduced = new_nans[new_nans > 0]

    if not newly_introduced.empty:
        print(
            "Warning: Non-numeric values were converted to NaN in the "
            "following columns:\n"
            + "\n".join(f"  {col}: {count} values"
                        for col, count in newly_introduced.items())
        )

    # ------------------------------------------------------------------
    # 3. Extract relevant columns for vectorized computation
    # ------------------------------------------------------------------
    rho_s = processed_data[rho_soil_col_tm3]
    rho_f = processed_data[rho_feedstock_col_tm3]

    j_s = processed_data[soil_j_col_molkg]
    j_f = processed_data[feedstock_j_col_molkg]
    j_mix = processed_data[post_weathering_j_col_molkg]

    i_s = processed_data[soil_i_col_molkg]
    i_f = processed_data[feedstock_i_col_molkg]
    i_mix = processed_data[post_weathering_i_col_molkg]

    # ------------------------------------------------------------------
    # 4. Compute intermediate shorthand terms (S13a–S13e)
    # ------------------------------------------------------------------
    a = rho_s * (j_mix - j_s)
    b = rho_f * (j_mix - j_f)
    c = rho_s * (i_mix - i_s)
    d = rho_f * (i_mix - i_f)
    e = rho_f * i_f

    denom = (a - b)

    # ------------------------------------------------------------------
    # 5. Compute endmember contributions and dissolution fraction τ
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        X_f = a / denom
        X_wf = (a * d - b * c) / (e * denom)
        tau = X_wf / (X_f + X_wf)

    # ------------------------------------------------------------------
    # 6. Store the computed dissolution fraction in the output column
    # ------------------------------------------------------------------
    processed_data[output_tau_SOMBA_col] = tau

    return processed_data


def SOMBA_tau_meta_simplified(
    data,
    soil_j_col_molkg,
    feedstock_j_col_molkg,
    soil_i_col_molkg,
    feedstock_i_col_molkg,
    post_weathering_j_col_molkg,
    post_weathering_i_col_molkg,
    rho_soil_col_tm3,
    rho_feedstock_col_tm3,
    v_sampled_layer_col_m3,
    output_diss_feedstock_i_col_molkg,
    output_Xs_soil_contribution_col,
    output_Xf_feed_contribution_col,
    output_Xwf_weathered_feed_contribution_col,
    output_tau_SOMBA_col,
    output_application_col_tha,
    output_pre_weathered_mix_i_col_molkg,
    output_pre_weathered_mix_j_col_molkg
):
    """
    Compute endmember contributions (X_s, X_f, X_wf), the dissolution fraction τ,
    the applied feedstock mass within the sampled volume, and the reconstructed
    pre-weathering (t = 0) mixture composition under the simplified SOMBA
    assumptions. The simplified formulation assumes that the weathered
    feedstock residue shares both the bulk density and mobile element
    composition of the surrounding soil.

    The calculation relies on two compositional tracers:
        • j: mobile/base cation tracer (e.g., Ca + Mg)
        • i: immobile element tracer (e.g., Ti or Zr)

    All tracer concentrations should be in consistent units, ideally mol kg⁻¹.
    Densities should be in t m⁻³. No clipping, bounding, or forcing is applied
    to mixing fractions; results retain physical and diagnostic freedom.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing all required composition and density columns.
    soil_j_col_molkg : str
        Column name for soil concentration of the mobile tracer j [mol kg⁻¹].
    feedstock_j_col_molkg : str
        Column name for feedstock concentration of tracer j [mol kg⁻¹].
    soil_i_col_molkg : str
        Column name for soil concentration of the immobile tracer i [mol kg⁻¹].
    feedstock_i_col_molkg : str
        Column name for feedstock concentration of tracer i [mol kg⁻¹].
    post_weathering_j_col_molkg : str
        Column name for post-weathering concentration of tracer j [mol kg⁻¹].
    post_weathering_i_col_molkg : str
        Column name for post-weathering concentration of tracer i [mol kg⁻¹].
    rho_soil_col_tm3 : str
        Column name for soil bulk density [t m⁻³].
    rho_feedstock_col_tm3 : str
        Column name for feedstock bulk density [t m⁻³].
    v_sampled_layer_col_m3 : str
        Column name for the sampled layer volume per area [m³ ha⁻¹].
    output_diss_feedstock_i_col_molkg : str
        Output column storing the derived concentration of dissolved feedstock i.
    output_Xs_soil_contribution_col : str
        Output column storing X_s (soil fraction in the mixture).
    output_Xf_feed_contribution_col : str
        Output column storing X_f (remaining fresh feedstock fraction).
    output_Xwf_weathered_feed_contribution_col : str
        Output column storing X_wf (weathered feedstock residue fraction).
    output_tau_SOMBA_col : str
        Output column storing the dissolution fraction τ.
    output_application_col_tha : str
        Output column storing estimated feedstock application [t ha⁻¹].
    output_pre_weathered_mix_i_col_molkg : str
        Output column storing reconstructed pre-weathering tracer i [mol kg⁻¹].
    output_pre_weathered_mix_j_col_molkg : str
        Output column storing reconstructed pre-weathering tracer j [mol kg⁻¹].

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with added columns for:
        • X_s, X_f, X_wf
        • τ (dissolution fraction)
        • estimated feedstock application (t ha⁻¹)
        • reconstructed pre-weathering composition (j and i)
    """

    # Copy input data to avoid altering the original
    df = data.copy()

    # ------------------------------------------------------------------
    # 1. Verify required columns are present
    # ------------------------------------------------------------------
    required_cols = [
        soil_j_col_molkg,
        feedstock_j_col_molkg,
        soil_i_col_molkg,
        feedstock_i_col_molkg,
        post_weathering_j_col_molkg,
        post_weathering_i_col_molkg,
        rho_soil_col_tm3,
        rho_feedstock_col_tm3,
        v_sampled_layer_col_m3
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the input data: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # 2. Convert required columns to numeric and report newly introduced NaNs
    # ------------------------------------------------------------------
    before_nan = df[required_cols].isna().sum()
    df[required_cols] = df[required_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )
    after_nan = df[required_cols].isna().sum()

    new_nans = (after_nan - before_nan)
    new_nans = new_nans[new_nans > 0]

    if not new_nans.empty:
        print(
            "Warning: Non-numeric values were converted to NaN in the "
            "following columns:\n"
            + "\n".join(f"  {col}: {count} values"
                        for col, count in new_nans.items())
        )

    # ------------------------------------------------------------------
    # 3. Compute shorthand coefficients (S13a–S13e)
    # ------------------------------------------------------------------
    rho_s = df[rho_soil_col_tm3]
    rho_f = df[rho_feedstock_col_tm3]
    v_mix = df[v_sampled_layer_col_m3]

    j_s = df[soil_j_col_molkg]
    j_f = df[feedstock_j_col_molkg]
    j_mix = df[post_weathering_j_col_molkg]

    i_s = df[soil_i_col_molkg]
    i_f = df[feedstock_i_col_molkg]
    i_mix = df[post_weathering_i_col_molkg]

    a = rho_s * (j_mix - j_s)
    b = rho_f * (j_mix - j_f)
    c = rho_s * (i_mix - i_s)
    d = rho_f * (i_mix - i_f)
    e = rho_f * i_f  # (S13e) correct sign convention

    denom = (a - b)

    # ------------------------------------------------------------------
    # 4. Compute endmember contributions (X_s, X_f, X_wf)
    #    Following S19d, S20, S24, and S15.
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        X_f = a / denom
        X_wf = (a * d - b * c) / (e * denom)
        X_s = 1.0 - X_f - X_wf

    df[output_Xs_soil_contribution_col] = X_s
    df[output_Xf_feed_contribution_col] = X_f
    df[output_Xwf_weathered_feed_contribution_col] = X_wf

    # ------------------------------------------------------------------
    # 5. Compute dissolution fraction τ (S28)
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = X_wf / (X_f + X_wf)
    df[output_tau_SOMBA_col] = tau

    # ------------------------------------------------------------------
    # 6. Compute feedstock application per sampled volume (t ha⁻¹)
    # ------------------------------------------------------------------
    df[output_application_col_tha] = (X_f + X_wf) * v_mix * rho_f

    # ------------------------------------------------------------------
    # 7. Compute pre-weathering (t = 0) mixed composition (S30 applied backward)
    # ------------------------------------------------------------------
    denom_pre = (rho_s * X_s) + (rho_f * (X_f + X_wf))

    df[output_pre_weathered_mix_i_col_molkg] = (
        (rho_s * X_s * i_s + rho_f * (X_f + X_wf) * i_f) / denom_pre
    )
    df[output_pre_weathered_mix_j_col_molkg] = (
        (rho_s * X_s * j_s + rho_f * (X_f + X_wf) * j_f) / denom_pre
    )

    # ------------------------------------------------------------------
    # 8. Compute dissolved feedstock immobile element concentration term
    #    (used for interpretability and consistency with derivations)
    # ------------------------------------------------------------------
    df[output_diss_feedstock_i_col_molkg] = i_s + (rho_f / rho_s) * i_f

    return df


def SOMBA_tau(
    data,
    soil_j_col_molkg,
    feedstock_j_col_molkg,
    wf_j_col_molkg,
    soil_i_col_molkg,
    feedstock_i_col_molkg,
    wf_i_col_molkg,
    post_weathering_j_col_molkg,
    post_weathering_i_col_molkg,
    rho_soil_col_tm3,
    rho_feedstock_col_tm3,
    rho_wf_col_tm3,
    output_tau_SOMBA_col
):
    """
    Compute the dissolution fraction τ for the general three-endmember
    soil–feedstock–weathered feedstock (s–f–wf) system. The formulation
    uses density-weighted mass balance of a mobile tracer (j) and an
    immobile tracer (i) to recover the endmember contributions:

        X_s  : soil fraction
        X_f  : remaining fresh feedstock fraction
        X_wf : weathered feedstock residue fraction

    The dissolution fraction is defined as:

        τ = X_wf / (X_f + X_wf)

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing required composition and density columns.
    soil_j_col_molkg, feedstock_j_col_molkg, wf_j_col_molkg : str
        Columns for tracer j in soil, fresh feedstock, and weathered residue.
    soil_i_col_molkg, feedstock_i_col_molkg, wf_i_col_molkg : str
        Columns for tracer i in the same three endmembers.
    post_weathering_j_col_molkg : str
        Column containing the post-weathering concentration of tracer j.
    post_weathering_i_col_molkg : str
        Column containing the post-weathering concentration of tracer i.
    rho_soil_col_tm3, rho_feedstock_col_tm3, rho_wf_col_tm3 : str
        Columns containing bulk densities (t m⁻³) of soil, feedstock,
        and weathered residue.
    output_tau_SOMBA_col : str
        Output column name for the computed dissolution fraction τ.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with the τ values added.

    """

    # Copy input to avoid modifying the original dataset
    processed_data = data.copy()

    # ------------------------------------------------------------------
    # 1. Verify required columns exist
    # ------------------------------------------------------------------
    required_cols = [
        soil_j_col_molkg, feedstock_j_col_molkg, wf_j_col_molkg,
        soil_i_col_molkg, feedstock_i_col_molkg, wf_i_col_molkg,
        post_weathering_j_col_molkg, post_weathering_i_col_molkg,
        rho_soil_col_tm3, rho_feedstock_col_tm3, rho_wf_col_tm3
    ]
    missing = [col for col in required_cols
               if col not in processed_data.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the input data: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # 2. Convert required columns to numeric and report any new NaN values
    # ------------------------------------------------------------------
    before_nan = processed_data[required_cols].isna().sum()
    processed_data[required_cols] = processed_data[required_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    after_nan = processed_data[required_cols].isna().sum()

    new_nans = (after_nan - before_nan)
    new_nans = new_nans[new_nans > 0]
    if not new_nans.empty:
        print(
            "Warning: Non-numeric values were converted to NaN in the "
            "following columns:\n"
            + "\n".join(f"  {col}: {count} values"
                        for col, count in new_nans.items())
        )

    # ------------------------------------------------------------------
    # 3. Extract vectors for computation
    # ------------------------------------------------------------------
    rho_s = processed_data[rho_soil_col_tm3]
    rho_f = processed_data[rho_feedstock_col_tm3]
    rho_wf = processed_data[rho_wf_col_tm3]

    j_s = processed_data[soil_j_col_molkg]
    j_f = processed_data[feedstock_j_col_molkg]
    j_wf = processed_data[wf_j_col_molkg]
    j_mix = processed_data[post_weathering_j_col_molkg]

    i_s = processed_data[soil_i_col_molkg]
    i_f = processed_data[feedstock_i_col_molkg]
    i_wf = processed_data[wf_i_col_molkg]
    i_mix = processed_data[post_weathering_i_col_molkg]

    # ------------------------------------------------------------------
    # 4. Compute density-weighted tracer offsets (eq. S3–S5 analogs)
    # ------------------------------------------------------------------
    a1 = rho_s * (j_mix - j_s)
    b1 = rho_f * (j_mix - j_f)
    c1 = rho_wf * (j_mix - j_wf)

    a2 = rho_s * (i_mix - i_s)
    b2 = rho_f * (i_mix - i_f)
    c2 = rho_wf * (i_mix - i_wf)

    # ------------------------------------------------------------------
    # 5. Closed-form solution for X_f and X_wf ratios (general case)
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -(c2 - (c1 * a2 / a1)) / (b2 - (b1 * a2 / a1))
        e = (-b1 * d - c1) / a1

        denom_X = e + d + 1.0
        X_wf = 1.0 / denom_X
        X_f = d * X_wf

        tau = X_wf / (X_f + X_wf)

    # ------------------------------------------------------------------
    # 6. Store output and return
    # ------------------------------------------------------------------
    processed_data[output_tau_SOMBA_col] = tau

    return processed_data


def SOMBA_tau_meta(
    data,
    soil_j_col_molkg,
    feedstock_j_col_molkg,
    wf_j_col_molkg,
    soil_i_col_molkg,
    feedstock_i_col_molkg,
    wf_i_col_molkg,
    post_weathering_j_col_molkg,
    post_weathering_i_col_molkg,
    rho_soil_col_tm3,
    rho_feedstock_col_tm3,
    rho_wf_col_tm3,
    v_sampled_layer_col_m3,
    output_Xs_soil_contribution_col,
    output_Xf_feed_contribution_col,
    output_Xwf_weathered_feed_contribution_col,
    output_tau_SOMBA_col,
    output_application_col_tha,
    output_pre_weathered_mix_i_col_molkg,
    output_pre_weathered_mix_j_col_molkg
):
    """
    Compute endmember contributions (X_s, X_f, X_wf), the dissolution fraction
    τ, estimated feedstock application (t ha⁻¹), and reconstructed pre-weathering
    mixture composition for the general three-endmember SOMBA system:

        Soil (s) – Fresh Feedstock (f) – Weathered Feedstock Residue (wf)

    Two tracers are used:
        • j : mobile cation tracer (e.g., Ca+Mg)
        • i : immobile tracer (e.g., Ti, Zr)

    Concentrations must be in consistent units (preferably mol kg⁻¹),
    densities in t m⁻³, and sampled layer volume in m³ ha⁻¹.

    The dissolution fraction is defined as:

        τ = X_wf / (X_f + X_wf)
    """

    processed_data = data.copy()

    # ------------------------------------------------------------------
    # 1. Verify required input columns
    # ------------------------------------------------------------------
    required_cols = [
        soil_j_col_molkg, feedstock_j_col_molkg, wf_j_col_molkg,
        soil_i_col_molkg, feedstock_i_col_molkg, wf_i_col_molkg,
        post_weathering_j_col_molkg, post_weathering_i_col_molkg,
        rho_soil_col_tm3, rho_feedstock_col_tm3, rho_wf_col_tm3,
        v_sampled_layer_col_m3
    ]
    missing = [col for col in required_cols if col not in processed_data.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the input data: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # 2. Convert required columns to numeric and report new NaNs
    # ------------------------------------------------------------------
    before_nan = processed_data[required_cols].isna().sum()
    processed_data[required_cols] = processed_data[required_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    after_nan = processed_data[required_cols].isna().sum()
    new_nans = (after_nan - before_nan)
    new_nans = new_nans[new_nans > 0]

    if not new_nans.empty:
        print(
            "Warning: Non-numeric values were converted to NaN in:\n"
            + "\n".join(f"  {col}: {count} values"
                        for col, count in new_nans.items())
        )

    # ------------------------------------------------------------------
    # 3. Extract numeric arrays
    # ------------------------------------------------------------------
    rho_s = processed_data[rho_soil_col_tm3]
    rho_f = processed_data[rho_feedstock_col_tm3]
    rho_wf = processed_data[rho_wf_col_tm3]
    v_mix = processed_data[v_sampled_layer_col_m3]

    j_s = processed_data[soil_j_col_molkg]
    j_f = processed_data[feedstock_j_col_molkg]
    j_wf = processed_data[wf_j_col_molkg]
    j_mix = processed_data[post_weathering_j_col_molkg]

    i_s = processed_data[soil_i_col_molkg]
    i_f = processed_data[feedstock_i_col_molkg]
    i_wf = processed_data[wf_i_col_molkg]
    i_mix = processed_data[post_weathering_i_col_molkg]

    # ------------------------------------------------------------------
    # 4. Compute density-weighted tracer offsets (eqs. S3–S5)
    # ------------------------------------------------------------------
    a1 = rho_s * (j_mix - j_s)
    b1 = rho_f * (j_mix - j_f)
    c1 = rho_wf * (j_mix - j_wf)

    a2 = rho_s * (i_mix - i_s)
    b2 = rho_f * (i_mix - i_f)
    c2 = rho_wf * (i_mix - i_wf)

    # ------------------------------------------------------------------
    # 5. Closed-form solution for mixing contributions
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -(c2 - (c1 * a2) / a1) / (b2 - (b1 * a2) / a1)
        e = (-b1 * d - c1) / a1

        denom = e + d + 1.0
        X_wf = 1.0 / denom
        X_f = d * X_wf
        X_s = e * X_wf

        tau = X_wf / (X_f + X_wf)

    # ------------------------------------------------------------------
    # 6. Feedstock application (t ha⁻¹)
    # ------------------------------------------------------------------
    application = (X_f + X_wf) * v_mix * rho_f

    # ------------------------------------------------------------------
    # 7. Pre-weathering mixture reconstruction (t = 0)
    # ------------------------------------------------------------------
    denom_pre = (rho_s * X_s) + (rho_f * (X_f + X_wf))
    mix_i_pre = (rho_s * X_s * i_s + rho_f * (X_f + X_wf) * i_f) / denom_pre
    mix_j_pre = (rho_s * X_s * j_s + rho_f * (X_f + X_wf) * j_f) / denom_pre

    # ------------------------------------------------------------------
    # 8. Store results
    # ------------------------------------------------------------------
    processed_data[output_Xs_soil_contribution_col] = X_s
    processed_data[output_Xf_feed_contribution_col] = X_f
    processed_data[output_Xwf_weathered_feed_contribution_col] = X_wf
    processed_data[output_tau_SOMBA_col] = tau
    processed_data[output_application_col_tha] = application
    processed_data[output_pre_weathered_mix_i_col_molkg] = mix_i_pre
    processed_data[output_pre_weathered_mix_j_col_molkg] = mix_j_pre

    return processed_data
