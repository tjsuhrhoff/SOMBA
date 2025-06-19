#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 25 2025

This script defines functions that align with the equations in Suhrhoff et al.
2025 and use soil mass balance approaches to calculate parameters from 
deployment data as well as post-weathering samples. The functions defined here
are:
    
1.  SOMBA_start: calculates pre-weathering soil+rock powder mix composition 
    from deployment data.
2.  SOMBA_end: estimates post-weathering composition from deployment data 
    as well as assumed fraction of feedstock dissolution.
3.  SOMBA_tau: calculates the rock powder dissolution fraction based on post-
    weathering sample composition.
4.  SOMBA_tau_meta: Same as SOMBA_tau, but computes further meta-data in 
    addition to the cation mass transfer coefficient/ dissolution fraction.

@author: Tim Jesper Suhrhoff
"""

import pandas as pd
import numpy as np


def SOMBA_start(
    data,
    soil_j_col_molkg, feedstock_j_col_molkg, r_m_t0, j_output_col_molkg,
    soil_i_col_molkg, feedstock_i_col_molkg, i_output_col_molkg
):
    """
    This function calculates the pre-weathering mix composition of a feedstock
    soil mixture based on deployment parameters.

    Although output will be correct as long as all mobile (j) and immobile (i)
    element concentrations have the same units, the use of molar concentrations
    is recommended to make easier using sums of elemental concentrations
    into the function.

    Parameters:
    -   data (pd.DataFrame): The input dataframe containing the required 
        columns.
    -   soil_j_col_molkg (str): Column for soil base cation concentration.
        [mol/kg]
    -   feedstock_j_col_molkg (str): Column for feedstock base cation
        concentration. [mol/kg]
    -   r_m_t0 (str): Column for feedstock mass mixing ratio at t=0 (i.e.,
        right after delopyment). []
    -   j_output_col_molkg (str): Column name for the calculated pre-weathering
        base cation concentration. To be defined when the function is
        called. [mol/kg]
    -   soil_i_col_molkg (str): Column for soil immobile element concentration.
        [mol/kg]
    -   feedstock_i_col_molkg (str): Column for feedstock immobile element
        concentration. [mol/kg]
    -   i_output_col_molkg (str): Column name for the calculated pre-weathering
        immobile element concentration. To be defined when the function is
        called. [mol/kg]


    Returns:
    processed_data (pd.DataFrame): dataframe with the new columns added.
    """

    processed_data = data.copy()

    # Ensure numeric conversion
    cols_to_convert = [soil_j_col_molkg, feedstock_j_col_molkg,
                       r_m_t0, soil_i_col_molkg, feedstock_i_col_molkg]
    processed_data[cols_to_convert] = processed_data[cols_to_convert].apply(
        pd.to_numeric, errors='coerce')

    # Compute pre-weathering mix concentrations for all rows
    processed_data[j_output_col_molkg] = (
        processed_data[feedstock_j_col_molkg] * processed_data[r_m_t0] +
        processed_data[soil_j_col_molkg] * (1 - processed_data[r_m_t0])
    )

    processed_data[i_output_col_molkg] = (
        processed_data[feedstock_i_col_molkg] * processed_data[r_m_t0] +
        processed_data[soil_i_col_molkg] * (1 - processed_data[r_m_t0])
    )

    return processed_data


def SOMBA_end(
    data,
    soil_j_col_molkg, F_d_col,
    output_post_weathering_j_col_molkg, v_feedstock_initial_m3ha_col,
    v_soil_initial_col_m3ha, rho_feedstock_col_tm3, rho_soil_col_tm3,
    feedstock_i_col_molkg, feedstock_j_col_molkg, soil_i_col_molkg,
    output_v_feedstock_end_col_m3ha, output_v_soil_end_col_m3ha,
    output_post_weathering_i_col_molkg
):
    """
    This function calculates the post-weathering soil-feedstock mix composition
    based on deployment data and assumed feedstock dissolution fractions,
    taking into account enrichment immobile elements due to feedstock mass 
    loss.

    Although output will be correct as long as all mobile (j) and immobile (i)
    element concentrations have the same units, the use of molar concentrations
    is recommended to make easier using sums of elemental concentrations
    into the function.

    Parameters:
    -   data (pd.DataFrame): The input dataframe containing the required 
        columns.
    -   soil_j_col_molkg (str): Column name for soil base cation concentration
        [mol/kg].
    -   F_d_col (str): Column name for the feedstock dissolution fraction 
        (base cation mass transfer coefficient) be modeled [].
    -   output_post_weathering_j_col_molkg (str): Column name for the 
        calculated post-weathering base cation concentrations. [mol/kg]
    -   v_feedstock_initial_m3ha_col (str): Column name for the initial volume 
        of feedstock applied to one hectare. [m3/ha]
    -   v_soil_initial_col_m3ha (str): Column name for the initial volume of 
        within the soil-feedstock mixed layer before weathering. [m3/ha]
    -   rho_feedstock_col_tm3 (str): Column name for the density of the 
        feedstock. [t/m3]
    -   rho_soil_col_tm3 (str): Column name for the density of soil. [t/m3]
    -   feedstock_i_col_molkg (str): Column name for the immobile element
        concentration of the feedstock. [mol/kg]
    -   soil_i_col_molkg (str): Column name for the immobile element
        concentration in soil. [mol/kg]
    -   output_v_feedstock_end_col_m3ha (str): Column name for the calculated
        post-weathering feedstock volume [m3/ha]
    -   output_v_soil_end_col_m3ha (str): Column name for the calculated 
        post-weathering volume of soil within the feedstock-soil mixed layer. 
        [m3/ha]
    -   output_post_weathering_i_col_molkg (str): Column name for the 
        calculated post-weathering immobile element concentration. [mol/kg]

    Returns:
    processed_data (pd.DataFrame): dataframe with the new columns added.
    """

    processed_data = data.copy()

    # Ensure numeric conversion for all columns
    for col in [
        soil_j_col_molkg, F_d_col,
        v_feedstock_initial_m3ha_col, v_soil_initial_col_m3ha,
        rho_feedstock_col_tm3, rho_soil_col_tm3,
        feedstock_i_col_molkg, feedstock_j_col_molkg, soil_i_col_molkg
    ]:
        processed_data[col] = pd.to_numeric(
            processed_data[col], errors='coerce')

    # Step 1: Calculate feedstock and soil volumes after weathering
    processed_data[output_v_feedstock_end_col_m3ha] = (
        processed_data[v_feedstock_initial_m3ha_col] *
        (1 - processed_data[F_d_col])
    )

    processed_data[output_v_soil_end_col_m3ha] = (
        processed_data[v_soil_initial_col_m3ha] + (
            processed_data[v_feedstock_initial_m3ha_col] -
            processed_data[output_v_feedstock_end_col_m3ha])
    )

    # Step 2: Calculate post-weathering base cation concentration
    processed_data[output_post_weathering_j_col_molkg] = (
        (processed_data[rho_soil_col_tm3] *
         processed_data[output_v_soil_end_col_m3ha] *
         processed_data[soil_j_col_molkg] +
         processed_data[rho_feedstock_col_tm3] *
         processed_data[output_v_feedstock_end_col_m3ha] *
         processed_data[feedstock_j_col_molkg]) /
        (processed_data[rho_soil_col_tm3] *
         processed_data[output_v_soil_end_col_m3ha] +
         processed_data[rho_feedstock_col_tm3] *
         processed_data[output_v_feedstock_end_col_m3ha])
    )

    # Step 3: Calculate post-weathering immobile element concentration
    processed_data[output_post_weathering_i_col_molkg] = (
        (processed_data[rho_soil_col_tm3] *
         processed_data[output_v_soil_end_col_m3ha] *
         processed_data[soil_i_col_molkg] +
         processed_data[rho_feedstock_col_tm3] *
         processed_data[v_feedstock_initial_m3ha_col] *
         processed_data[feedstock_i_col_molkg]) /
        (processed_data[rho_soil_col_tm3] *
         processed_data[output_v_soil_end_col_m3ha] +
         processed_data[rho_feedstock_col_tm3] *
         processed_data[output_v_feedstock_end_col_m3ha])
    )

    return processed_data


def SOMBA_tau(
    data,
    soil_j_col_molkg, feedstock_j_col_molkg, soil_i_col_molkg,
    feedstock_i_col_molkg, post_weathering_j_col_molkg,
    post_weathering_i_col_molkg, rho_soil_col_tm3,
    rho_feedstock_col_tm3, output_tau_SOMBA_col
):
    """
    This function calculates the base cation mass transfer coefficient τ-i 
    (here called tau) from post-weathering sample composition, as well as
    soil baseline and feedstock compositions. The formulation of the
    soil mass balance approach takes into account enrichment of immobile (and
    mobile) elements due to feedstock mass loss from the system. This function
    only returns the mass transfer coefficient without further meta-data. For
    this functionality, see function SOMBA_tau_meta.

    Although output will be correct as long as all mobile (j) and immobile (i)
    element concentrations have the same units, the use of molar concentrations
    is recommended to make easier using sums of elemental concentrations
    into the function.

    Parameters:
    -   data (pd.DataFrame): Input dataframe with required columns.
    -   soil_j_col_molkg (str): Column name for soil base cation concentration
        [mol/kg].
    -   feedstock_j_col_molkg (str): Column name for feedstock base cation
        concentration. [mol/kg]
    -   soil_i_col_molkg (str): Column name for soil immobile element 
        concentration. [mol/kg]
    -   feedstock_i_col_molkg (str): Column name for feedstock immobile 
        element concentration. [mol/kg]
    -   post_weathering_j_col_molkg (str): Column name for post-weathering
        mix base cation concentration. When deployment data is used, this 
        corresponds to the composition of post-deployment samples sampled after 
        weathering has occured.[mol/kg]
    -   post_weathering_i_col_molkg (str): Column name for post-weathering mix 
        immobile element concentration. When deployment data is used, this 
        corresponds to the composition of post-deployment samples sampled after 
        weathering has occured.[mol/kg]
    -   rho_soil_col_tm3 (str): Column name for soil density. [t/m3]
    -   rho_feedstock_col_tm3 (str): Column name for feedstock density. [t/m3]
    -   output_tau_SOMBA_col (str): Column name for base cation mass transfer
        coefficient/dissolution fraction output. []

    Returns:
    processed_data (pd.DataFrame): Dataframe with only the dissolution fraction
    column added.
    """

    processed_data = data.copy()

    # Ensure numeric conversion
    for col in [soil_j_col_molkg, feedstock_j_col_molkg, soil_i_col_molkg,
                feedstock_i_col_molkg, post_weathering_j_col_molkg,
                post_weathering_i_col_molkg, rho_soil_col_tm3,
                rho_feedstock_col_tm3]:
        processed_data[col] = pd.to_numeric(
            processed_data[col], errors='coerce')

    # Iterate over all rows to compute dissolution fraction
    for index, row in processed_data.iterrows():

        # Extract values for endmember compositions and densities
        rho_s = row[rho_soil_col_tm3]
        rho_f = row[rho_feedstock_col_tm3]
        cat_s = row[soil_j_col_molkg]
        cat_f = row[feedstock_j_col_molkg]
        cat_mix = row[post_weathering_j_col_molkg]
        im_s = row[soil_i_col_molkg]
        im_f = row[feedstock_i_col_molkg]
        im_mix = row[post_weathering_i_col_molkg]

        # Compute intermediate values (eq. S13a-S13e)
        a = rho_s * (cat_mix - cat_s)
        b = rho_f * (cat_mix - cat_f)
        c = rho_s * (im_mix - im_s)
        d = rho_f * (im_mix - im_f)
        e = rho_f * im_f

        denominator = a - b
        if denominator == 0 or e == 0:
            tau = np.nan
        else:
            X_f = a / denominator
            X_wf = (a * d - b * c) / (e * denominator)
            tau = X_wf / (X_wf + X_f)

        processed_data.at[index, output_tau_SOMBA_col] = tau

    return processed_data


def SOMBA_tau_meta(
    data,
    soil_j_col_molkg, feedstock_j_col_molkg, soil_i_col_molkg,
    feedstock_i_col_molkg, post_weathering_j_col_molkg,
    post_weathering_i_col_molkg,  rho_soil_col_tm3, rho_feedstock_col_tm3,
    v_sampled_layer_col_m3, output_diss_feedstock_i_col_molkg,
    output_Xs_soil_contribution_col, output_Xf_feed_contribution_col,
    output_Xd_diss_feed_contribution_col, output_tau_SOMBA_col,
    output_application_col_tha, output_pre_weathered_mix_i_col_molkg,
    output_pre_weathered_mix_j_col_molkg
):
    """
    [Docstring remains unchanged]
    """

    processed_data = data.copy()

    # Ensure numeric conversion
    for col in [soil_j_col_molkg, feedstock_j_col_molkg, soil_i_col_molkg,
                feedstock_i_col_molkg, post_weathering_j_col_molkg,
                post_weathering_i_col_molkg, rho_soil_col_tm3,
                rho_feedstock_col_tm3, v_sampled_layer_col_m3]:
        processed_data[col] = pd.to_numeric(
            processed_data[col], errors='coerce')

    # Compute dissolved feedstock immobile element concentration and store
    processed_data[output_diss_feedstock_i_col_molkg] = (
        (processed_data[soil_i_col_molkg] +
         processed_data[rho_feedstock_col_tm3] /
         processed_data[rho_soil_col_tm3] *
         processed_data[feedstock_i_col_molkg])
    )

    # Iterate over all rows to compute dissolution fraction and meta-data
    for index, row in processed_data.iterrows():

        # Extract values for endmember compositions and densities
        rho_f = row[rho_feedstock_col_tm3]
        rho_s = row[rho_soil_col_tm3]
        cat_s = row[soil_j_col_molkg]
        cat_f = row[feedstock_j_col_molkg]
        cat_mix = row[post_weathering_j_col_molkg]
        im_s = row[soil_i_col_molkg]
        im_f = row[feedstock_i_col_molkg]
        im_mix = row[post_weathering_i_col_molkg]

        # Compute intermediate values (eq. S13a-S13e)
        a = rho_s * (cat_mix - cat_s)
        b = rho_f * (cat_mix - cat_f)
        c = rho_s * (im_mix - im_s)
        d = rho_f * (im_mix - im_f)
        e = rho_f * im_f

        # Avoid division by zero
        if (a - b) == 0 or (e * (a - b)) == 0:
            X_f = np.nan
            X_wf = np.nan
            X_s = np.nan
        else:
            X_f = a / (a - b)
            X_wf = (a * d - b * c) / (e * (a - b))
            X_s = 1 - X_f - X_wf

        # Store the contributions
        processed_data.at[index, output_Xs_soil_contribution_col] = X_s
        processed_data.at[index, output_Xf_feed_contribution_col] = X_f
        processed_data.at[index, output_Xd_diss_feed_contribution_col] = X_wf

    # Compute base cation mass transfer coefficient and store as output
    processed_data[output_tau_SOMBA_col] = (
        processed_data[output_Xd_diss_feed_contribution_col] / (
            processed_data[output_Xd_diss_feed_contribution_col] +
            processed_data[output_Xf_feed_contribution_col]
        ))

    # Compute amount of feedstock added to the sampled soil volume
    processed_data[output_application_col_tha] = (
        (processed_data[output_Xf_feed_contribution_col] +
         processed_data[output_Xd_diss_feed_contribution_col]) *
        processed_data[v_sampled_layer_col_m3] *
        processed_data[rho_feedstock_col_tm3]
    )

    # Compute pre-weathering soil-feedstock mix composition (immobile and mobile)
    processed_data[output_pre_weathered_mix_i_col_molkg] = (
        ((processed_data[rho_soil_col_tm3] *
          processed_data[output_Xs_soil_contribution_col] *
          processed_data[soil_i_col_molkg]) +
         (processed_data[rho_feedstock_col_tm3] * (
             processed_data[output_Xf_feed_contribution_col] +
             processed_data[output_Xd_diss_feed_contribution_col]) *
            processed_data[feedstock_i_col_molkg])) /
        ((processed_data[rho_soil_col_tm3] *
          processed_data[output_Xs_soil_contribution_col]) +
         (processed_data[rho_feedstock_col_tm3] * (
             processed_data[output_Xf_feed_contribution_col] +
             processed_data[output_Xd_diss_feed_contribution_col])))
    )

    processed_data[output_pre_weathered_mix_j_col_molkg] = (
        ((processed_data[rho_soil_col_tm3] *
          processed_data[output_Xs_soil_contribution_col] *
          processed_data[soil_j_col_molkg]) +
         (processed_data[rho_feedstock_col_tm3] * (
             processed_data[output_Xf_feed_contribution_col] +
             processed_data[output_Xd_diss_feed_contribution_col]) *
            processed_data[feedstock_j_col_molkg])) /
        ((processed_data[rho_soil_col_tm3] *
          processed_data[output_Xs_soil_contribution_col]) +
         (processed_data[rho_feedstock_col_tm3] * (
             processed_data[output_Xf_feed_contribution_col] +
             processed_data[output_Xd_diss_feed_contribution_col])))
    )

    return processed_data
