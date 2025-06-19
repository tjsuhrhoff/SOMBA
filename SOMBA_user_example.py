#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 9th, 2025

This script demonstrates how to use the SOMBA_tau and SOMBA_tau_meta functions 
defined for the soil based mass balance approach to quantifying the dissolution
fraction / mass transport coefficient in EW field trials (see Suhrhoff et al. 
2025).

The script has two main parts:
    1.  Data is loaded from a csv file. 
    2.  SOMBA_tau and SOMBA_tau_meta are called based on the input data frame.  
        If users use their own data they may need to adjust the column names to
        reflect the naming convention of their input data.


Please note that even if the code gives a sensible dissolution fraction, it 
does not necessarily mean that results are statistically significant or 
even physical - in addition to the analysis here, users should plot the data to
visually inspect that data falls within the domain of the mixing model or
otherwise make sure that all endmemer contributions are positive. In addition,
users have to perform statistical analyses considering all sources of 
uncertainties (e.g., using Monte Carlo simulations) to investigate whether the 
signal is robust statistically. Furthermore, due to the non-self-averaging
behavior, dissolution fractions should always be calculated from sample 
population means. Calculating dissolution fractions for individual samples and
then taking their average will not give the same result and will often be 
inaccurate, particularly when some samples fall outside of the mixing triangle. 
See Suhrhoff et al. (2025) for more detail.


@author: Τim Jesper Suhrhoff
"""


import pandas as pd
from SOMBA import SOMBA_tau, SOMBA_tau_meta


############################ loading of input data ############################
df = pd.read_csv("user_example_data/example_data.csv")


####################### example use of SOMBA functions ########################
# If you have imported your own data, make sure to adjust the column names
# according to the names in your inputs when calling the SOMBA functions.

# Compute tau from SOMBA_tau function. Note that the the function
# SOMBA_tau_meta also calculates the dissolution fraction. When SOMBA_tau_meta
# is called, it is therefore not necessary to also call SOMBA_tau - it is only
# demonstrated here to provide a template for when users are only interested in
# calculating tau.
df = SOMBA_tau(
    data=df,
    soil_j_col_molkg='soil_j_molkg',   # adjust column name to reflect input
    feedstock_j_col_molkg='feedstock_j_molkg',
    soil_i_col_molkg='soil_i_molkg',
    feedstock_i_col_molkg='feedstock_i_molkg',
    post_weathering_j_col_molkg='post_weathering_mix_j_molkg',
    post_weathering_i_col_molkg='post_weathering_mix_i_molkg',
    rho_soil_col_tm3='rho_soil_tm3',
    rho_feedstock_col_tm3='rho_feedstock_tm3',
    output_tau_SOMBA_col='tau_SOMBA'  # define names for output columns
)

# Compute tau from SOMBA in addition to additional metadata based on function
# SOMBA_tau_meta. Note that if SOMBA_tau is also called, the tau output
# parameter should have a different name.
df = SOMBA_tau_meta(
    data=df,
    soil_j_col_molkg='soil_j_molkg',    # adjust column name to reflect input
    feedstock_j_col_molkg='feedstock_j_molkg',
    soil_i_col_molkg='soil_i_molkg',
    feedstock_i_col_molkg='feedstock_i_molkg',
    post_weathering_j_col_molkg='post_weathering_mix_j_molkg',
    post_weathering_i_col_molkg='post_weathering_mix_i_molkg',
    rho_soil_col_tm3='rho_soil_tm3',
    rho_feedstock_col_tm3='rho_feedstock_tm3',
    v_sampled_layer_col_m3='v_sampled_layer_m3ha',
    # define names for output columns
    output_diss_feedstock_i_col_molkg='diss_feedstock_i_molkg',
    output_Xs_soil_contribution_col='Xs_soil_contribution',
    output_Xf_feed_contribution_col='Xf_feed_contribution',
    output_Xd_diss_feed_contribution_col='Xwf_weathered_feed_contribution',
    output_tau_SOMBA_col='tau_SOMBA_meta',
    output_application_col_tha='application_amount_output_tha',
    output_pre_weathered_mix_i_col_molkg='pre_weathering_mix_output_i_molkg',
    output_pre_weathered_mix_j_col_molkg='pre_weathering_mix_output_j_molkg'
)


# Now we export the generated model outputs.
df.to_csv("user_example_data/example_ouput_data.csv", index=False)
