# SOMBA Tools – Supporting Files for “Updated framework and signal-to-noise analysis of soil mass balance approaches for quantifying enhanced weathering on managed lands”

This folder contains the supplementary computational tools provided alongside the manuscript on the SOMBA framework. These resources are intended to facilitate both initial analyses and deeper statistical investigations of oral bioavailability data.

This folder contains the supplementary computational tools provided alongside the manuscript describing the Soil-Based Mass Balance (SOMBA) framework for quantifying feedstock dissolution and soil compositional change in enhanced weathering (EW) systems. These resources are intended to support both initial exploratory analyses and rigorous MRV workflows based on measured soil and feedstock chemistry.


## License

This software is licensed under GNU General Public License v3 (GPL-3). This means you may copy, distribute and modify the software as long as you track changes/dates in source files. Any modifications to or software including (via compiler) GPL-licensed code must also be made available under the GPL along with build & install instructions.


## Collaboration

If you plan to use these tools in your research, please get in touch! We are happy to provide guidance on sampling strategies, statistical modeling, assist with methodological implementation, or explore joint development of new features.

If you use the SOMBA tools in work leading to a peer-reviewed publication, we would appreciate if you reached out early in the process. We are enthusiastic about supporting new applications of the framework and look forward to contributing as co-authors where appropriate.


## Contents

- `SOMBA.py`  
A Python script that defines the SOMBA functions for the simplified and non-simplified frameworks (SOMBA_start, SOMBA_end, SOMBA_end_simplified, SOMBA_tau, SOMBA_tau_meta, SOMBA_tau_simplified, SOMBA_tau_meta_simplified) used in the other Python manuscripts. See Suhrhoff et al. (2025) for details and derivations.

- `SOMBA_user_example_simplified.py`  
A Python script that loads user-provided input data and computes the SOMBA parameters as described in the main manuscript based on the simplified formulation in which the weathered feedstock residue is assumed to share selected properties with the baseline soil.

- `SOMBA_verification_simplified.py`  
A Python script demonstrating the internal consistency of the simplified SOMBA framework with synthetic data: the calculated SOMBA parameters are equivalent to the assumed a priori values 

- `SOMBA_template_simplified.xlsx`  
An Excel spreadsheet that calculates the dissolution fraction based on the simplified SOMBA framework. This template is designed for initial exploration and parameter testing, but ultimately any claims on dissolution fractions should always involve advanced statistical modeling such as Monte Carlo simulations

- `SOMBA_user_example.py`  
Example script demonstrating how to load user-provided soil and feedstock chemistry data, define deployment parameters, and compute the dissolution fraction and mixing contributions using the full (i.e., non simplified) SOMBA framework.

- `SOMBA_verification.py`  
Script that constructs synthetic datasets and demonstrates the internal consistency of the full SOMBA formulation. The script verifies that the inferred dissolution fraction and feedstock contribution match the assumed input values under controlled conditions.

- `SOMBA_template.xlsx`  
Spreadsheet implementation of the full SOMBA framework for initial exploration, teaching, and rapid scenario testing. Useful for conceptual understanding and sensitivity checks.

- `license.txt`  
License terms (GLP-3)


## Getting Started

### Prerequisites

To run the Python scripts, ensure you have Python 3.7+ installed with the following packages:

- `numpy`
- `pandas`
- `matplotlib` (optional, for visualization)
- `scipy`
