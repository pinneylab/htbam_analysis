# htbam_analysis
Duncan Muir
Nicholas Freitas
Jonathan Zhang

Credits: Daniel Mohktari and Scott Longwell
___
![BuildStatus](https://github.com/pinneylab/htbam_analysis/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/pinneylab/htbam_analysis/badge.svg?branch=main)](https://coveralls.io/github/pinneylab/htbam_analysis?branch=main)

## Overview
This is a *WORK IN PROGRESS* package for processing and analysing various assays from the HTBAM/HTMEK platform.

## Installation

### Conda Environment Setup

(Recommended) Create a fresh conda environment with python=3.12 using:

```conda create -n htbam_analysis python=3.12```

and activate using:

```conda activate htbam_analysis```

#### Install Stable Release via Wheel File
Download the latest wheel file from the [Release Page](https://github.com/pinneylab/htbam_analysis/releases)

Then, install the package to your conda environment using:

```pip install /path/to/downloaded/wheel.file```

#### For latest code (Not recommended) clone the repo and install locally

1. Clone  this repo from the pinneylab Github
2. Change directory to unzipped package path
    - `$ cd /repo-download-dir`
3. pip install the package in place and make editable
    - `$ pip install -e .`

## Processing and Analyzing Data

Our processing and analysis is done in Jupyter notebooks. To get started, download the [latest release](https://github.com/pinneylab/htbam_notebooks/releases/latest) of our notebooks repo.

Then in your conda environment, start the Jupyter server with the command `jupyter notebook`.


## To-Do:
- [] Flexible initialization of `LocalHtbamDBAPI`. One should be able to read in any combination of button quant, standard curve, kinetic, binding, and/or stability data. 
    - If it makes sense, could repurpose the `add_run()` method for loading datasets individually. If not, could make a new `load_data()` method.
- [] Implement the `process_dataframe_binding()` function within [csv_processing.py](./src/htbam_analysis/db_api/csv_processing.py). 
- [] Represent binding data as `Data4D` with a dummy time dimension, consistent with how we represent standard curve data. The dependent variable dimension will be the fluorescence ratio of post-wash prey to post-wash bait.
- [] Implement a function in [fit.py](./src/htbam_analysis/analysis/fit.py) for fitting binding isotherms to data. Again, keep the API and logic consistent with what is already implemented. 
    - There should be an optional argument for specifying a list of identifiers for tight-binders. If this argument is provided, the code will fix the value of rmax, estimated from the rmax fit to the tight-binders, to fit the rest of the data.
    - Similar to fitting standards and initial rates, one should be able to specify custom fit windows on a per-concentration basis.
- [] Implement a function to visualize binding isotherms in [plot.py](./src/htbam_analysis/analysis/plot.py)
- [] Implement plotting methods analgous to `export_MM_sample_data`, `export_MM_chamber_data`, and `export_end_to_end_summary_by_sample` in `HTBAMExperiment`.