# htbam_analysis
Duncan Muir & Nicholas Freitas

Credits: Daniel Mohktari and Scott Longwell
___
![BuildStatus](https://github.com/pinneylab/htbam_analysis/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/pinneylab/htbam_analysis/graph/badge.svg?token=DWPK36TQQS)](https://codecov.io/gh/pinneylab/htbam_analysis)

## Overview
This is a *WORK IN PROGRESS* package for processing and analysing various assays from the HTBAM/HTMEK platform.

## Installation

### Conda Environment Setup

(Recommended) Create a fresh conda environment with python=3.9 using:

```conda create -n htbam_analysis python=3.9```

and activate using:

```conda activate htbam_analysis```

#### Intall Stable Release via Wheel File
Download the latest wheel file from the [Release Page](https://github.com/pinneylab/htbam_analysis/releases)

Then, install the package to your conda environment using:

```pip install /path/to/downloaded/wheel.file```

#### For latest code (Not recommended) clone the repo and install locally

1. Clone  this repo from the pinneylab Github
2. Change directory to unzipped package path
    - `$ cd /repo-download-dir`
3. pip install the package in place and make editable
    - `$ pip install -e .`

