import time
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy
from typing import List, Dict

def fit_luminance_vs_time(data: dict, *, min_pts: int = 2, start_timepoint: int = 0, end_timepoint: int = -1):
    """
    Fit y = β0 + β1·x with scikit-learn, returning slope & intercept
    in a data-dict that mirrors the original structure.

    Parameters
    ----------
    data : dict
        Dictionary in "kinetics" format (see htbam_db_api.py).
    min_pts : int, optional
        Minimum number of (x, y) pairs required for a fit
        (default 2).
    start_timepoint : int, optional
        First timepoint to include in the fit (default 0).
    end_timepoint : int, optional
        Last timepoint to include in the fit (default -1, i.e. all points).

    Returns
    -------
    result : dict
        New data dictionary with
        result["dep_vars"]["slope"]      – slope array
        result["dep_vars"]["intercept"]  – intercept array
        plus a shallow copy of  data["indep_vars"]  for context.
    """
    start = time.time()
    
    data_type = data['data_type']
    assert data_type == 'kinetics', f"Data type {data_type} not supported. Requires 'kinetics' format."
    
    indep = data["indep_vars"]
    dep   = data["dep_vars"]

    x = "time"
    y = "luminance"

    if y not in dep:
        raise KeyError(f"'{y}' not in data['dep_vars']")
    if x not in indep:
        raise KeyError(f"'{x}' not in data['indep_vars']")

    Y = dep[y]                                # (n_conc , n_time , n_chamb)

    n_conc, n_time, n_chamb = Y.shape
    model = LinearRegression()

    T = indep["time"]                      # (n_conc , n_time)
    slope     = np.full((n_conc, n_chamb), np.nan, dtype=float)
    intercept = np.full_like(slope, np.nan)

    # What subset of points are we fitting on?
    RFU_for_fitting = Y[:, start_timepoint:end_timepoint, :]  # skip the first couple points, because we get weird values
    time_array_for_fitting = T[:, start_timepoint:end_timepoint]  # skip the first couple points, because we get weird values

    for i in range(n_conc):
        Xi = time_array_for_fitting[i].reshape(-1, 1)          # (Ti, 1)
        yi = RFU_for_fitting[i]                          # (Ti, K)

        # 1. keep chambers that have *no* NaNs over time
        good_chamb = ~np.isnan(yi).any(axis=0)     # (K,) boolean mask
        if not good_chamb.any():                   # nothing left to fit
            continue

        y_good = yi[:, good_chamb]                 # (Ti, K_good)

        # 2. (optional) drop time points that are still NaN in *any* kept chamber
        #good_rows = ~np.isnan(y_good).any(axis=1)  # (Ti,) mask
        good_rows = np.ones(y_good.shape[0], dtype=bool)
        Xi_c, y_good_c = Xi[good_rows], y_good[good_rows]

        if len(Xi_c) < 2:                          # need ≥2 points for a line
            continue

        # 3. multi-output linear regression
        model.fit(Xi_c, y_good_c)
        intercept[i, good_chamb] = model.intercept_      # (K_good,)
        slope[i,     good_chamb] = model.coef_[:, 0]     # (K_good,)

    indep_out = {
        "concentration": indep["concentration"],
        "chamber_IDs"  : indep["chamber_IDs"]
    }

    elapsed = time.time() - start
    print(f'Fit slopes for {n_chamb} wells at {n_conc} concentrations.')
    print('Elapsed', np.round(elapsed, 3), 'seconds.')

    # assemble result dict (shallow copy of independents for context)
    result = {
        "indep_vars": deepcopy(indep_out),
        "dep_vars"  : {
            "slope"     : slope,    # (n_conc, n_chamb)
            "intercept" : intercept,# (n_conc, n_chamb)
        },
        "meta": {
            "fit" : f"{y}_vs_{x}",
            "model": "LinearRegression",
        }
    }
    return result

def fit_luminance_vs_concentration(data: dict, *, min_pts: int = 2, timepoint: int = -1):
    """
    Fit y = β0 + β1·x with scikit-learn, returning slope & intercept
    in a data-dict that mirrors the original structure.

    Parameters
    ----------
    data : dict
        Dictionary in "kinetics" format (see htbam_db_api.py).
    x : {"concentration", "time"}
        Which independent variable to use.
    y : key in  data["dep_vars"]
        Name of the dependent variable to fit.
    min_pts : int, optional
        Minimum number of (x, y) pairs required for a fit
        (default 2).
    timepoint : int, optional
        Timepoint to use for the fit (default -1, i.e. last timepoint).

    Returns
    -------
    result : dict
        New data dictionary with
        result["dep_vars"]["slope"]      – slope array
        result["dep_vars"]["intercept"]  – intercept array
        plus a shallow copy of  data["indep_vars"]  for context.
    """
    start = time.time()
    data_type = data['data_type']
    assert data_type == 'kinetics', f"Data type {data_type} not supported. Requires 'kinetics' format."

    indep = data["indep_vars"]
    dep   = data["dep_vars"]

    x_label = "concentration"
    y_label = "luminance"

    if y_label not in dep:
        raise KeyError(f"'{y_label}' not in data['dep_vars']")
    if x_label not in indep:
        raise KeyError(f"'{x_label}' not in data['indep_vars']")

    Y = dep[y_label]                                # (n_conc , n_time , n_chamb)
    Yi = Y[:, timepoint, :]                          # shape (n_conc , n_chamb)

    n_conc, n_time, n_chamb = Y.shape
    model = LinearRegression()
    
    X_vec = indep[x_label]        # (n_conc,)
    X_all = X_vec.reshape(-1, 1)          # (n_conc, 1)
    # slope/intercept per (time , chamber)
    slope     = np.full((n_chamb), np.nan, dtype=float)
    intercept = np.full_like(slope, np.nan)

    # In order to properly mask and remove NaN concentrations, we're iterating over chambers here.
    # This skullduggery makes it slower than the more difficult luminance vs time fit, but it still runs in ~1 second.
    for j in range(n_chamb):
        y = Yi[:, j]                      # (n_conc,)
        good = ~np.isnan(y) & ~np.isnan(X_vec)           # per-chamber mask

        # need at least two distinct concentration points
        if good.sum() < 2 or np.unique(X_vec[good]).size < 2:
            continue
        
        model.fit(X_all[good], y[good])
        intercept[j] = model.intercept_
        slope[j]     = model.coef_[0]

    elapsed = time.time() - start
    print(f'Fit slopes for {n_chamb} wells.')
    print('Elapsed', np.round(elapsed, 3), 'seconds.')

    # axes for context
    indep_out = {
        "time"       : indep["time"],      # (n_conc , n_time)
        "chamber_IDs": indep["chamber_IDs"]
    }

    # assemble result dict (shallow copy of independents for context)
    result = {
        "indep_vars": deepcopy(indep_out),
        "dep_vars"  : {
            "slope"     : slope,     # (n_chamb,)
            "intercept" : intercept, # (n_chamb,)
        },
        "meta": {
            "fit" : f"{y_label}_vs_{x_label}",
            "model": "LinearRegression",
        }
    }
    return result
