import time
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy
from typing import List, Dict
from scipy.optimize import curve_fit
import inspect

### Models:
# These get passed to scipy's curve_fit function. The first argument is is always x, followed by the parameters to fit.

# Linear: We use sklearn's LinearRegression for speed.

# Michaelis-Menten:
def mm_model(x, v_max, K_m):
    """
    Michaelis-Menten model for enzyme kinetics.
    
    Parameters
    ----------
    x : array-like
        Substrate concentration.
    v_max : float
        Maximum reaction rate.
    K_m : float
        Michaelis constant.
    
    Returns
    -------
    array-like
        Reaction rate at each substrate concentration.
    """
    return v_max * x / (K_m + x)

# Inhibition model:
def inhibition_model(x, r_max, r_min, ic50):
    """
    Inhibition model for dose-response curves.
    
    Parameters
    ----------
    x : array-like
        Inhibitor concentration.
    r_max : float
        Maximum response.
    r_min : float
        Minimum response.
    ic50 : float
        Concentration at which the response is half-maximal.
    
    Returns
    -------
    array-like
        Response at each inhibitor concentration.
    """
    return r_min + (r_max - r_min) / (1 + (x / ic50))


### Fitting functions for DB objects:
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
    assert data_type == 'RFU_data', f"Data type {data_type} not supported. Requires 'RFU_data' format."
    
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
    r_squared = np.full_like(slope, np.nan)

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
        r_squared[i, good_chamb] = model.score(Xi_c, y_good_c)  # R² for each chamber

    indep_out = {
        "concentration": indep["concentration"],
        "chamber_IDs"  : indep["chamber_IDs"],
        "sample_IDs"   : indep["sample_IDs"],
    }

    elapsed = time.time() - start
    print(f'Fit slopes for {n_chamb} wells at {n_conc} concentrations.')
    print('Elapsed', np.round(elapsed, 3), 'seconds.')

    # assemble result dict (shallow copy of independents for context)
    result = {
        'data_type': 'linear_fit_data',
        "indep_vars": deepcopy(indep_out),
        "dep_vars"  : {
            "slope"     : slope,    # (n_conc, n_chamb)
            "intercept" : intercept,# (n_conc, n_chamb)
            "r_squared" : r_squared,# (n_conc, n_chamb)
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
        Dictionary in "RFU_data" format (see htbam_db_api.py).
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
    assert data_type == 'RFU_data', f"Data type {data_type} not supported. Requires 'RFU_data' format."

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
    r_squared = np.full_like(slope, np.nan)

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
        r_squared[j] = model.score(X_all[good], y[good])  # R² for each chamber

    elapsed = time.time() - start
    print(f'Fit slopes for {n_chamb} wells.')
    print('Elapsed', np.round(elapsed, 3), 'seconds.')

    # axes for context
    indep_out = {
        "time"       : indep["time"],      # (n_conc , n_time)
        "chamber_IDs": indep["chamber_IDs"],
        "sample_IDs" : indep["sample_IDs"]
    }

    # assemble result dict (shallow copy of independents for context)
    result = {
        'data_type': 'linear_fit_data', # Is this the correct format for linear_fit_data? Dimensions are different I think.
        "indep_vars": deepcopy(indep_out),
        "dep_vars"  : {
            "slope"     : slope,     # (n_chamb,)
            "intercept" : intercept, # (n_chamb,)
            "r_squared" : r_squared, # (n_chamb,)
        },
        "meta": {
            "fit" : f"{y_label}_vs_{x_label}",
            "model": "LinearRegression",
        }
    }
    return result

def fit_initial_rates_vs_concentration_with_function(data: dict,
                                  model_func: callable,
                                  *,
                                  min_pts: int = 2,
                                  bounds: tuple = (-np.inf, np.inf),
                                  maxfev: int = 10000,
                                  p0: List[float] = None):
    """
    Fit a user-defined nonlinear function to initial rates vs substrate concentrations.
    Parameters
    ----------
    data : dict
        Dictionary in "RFU_data" format (see htbam_db_api.py).
    model_func : callable
        Callable of the form model_func(x, *params) -> y. The first argument is x,
        followed by N parameters to fit. For example:

            def mm_model(x, v_max, K_m):
                return v_max * x / (K_m + x)
    min_pts : int, optional
        Minimum number of (x, y) pairs required for a fit (default 2).
    bounds : 2-tuple of array-like, optional
        Lower and upper bounds on parameters, passed to `curve_fit`.
        Defaults to no bounds (i.e. `(-inf, +inf)`).
    maxfev : int, optional
        Maximum number of function evaluations in `curve_fit`. Default is 10000.
    p0 : sequence or None, optional
        Initial guess for the fit parameters. If None, defaults to `[1.0]*N_params`.
        Must have length = number of parameters in model_func (i.e. signature minus 1).
    Returns
    -------
    result : dict
        New data dictionary with
        result["dep_vars"]["fit_params"]  – list of fitted parameter arrays (length n_chambers, N_params, )
        result["meta"]["model"]       – string describing the model used
        result["meta"]["parameters"] – list of parameter names
        plus a shallow copy of  data["indep_vars"]  for context.
    """

    start = time.time()
    data_type = data['data_type']
    assert data_type in ("linear_fit_data", "linear_fit_data_mask"), f"Data type {data_type} not supported. Requires 'linear_fit_data' or 'linear_fit_data_mask' format."

    indep = data["indep_vars"]
    dep   = data["dep_vars"]

    x_label = "concentration"
    y_label = "slope"

    if y_label not in dep:
        raise KeyError(f"'{y_label}' not in data['dep_vars']")
    if x_label not in indep:
        raise KeyError(f"'{x_label}' not in data['indep_vars']")

    Y = dep[y_label]                                # (n_conc, n_chamb)

    n_conc, n_chamb = Y.shape
    model = LinearRegression()
    
    X_vec = indep[x_label]        # (n_conc,)
    #X_all = X_vec.reshape(-1, 1)          # (n_conc, 1)

    curve_fit_results_dict = fit_nonlinear_models(
        X_vec, 
        Y, 
        model_func, 
        p0=None, 
        bounds=(-np.inf, np.inf), 
        maxfev=10000
    )

    elapsed = time.time() - start

    num_successful_fits = np.sum(~np.isnan(curve_fit_results_dict["params"]).any(axis=1))
    print(f'Successfully fit nonlinear model for {num_successful_fits} wells.')
    print('Elapsed', np.round(elapsed, 3), 'seconds.')

    # # axes for context
    # indep_out = {
    #     "chamber_IDs": indep["chamber_IDs"],
    #     "sample_IDs" : indep["sample_IDs"]
    # }

    # assemble result dict (shallow copy of independents for context)
    result = {
        'data_type': 'linear_fit_data', # Is this the correct format for linear_fit_data? Dimensions are different I think.
        "indep_vars": deepcopy(indep),
        "dep_vars"  : {
            "fit_params" : curve_fit_results_dict["params"],  # (n_chamb, N_params)
            "covariances": curve_fit_results_dict["covariances"],  # (n_chamb, N_params, N_params)
            "y_pred"     : curve_fit_results_dict["y_pred"],  # (n_conc, n_chamb)
            "r_squared"  : curve_fit_results_dict["r_squared"],  # (n_chamb,)
        },
        "meta": {
            "fit" : f"{y_label}_vs_{x_label}",
            "model": model_func.__name__,
            "parameters": list(inspect.signature(model_func).parameters.keys())[1:],  # skip the first 'x' argument
        }
    }
    return result
                            
def transform_data(data: dict, apply_to: str, store_as: str, function: callable, flatten: bool = False, data_type: str = None) -> dict:
    """
    Transform a data dictionary by applying a function to the specified dependent variable.

    TODO: Currently, this uses a single lambda to apply to all values. Would be nice to pass in a matrix or something too, so we can divide all chamber values by standard curve slope, for example.
    Parameters
    ----------
    data : dict
        Dictionary in "RFU_data" format (see htbam_db_api.py).
    apply_to : str
        Key in data["dep_vars"] to apply the function to.
    store_as : str
        Key in data["dep_vars"] to store the transformed variable.
    function : callable
        Function to apply to the dependent variable.
    flatten : bool, optional
        If True, flatten the output array (default False).

    Returns
    -------
    result : dict
        New data dictionary with transformed dependent variable.
    """
    if apply_to not in data["dep_vars"]:
        raise KeyError(f"'{apply_to}' not in data['dep_vars']")

    transformed_data = deepcopy(data)
    transformed_data["dep_vars"][store_as] = function(transformed_data["dep_vars"][apply_to])
    transformed_data.pop(apply_to, None)  # Remove the original variable if needed

    if flatten:
        transformed_data["dep_vars"][apply_to] = transformed_data["dep_vars"][apply_to].flatten()

    if data_type is not None:
        transformed_data["data_type"] = data_type

    return transformed_data

### Generalized fitting function:
def fit_nonlinear_models(
    X_vec, 
    Y, 
    model_func, 
    p0=None, 
    bounds=(-np.inf, np.inf), 
    maxfev=10000
):
    """
    Fit a user-defined nonlinear function `model_func` to each column of Y vs X_vec.

    Parameters
    ----------
    X_vec : array-like, shape (n_points,)
        Independent variable (e.g., substrate concentrations).
    Y : array-like, shape (n_points, n_series)
        Dependent data (e.g., initial rates), with each column as a separate series (chamber).
    model_func : callable
        Callable of the form model_func(x, *params) -> y. The first argument is x, 
        followed by N parameters to fit. For example:
        
            def mm_model(x, v_max, K_m):
                return v_max * x / (K_m + x)
        
    p0 : sequence or None, optional
        Initial guess for the fit parameters. If None, defaults to `[1.0]*N_params`.  
        Must have length = number of parameters in model_func (i.e. signature minus 1).
    bounds : 2-tuple of array-like, optional
        Lower and upper bounds on parameters, passed to `curve_fit`.  
        Defaults to no bounds (i.e. `(-inf, +inf)`).
    maxfev : int, optional
        Maximum number of function evaluations in `curve_fit`. Default is 10000.

    Returns
    -------
    result : dict with keys
      • 'params'      : list of length n_series; each entry is the fitted parameter array (length N_params) or None if fit failed  
      • 'covariances': list of length n_series; each entry is the covariance matrix (N_params×N_params) or None  
      • 'y_pred'      : array, shape (n_points, n_series); predicted y values (NaN where fit wasn’t done)  
      • 'r_squared'   : array, shape (n_series,); R² of each fit (NaN if not computed)
    """
    # Convert to numpy arrays
    X = np.asarray(X_vec, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 1:
        raise ValueError("X_vec must be 1D")
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array with shape (n_points, n_series)")
    n_points, n_series = Y.shape

    # Determine how many parameters model_func takes (excluding x)
    sig = inspect.signature(model_func)
    num_params = len(sig.parameters) - 1
    if num_params < 1:
        raise ValueError("model_func must accept at least one parameter besides x")

    # Prepare output containers
    params_list = [None] * n_series
    pcov_list = [None] * n_series
    Y_pred = np.full_like(Y, np.nan)
    r_squared = np.full((n_series,), np.nan, dtype=float)

    # Default initial guess: all ones
    if p0 is None:
        p0 = [1.0] * num_params
    else:
        if len(p0) != num_params:
            raise ValueError(f"p0 must have length {num_params}")

    # R² helper
    def compute_r2(y_obs, y_fit):
        ss_res = np.nansum((y_obs - y_fit) ** 2)
        ss_tot = np.nansum((y_obs - np.nanmean(y_obs)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Loop over each series (e.g., each chamber)
    for j in range(n_series):
        y = Y[:, j]
        mask = (~np.isnan(X)) & (~np.isnan(y))
        # Need more data points than parameters to fit
        if mask.sum() <= num_params:
            continue

        x_fit = X[mask]
        y_fit = y[mask]

        try:
            popt, pcov = curve_fit(
                model_func, 
                x_fit, 
                y_fit, 
                p0=p0, 
                bounds=bounds, 
                maxfev=maxfev
            )
            params_list[j] = popt
            pcov_list[j] = pcov

            # Compute predictions over the full X domain
            y_full_pred = model_func(X, *popt)
            Y_pred[:, j] = y_full_pred

            # Compute R² over masked points only
            r_squared[j] = compute_r2(y_fit, y_full_pred[mask])
        except (RuntimeError, ValueError):
            # If fit fails, leave entries as None/NaN
            continue

    # Turn the lists of parameters and covariances into numpy arrays
    params_list = np.array([np.array(p) if p is not None else np.full(num_params, np.nan) for p in params_list])
    pcov_list = np.array([np.array(cov) if cov is not None else np.full((num_params, num_params), np.nan) for cov in pcov_list])

    # Same for y_pred, and r_squared
    Y_pred = np.array(Y_pred)
    r_squared = np.array(r_squared)

    return {
        'params': params_list,
        'covariances': pcov_list,
        'y_pred': Y_pred,
        'r_squared': r_squared
    }