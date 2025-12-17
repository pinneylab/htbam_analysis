import time
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy
from typing import List, Dict, Tuple
from scipy.optimize import curve_fit
import inspect

from htbam_db_api.data import Data4D, Data3D, Data2D, Meta
from htbam_db_api.units.units import units as ureg

from pint.errors import DimensionalityError

### Decorator to add preferred units to functions:
def set_units(**kwargs):
    def decorator(func):
        func.get_param_units = kwargs
        return func
    return decorator

### Models:
# These get passed to scipy's curve_fit function. The first argument is is always x, followed by the parameters to fit.

# Linear: We use sklearn's LinearRegression for speed.

# Michaelis-Menten:
@set_units( x = ureg.uM,                # x is the independent variable (input)
            v_max = ureg.uM / ureg.s,   # v_max and K_m are the parameters to fit
            K_m = ureg.uM,              
            y = ureg.uM / ureg.s)       # y is the dependent variable (output)
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
@set_units(x = ureg.uM,                 # x is the independent variable (input)
            r_max = ureg.dimensionless, # r_max, r_min, and ic50 are the parameters to fit
            r_min = ureg.dimensionless, #
            ic50 = ureg.uM,             #
            y = ureg.dimensionless)     # y is the dependent variable (output)
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
def fit_concentration_vs_time(data: Data4D, *, min_pts: int = 2, start_timepoint: int = 0, end_timepoint: int = -1,
                              fit_windows_per_concentration: dict = None) -> Data3D:
    """
    Fit y = β0 + β1·x with scikit-learn, returning slope & intercept
    in a data-dict that mirrors the original structure.

    Parameters
    ----------
    data : Data4D
        Data object in Data4D format (see htbam_db_api.data).
    min_pts : int, optional
        Minimum number of (x, y) pairs required for a fit
        (default 2).
    start_timepoint : int, optional
        First timepoint to include in the fit (default 0).
    end_timepoint : int, optional
        Last timepoint to include in the fit (default -1, i.e. all points).
    fit_windows_per_concentration : dict, optional
        Dictionary mapping concentration values to (start, end) timepoint tuples.
        If provided, these override start_timepoint and end_timepoint for each concentration.

    Returns
    -------
    result : Data3D
        Data object with dep_var of shape (n_conc, n_chamb, 3) containing:
    - slope
    - intercept
    - r_squared
    """
    start = time.time()
    
    assert type(data) == Data4D, f"Data type {type(data)} not supported. Requires 'Data4D' format."
    
    indep = data.indep_vars  # (n_conc, n_time)
    dep   = data.dep_var

    x_label = "time"
    y_label = "concentration"

    if y_label not in data.dep_var_type:
        raise KeyError(f"'{y_label}' not in data.dep_var_type.")
    # if x not in data.indep_vars:
    #     raise KeyError(f"'{x}' not in data.indep_vars.")

    y_idx = data.dep_var_type.index(y_label)

    # Unit handling
    x_unit = indep.time.units
    y_unit = data.dep_var_units[y_idx]

    slope_unit = y_unit / x_unit
    intercept_unit = y_unit
    r_squared_unit = ureg.dimensionless

    Y = dep[..., y_idx]                              # (n_conc , n_time , n_chamb)
    if hasattr(Y, 'magnitude'): Y = Y.magnitude

    n_conc, n_time, n_chamb = Y.shape
    model = LinearRegression()

    T = indep.time                                   # (n_conc , n_time)
    if hasattr(T, 'magnitude'): T = T.magnitude

    slope     = np.full((n_conc, n_chamb), np.nan, dtype=float)
    intercept = np.full_like(slope, np.nan)
    r_squared = np.full_like(slope, np.nan)

    # If user provided per-concentration fit windows, validate them and prepare a mapping
    concs = indep.concentration
    if hasattr(concs, 'magnitude'):
        concs = concs.magnitude
    concs = np.asarray(concs)
    per_conc_windows = None
    if fit_windows_per_concentration is not None:
        if not isinstance(fit_windows_per_concentration, dict):
            raise TypeError("fit_windows_per_concentration must be a dict mapping concentration -> (start, end)")

        # Ensure every concentration in data has an entry in the provided dict
        missing = [float(c) for c in concs if float(c) not in map(float, fit_windows_per_concentration.keys())]
        if len(missing) > 0:
            raise KeyError(f"fit_windows_per_concentration missing windows for concentrations: {missing}")

        # Normalize and validate windows
        per_conc_windows = {}
        for k, v in fit_windows_per_concentration.items():
            try:
                kf = float(k)
            except Exception:
                raise KeyError(f"Invalid concentration key: {k}")
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise ValueError(f"Window for concentration {k} must be a (start, end) tuple")
            s, e = int(v[0]), int(v[1])
            per_conc_windows[float(kf)] = (s, e)

    # What subset of points are we fitting on? If per-concentration windows provided, we'll slice inside the loop.
    if per_conc_windows is None:
        RFU_for_fitting = Y[:, start_timepoint:end_timepoint, :]  # skip the first couple points, because we get weird values
        time_array_for_fitting = T[:, start_timepoint:end_timepoint]  # skip the first couple points, because we get weird values

    for i in range(n_conc):
        # If per-concentration windows were provided, use them for this concentration
        if per_conc_windows is not None:
            conc_val = float(concs[i])
            start_idx, end_idx = per_conc_windows[conc_val]
            Xi = T[i, start_idx:end_idx].reshape(-1, 1)          # (Ti, 1)
            yi = Y[i, start_idx:end_idx, :]                      # (Ti, K)
        else:
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

    output_data = Data3D(
        indep_vars=deepcopy(data.indep_vars),
        dep_var=np.stack((slope, intercept, r_squared), axis=-1),  # (n_conc, n_chamb, 3)
        dep_var_type=["slope", "intercept", "r_squared"],
        dep_var_units=[slope_unit, intercept_unit, r_squared_unit],
        meta=data.meta
    )
    n_conc = indep.time.shape[0]  # number of concentrations
    n_chamb = indep.chamber_IDs.shape[0]  # number of chambers

    elapsed = time.time() - start
    print(f'Fit slopes for {n_chamb} wells at {n_conc} concentrations.')
    print('Elapsed', np.round(elapsed, 3), 'seconds.')

    return output_data

def fit_luminance_vs_concentration(data: Data4D, *, min_pts: int = 2, timepoint: int = -1) -> Data2D:
    """
    Fit y = β0 + β1·x with scikit-learn, returning slope & intercept
    in a data-dict that mirrors the original structure.

    Parameters
    ----------
    data : Data4D
        Dictionary in "Data4D" format (see htbam_db_api.data).
    min_pts : int, optional
        Minimum number of (x, y) pairs required for a fit
        (default 2).
    timepoint : int, optional
        Timepoint to use for the fit (default -1, i.e. last timepoint).

    Returns
    -------
    Data2D: data object with dep_var of shape (n_chamb, 3) containing:
    - slope
    - intercept
    - r_squared
    """
    start = time.time()

    assert type(data) is Data4D, "Data must be in RFU_data format Data4D."

    indep = data.indep_vars.concentration  # (n_conc,)
    dep   = data.dep_var

    x_label = "concentration"
    y_label = "luminance"

    # Check if the labels are in the data
    if y_label not in data.dep_var_type:
        raise KeyError(f"'{y_label}' not in data.dep_var_type.")

    y_idx = data.dep_var_type.index(y_label)

    # Unit handling
    # We strip units for fitting to avoid warnings, then re-apply them to the results
    x_unit = indep.units
    y_unit = data.dep_var_units[y_idx]
    
    # Calculate output units
    slope_unit = y_unit / x_unit
    intercept_unit = y_unit
    r2_unit = ureg.dimensionless

    Y = dep[..., y_idx]                              # (n_conc , n_time , n_chamb)
    if hasattr(Y, 'magnitude'): Y = Y.magnitude      # Strip units
    
    Yi = Y[:, timepoint, :]                          # shape (n_conc , n_chamb)

    n_conc, n_time, n_chamb = Y.shape
    model = LinearRegression()
    
    X_vec = indep        # (n_conc,)
    if hasattr(X_vec, 'magnitude'): X_vec = X_vec.magnitude # Strip units
    
    X_all = X_vec.reshape(-1, 1)     # (n_conc, 1)
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
        r_squared[j] = model.score(X_all[good], y[good])  # R² for each chamber ## temp

    elapsed = time.time() - start
    print(f'Fit slopes for {n_chamb} wells.')
    print('Elapsed', np.round(elapsed, 3), 'seconds.')

    output_data = Data2D(
        indep_vars=data.indep_vars,
        dep_var=np.stack((slope, intercept, r_squared), axis=-1),  # (n_chamb, 3)
        dep_var_type=["slope", "intercept", "r_squared"],
        dep_var_units=[slope_unit, intercept_unit, r2_unit],
        meta=data.meta
    )

    return output_data

def fit_initial_rates_vs_concentration_with_function(
    data: Data3D,
    model_func: callable,
    *,
    min_pts: int = 2,
    bounds: tuple = (-np.inf, np.inf),
    maxfev: int = 10000,
    p0: List[float] = None,
) -> Tuple[Data2D, Data3D]:
    """
    Fit a user-defined nonlinear function to initial rates vs substrate concentrations.
    Parameters
    ----------
    data : Data3D
        Data object with fit initial rates.
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
    Data2D: data object with dep_var of shape (n_chamb, N_params + 1) containing:
    - fitted parameters (N_params)
    - R² values (r_squared)
    Data3D: data object with dep_var of shape (n_conc, n_chamb, 1) containing:
    - predicted y values (y_pred)
    """
    assert isinstance(data, Data3D), "data must be Data3D."
    start = time.time()
    # extract substrate concentrations and initial rates
    X = data.indep_vars.concentration                 # (n_conc,)
    slope_idx = data.dep_var_type.index("slope")
    Y_unit = data.dep_var_units[slope_idx]
    Y = data.dep_var[..., slope_idx] * Y_unit         # (n_conc, n_chamb)

    # perform fits
    # Here we're passing in X and Y as pint Quantities, which we'll handle in the fit_nonlinear_models function
    results = fit_nonlinear_models(
        X,
        Y,
        model_func,
        p0=p0 or [1.0] * (len(inspect.signature(model_func).parameters) - 1),
        bounds=bounds,
        maxfev=maxfev
    )
    params = results["params"]            # (n_chamb, N_params)
    param_units = results["param_units"]   # (N_params,)
    y_pred = results["y_pred"]            # (n_conc, n_chamb)
    r2 = results["r_squared"]             # (n_chamb,)

    num_successful_fits = np.sum(~np.isnan(params).any(axis=1))
    print(f'Successfully fit nonlinear model for {num_successful_fits} wells.')
    print('Elapsed', np.round(time.time() - start, 3), 'seconds.')

    # 1) build Data2D of params + R²
    param_names = list(inspect.signature(model_func).parameters.keys())[1:]
    dep2d = np.concatenate([params, r2[:,None]], axis=1)  # (n_chamb, Np+1)
    names2d = param_names + ["r_squared"]
    meta2d  = Meta(fit_type=model_func.__name__,)
    params_data = Data2D(
        indep_vars=deepcopy(data.indep_vars),
        dep_var=dep2d,
        dep_var_type=names2d,
        dep_var_units=param_units + [ureg.dimensionless], # adding r_squared unit (dimensionless)
        meta=meta2d
    )

    # 2) build Data3D of predictions
    n_conc, n_chamb = y_pred.shape
    pred3d = y_pred.reshape(n_conc, n_chamb, 1)  # (n_conc, n_chamb, 1)
    meta3d = Meta(fit_type=model_func.__name__)
    ypred_data = Data3D(
        indep_vars=deepcopy(data.indep_vars),
        dep_var=pred3d,
        dep_var_type=["y_pred"],
        dep_var_units=[Y_unit],
        meta=meta3d
    )

    return params_data, ypred_data

### Generalized fitting function:
def fit_nonlinear_models(
    X_vec, 
    Y_vec, 
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
    Y_vec : array-like, shape (n_points, n_series)
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

    # First, does our fitting function have units?
    if hasattr(model_func, 'get_param_units'):
        param_units = model_func.get_param_units
        # Double check that param_units has x (input), y (output) and all the named arguments in the model_func signature
        if 'x' not in param_units or 'y' not in param_units:
            raise ValueError(f"Model function {model_func.__name__} must have 'x' and 'y' units specified in @set_units decorator.")

        sig = inspect.signature(model_func)
        for name in sig.parameters:
            if name not in param_units:
                raise ValueError(f"Model function {model_func.__name__} must have '{name}' units specified in @set_units decorator.")
    else:
        raise ValueError("Model function must have a 'get_param_units' method. (Use @set_units decorator)")

    # Convert X, Y to units specified in @set_units decorator:
    # I'm going to copy them first so we don't modify the original data
    X = X_vec.copy()
    Y = Y_vec.copy()
    try:
        X = X.to(param_units['x'])
        Y = Y.to(param_units['y'])
    except DimensionalityError as e:
        # Adding a clarifying message here.
        raise ValueError(f"{e}\nX and Y must be convertible to units specified in @set_units decorator. X has units {X.units}, Y has units {Y.units}")


    # Now, we convert to numpy arrays for model fitting (no units allowed)
    #X = np.asarray(X, dtype=float)
    if hasattr(X, 'magnitude'):
        X = X.magnitude
        
    #Y = np.asarray(Y, dtype=float)
    if hasattr(Y, 'magnitude'):
        Y = Y.magnitude

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
    # make list of param units, matching order in func signature:
    param_units = [param_units[name] for name in sig.parameters if name not in ['x', 'y']]

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
        'param_units': param_units,
        'covariances': pcov_list,
        'y_pred': Y_pred,
        'r_squared': r_squared
    }


# #
# #fit(my_data: DataND, x_label='concentration', y_label='initial_rate', fit_function: callable)
# def fit_general(
#     data: Data3D,
#     x_label: str,
#     y_label: str,
#     fit_function: callable,
#     *,
#     min_pts: int = 2,
#     bounds: tuple = (-np.inf, np.inf),
#     maxfev: int = 10000,
#     p0: List[float] = None
# ) -> Tuple[Data2D, Data3D]:
#     """
#     Fit a user-defined nonlinear function to y vs x.

#     Parameters
#     ----------
#     data : Data3D
#         Data object with indep_vars and dep_var.
#     x_label : str
#         Label of the independent variable in indep_vars.
#     y_label : str
#         Label of the dependent variable in dep_var_type.
#     fit_function : callable
#         Callable of the form fit_function(x, *params) -> y. The first argument is x,
#         followed by N parameters to fit. For example:

#             def mm_model(x, v_max, K_m):
#                 return v_max * x / (K_m + x)
#     min_pts : int, optional
#         Minimum number of (x, y) pairs required for a fit (default 2).
#     bounds : 2-tuple of array-like, optional
#         Lower and upper bounds on parameters, passed to `curve_fit`.
#         Defaults to no bounds (i.e. `(-inf, +inf)`).
#     maxfev : int, optional
#         Maximum number of function evaluations in `curve_fit`. Default is 10000.
#     p0 : sequence or None, optional
#         Initial guess for the fit parameters. If None, defaults to `[1.0]*N_params`.
#         Must have length = number of parameters in fit_function (i.e. signature minus 1).

#     Returns
#     -------
#     Data2D: data object with dep_var of shape (n_chamb, N_params + 1) containing:
#     - fitted parameters (N_params)
#     - R² values (r_squared)
#     Data3D: data object with dep_var of shape (n_points, n_chamb, 1) containing:
#     - predicted y values (y_pred)
#     """
#     assert isinstance(data, Data3D), "data must be Data3D."
#     start = time.time()
    
#     indep = data.indep_vars
#     dep   = data.dep_var

#     if x_label not in indep.__dict__:
#         raise KeyError(f"'{x_label}' not in indep_vars.")
#     if y_label not in data.dep_var_type:
#         raise KeyError(f"'{y_label}' not in dep_var_type.")

#     x = getattr(indep, x_label)                       # (n_points,)
#     y_idx = data.dep_var_type.index(y_label)
#     y = dep[..., y_idx]                               # (n_points, n_chamb)

#     # perform fits
#     results = fit_nonlinear_models(
#         x,
#         y,
#         fit_function,
#         p0=p0 or [1.0] * (len(inspect.signature(fit_function).parameters) - 1),
#         bounds=bounds,
#         maxfev=maxfev
#     )
#     params = results["params"]            # (n_chamb, N_params)
#     y_pred = results["y_pred"]            # (n_points, n_chamb)
#     r2 = results["r_squared"]             # (n_chamb,)

#     num_successful_fits = np.sum(~np.isnan(params).any(axis=1))
#     print(f'Successfully fit nonlinear model for {num_successful_fits} wells.')
#     print('Elapsed', np.round(time.time() - start, 3), 'seconds.')

#     # 1) build Data2D of params + R²
#     param_names = list(inspect.signature(fit_function).parameters.keys())[1:]
#     dep2d = np.concatenate([params, r2[:,None]], axis=1)  # (n_chamb, Np+1)
#     names2d = param_names + ["r_squared"]
#     meta2d  = Meta(fit_type=fit_function.__name__,)
#     params_data = Data2D(
#         indep_vars=deepcopy(data.indep_vars),
#         dep_var=dep2d,
#         dep_var_type=names2d,
#         meta=meta2d
#     )
#     # 2) build Data3D of predictions
#     n_points, n_chamb = y_pred.shape
#     pred3d = y_pred.reshape(n_points, n_chamb, 1)   
#     meta3d = Meta(fit_type=fit_function.__name__)
#     ypred_data = Data3D(
#         indep_vars=deepcopy(data.indep_vars),
#         dep_var=pred3d,
#         dep_var_type=["y_pred"],
#         meta=meta3d
#     )

#     return params_data, ypred_data