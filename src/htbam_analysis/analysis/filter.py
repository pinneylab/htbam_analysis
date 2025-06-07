import time
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy
from typing import List, Dict
from htbam_db_api.data import Data4D, Data3D, Data2D, Meta

### Per-well filters
# Standard curve R2 cutoff
# Expression threshold

# Per-concentration filters
# Initial rate R2 threshhold
# Initial rate positive slope threshhold

def filter_initial_rates_r2_cutoff(initial_rate_data, r2_cutoff):
    '''
    Filter initial rates data.
    Inputs:
        initial_rate_data: Data3D (dep var shape: (n_conc, n_chambers, n_dep_vars)) containing the initial rate data
        r2_cutoff: the R2 cutoff for filtering initial rates
    Returns:
        output_data: a Data3D with a boolean mask indicating which initial rates pass the R2 cutoff
    '''

    assert type(initial_rate_data) == Data3D, "initial_rate_data must be of type Data3D."
    
    # first we grab the R2 values:
    r_squared_idx = initial_rate_data.dep_var_type.index('r_squared')
    initial_rates_r_squared = initial_rate_data.dep_var[..., r_squared_idx]
    mask = initial_rates_r_squared >= r2_cutoff # (n_conc, n_chambers)
    # Turn to (n_conc, n_chambers, 1)
    mask = np.expand_dims(mask, axis=-1)  # (n_conc, n_chambers, 1)
    
    # Turn into db object:
    metadata = Meta(
        mask_type = 'R2_cutoff',
        mask_cutoff = r2_cutoff,
    )
    output_data = Data3D(
        indep_vars=initial_rate_data.indep_vars,
        dep_var=mask,
        dep_var_type=["mask"],
        meta=metadata
    )
    
    return output_data

def filter_r2_cutoff(data, r2_cutoff):
    '''
    Filter initial rates data.
    Inputs:
        data: DataND (dep var shape: (..., n_dep_vars)) containing the fit params
        r2_cutoff: the R2 cutoff for filtering initial rates
    Returns:
        output_data: a Data3D with a boolean mask indicating which initial rates pass the R2 cutoff
    '''

    assert type(data) in [Data4D, Data3D, Data2D], "initial_rate_data must be of type Data4D, Data3D, or Data2D."
    
    dtype = type(data)

    # first we grab the R2 values:
    r_squared_idx = data.dep_var_type.index('r_squared')
    r_squared = data.dep_var[..., r_squared_idx]
    mask = r_squared >= r2_cutoff # (n_conc, n_chambers)
    # Turn to (n_conc, n_chambers, 1)
    mask = np.expand_dims(mask, axis=-1)  # (n_conc, n_chambers, 1)
    
    # Turn into db object:
    metadata = Meta(
        mask_type = 'R2_cutoff',
        mask_cutoff = r2_cutoff,
    )
    output_data = dtype(
        indep_vars=data.indep_vars,
        dep_var=mask,
        dep_var_type=["mask"],
        meta=metadata
    )
    
    return output_data

def filter_initial_rates_positive_cutoff(initial_rate_data: Data3D):
    """
    Filter initial rates for positive slopes using Data3D.
    """
    assert isinstance(initial_rate_data, Data3D), "initial_rate_data must be Data3D."
    # extract slope index and values
    slope_idx = initial_rate_data.dep_var_type.index("slope")
    slopes = initial_rate_data.dep_var[..., slope_idx]           # (n_conc, n_chambers)
    mask = slopes > 0
    mask = np.expand_dims(mask, axis=-1)                         # (n_conc, n_chambers, 1)
    metadata = Meta(mask_type="positive_slope")
    return Data3D(
        indep_vars=initial_rate_data.indep_vars,
        dep_var=mask,
        dep_var_type=["mask"],
        meta=metadata,
    )

def filter_expression_cutoff(expression_data: Data2D, initial_rate_data: Data3D, expression_cutoff: float):
    """
    Filter initial rates based on expression threshold using Data2D and Data3D.
    """
    assert isinstance(expression_data, Data2D), "expression_data must be Data2D."
    assert isinstance(initial_rate_data, Data3D), "initial_rate_data must be Data3D."
    conc_idx = expression_data.dep_var_type.index("concentration")
    expr_vals = expression_data.dep_var[..., conc_idx]           # (n_chambers,)
    mask1 = expr_vals >= expression_cutoff
    n_conc, n_chamb = initial_rate_data.dep_var.shape[:2]
    mask = np.expand_dims(np.tile(mask1, (n_conc, 1)), axis=-1)  # (n_conc, n_chambers, 1)
    metadata = Meta(mask_type="expression_cutoff", mask_cutoff=expression_cutoff)
    return Data3D(
        indep_vars=initial_rate_data.indep_vars,
        dep_var=mask,
        dep_var_type=["mask"],
        meta=metadata,
    )

def filter_standard_curve_r2_cutoff(standard_curve_fit_data: Data2D, initial_rate_data: Data3D, r2_cutoff: float):
    """
    Filter standard curve data based on R2 cutoff using Data2D and Data3D.
    """
    assert isinstance(standard_curve_fit_data, Data2D), "standard_curve_fit_data must be Data2D."
    assert isinstance(initial_rate_data, Data3D), "initial_rate_data must be Data3D."
    r2_idx = standard_curve_fit_data.dep_var_type.index("r_squared")
    r2_vals = standard_curve_fit_data.dep_var[..., r2_idx]      # (n_chambers,)
    mask1 = r2_vals >= r2_cutoff
    n_conc, n_chamb = initial_rate_data.dep_var.shape[:2]
    mask = np.expand_dims(np.tile(mask1, (n_conc, 1)), axis=-1)  # (n_conc, n_chambers, 1)
    metadata = Meta(mask_type="standard_curve_R2_cutoff", mask_cutoff=r2_cutoff)
    return Data3D(
        indep_vars=initial_rate_data.indep_vars,
        dep_var=mask,
        dep_var_type=["mask"],
        meta=metadata,
    )