import time
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy
from typing import List, Dict
from htbam_analysis.db_api.data import Data4D, Data3D, Data2D, Meta
from htbam_analysis.db_api.units import units

### Per-well filters
# Standard curve R2 cutoff
# Expression threshold

# Per-concentration filters
# Initial rate R2 threshhold
# Initial rate positive slope threshhold

def make_custom_mask (data, mask, info=""):
    '''
    Apply a custom mask to data.
    Inputs:
        data: DataND (dep var shape: (..., n_dep_vars)) containing the data to be masked
        mask: DataND (dep var shape: (n_conc, n_chambers, 1)) containing the boolean mask
    Returns:
        output_data: a DataND with a boolean mask applied to the dep_var
    '''
    
    assert type(data) in [Data4D, Data3D, Data2D], "data must be of type Data4D, Data3D, or Data2D."
    
    dtype = type(data)
    
    # Check that the shapes are compatible:
    assert data.dep_var.shape[0:2] == mask.shape[0:2], "data and mask must have the same number of concentrations and chambers."
    
    # Turn into db object:
    metadata = Meta(
        mask_type = 'custom',
        #mask_info = info,
    )
    output_data = dtype(
        indep_vars=data.indep_vars,
        dep_var=mask,
        dep_var_type=["mask"],
        dep_var_units=[units.dimensionless], # Boolean masks are dimensionless
        meta=metadata
    )
    
    return output_data

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
        dep_var_units=[units.dimensionless], # Boolean masks are dimensionless
        meta=metadata
    )
    
    return output_data

def filter_by_sample_id(data, sample_ids: List[str]):
    '''
    Filter data by sample IDs.
    Inputs:
        data: DataND (dep var shape: (..., n_dep_vars)) containing the data to be filtered
        sample_ids: list of sample IDs to keep
    Returns:
        output_data: a DataND with a boolean mask indicating which samples are in the sample_ids list
    '''

    assert type(data) in [Data4D, Data3D, Data2D], "data must be of type Data4D, Data3D, or Data2D."
    
    dtype = type(data)
    
    # first we grab the sample IDs:
    all_sample_ids = data.indep_vars.sample_IDs  # (n_chambers,)
    mask = np.isin(all_sample_ids, sample_ids)  # (n_chambers,)
    n_conc = data.dep_var.shape[0]
    # Turn to (n_conc, n_chambers, 1)
    mask = np.expand_dims(np.tile(mask, (n_conc, 1)), axis=-1)  # (n_conc, n_chambers, 1)
    
    # Turn into db object:
    metadata = Meta(
        mask_type = 'sample_id_filter',
        #mask_info = f"Filtered to sample IDs: {', '.join(sample_ids)}",
    )
    output_data = dtype(
        indep_vars=data.indep_vars,
        dep_var=mask,
        dep_var_type=["mask"],
        dep_var_units=[units.dimensionless], # Boolean masks are dimensionless
        meta=metadata
    )
    
    return output_data

def filter_number_replicates(data, min_replicates, var_to_check):
    '''
    Filter data by number of replicates.
    Inputs:
        data: DataND (dep var shape: (..., n_dep_vars)) containing the data to be filtered
        min_replicates: minimum number of replicates required to keep a chamber
        var_to_check: the dependent variable to check for non-NaN values (e.g., 'K_m' or 'V_max')
    Returns:
        output_data: a DataND with a boolean mask indicating which chambers have at least min_replicates non-NaN values
    '''

    assert type(data) in [Data4D, Data3D, Data2D], "data must be of type Data4D, Data3D, or Data2D."
    
    dtype = type(data)
    
    # Determine which chambers have any non-NaN value for the requested variable across concentrations
    var_idx = data.dep_var_type.index(var_to_check)
    # data.dep_var[..., var_idx] -> can be 1D (Data2D) or 2D (n_conc, n_chambers) etc.
    var_values = np.asarray(data.dep_var[..., var_idx])

    # Normalize to per-chamber boolean: True if chamber has at least one non-NaN value
    if var_values.ndim == 0:
        # single scalar -> single chamber
        chamber_has_value = np.array([not np.isnan(var_values)])
    elif var_values.ndim == 1:
        # shape (n_chambers,)
        chamber_has_value = ~np.isnan(var_values)
    else:
        # e.g., shape (n_conc, n_chambers, ...) -> any non-NaN across the first axis (concentrations)
        chamber_has_value = np.any(~np.isnan(var_values), axis=0)
    chamber_has_value = np.ravel(chamber_has_value)  # ensure 1D array (n_chambers,)

    # Group by sample_ID and count how many chambers in each group have a value
    sample_ids = np.asarray(data.indep_vars.sample_IDs)  # (n_chambers,)
    assert sample_ids.shape[0] == chamber_has_value.shape[0], "sample_IDs length must match number of chambers."

    unique_ids, inverse_idx = np.unique(sample_ids, return_inverse=True)
    # Use bincount to count valid chambers per group (robust and fast)
    group_counts = np.bincount(inverse_idx, weights=chamber_has_value.astype(int))
    # Map group counts back to per-chamber counts
    per_chamber_counts = group_counts[inverse_idx]  # (n_chambers,)

    # Build boolean mask per chamber: keep chambers whose group has at least min_replicates valid wells
    chamber_mask = per_chamber_counts >= min_replicates  # (n_chambers,)

    # Return per-chamber mask with shape (n_chambers, 1)
    # (previous code incorrectly tiled the mask producing length n_conc * n_chambers)
    mask = np.expand_dims(chamber_mask, axis=-1)  # (n_chambers, 1)
    
    # Turn into db object:
    metadata = Meta(
        mask_type = 'min_replicates_by_sampleID',
        mask_cutoff = min_replicates,
    )
    output_data = dtype(
        indep_vars=data.indep_vars,
        dep_var=mask,
        dep_var_type=["mask"],
        dep_var_units=[units.dimensionless], # Boolean masks are dimensionless
        meta=metadata
    )
    
    return output_data

def filter_number_concentrations(data, min_concentrations, var_to_check):
    '''
    For each chamber, check how many concentrations have non-nan values in the specified variable.
    Inputs:
        data: DataND (dep var shape: (..., n_dep_vars)) containing the data to be filtered
        min_concentrations: minimum number of concentrations required to keep a chamber
        var_to_check: the dependent variable to check for non-NaN values (e.g., 'slope' or 'intercept')
        Returns:
            output_data: a Data2D with a boolean mask indicating which chambers have at least min_concentrations non-NaN values

    '''
    assert type(data) in [Data4D, Data3D, Data2D], "data must be of type Data4D, Data3D, or Data2D."

    dtype = Data2D

    # Get index for requested dependent variable
    var_idx = data.dep_var_type.index(var_to_check)
    var_values = np.asarray(data.dep_var[..., var_idx])

    # We expect var_values to have shape (n_conc, n_chambers) for Data3D/4D
    # Normalize to 2D (n_conc, n_chambers) or 1D (n_chambers,) for Data2D
    if var_values.ndim == 0:
        # single scalar -> single chamber, single concentration
        counts_per_chamber = np.array([0 if np.isnan(var_values) else 1])
    elif var_values.ndim == 1:
        # shape (n_chambers,) -> count non-nan per chamber (0 or 1)
        counts_per_chamber = (~np.isnan(var_values)).astype(int)
    else:
        # e.g., shape (n_conc, n_chambers, ...) -> count non-nan across concentration axis
        # First reduce any extra trailing dimensions by considering a value non-nan if any element is non-nan
        if var_values.ndim > 2:
            # collapse trailing axes into a single presence boolean per (conc, chamber)
            presence = ~np.isnan(var_values)
            # any non-nan across trailing axes
            presence = np.any(presence, axis=tuple(range(2, var_values.ndim)))
        else:
            presence = ~np.isnan(var_values)  # (n_conc, n_chambers)

        # Count non-NaN concentrations per chamber
        counts_per_chamber = np.sum(presence.astype(int), axis=0)  # (n_chambers,)

    counts_per_chamber = np.ravel(counts_per_chamber)

    # Build boolean mask per chamber: True if chamber has at least min_concentrations valid concentrations
    chamber_mask = counts_per_chamber >= min_concentrations  # (n_chambers,)

    # Return per-chamber mask with shape (n_chambers, 1)
    mask = np.expand_dims(chamber_mask, axis=-1)

    metadata = Meta(
        mask_type = 'min_concentrations',
        mask_cutoff = min_concentrations,
    )
    output_data = dtype(
        indep_vars=data.indep_vars,
        dep_var=mask,
        dep_var_type=["mask"],
        dep_var_units=[units.dimensionless], # Boolean masks are dimensionless
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
        dep_var_units=[units.dimensionless], # Boolean masks are dimensionless
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
        dep_var_units=[units.dimensionless], # Boolean masks are dimensionless
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
        dep_var_units=[units.dimensionless], # Boolean masks are dimensionless
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
        dep_var_units=[units.dimensionless], # Boolean masks are dimensionless
        meta=metadata,
    )