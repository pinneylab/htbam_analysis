import time
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy
from typing import List, Dict

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
        initial_rate_data: a dictionary containing the slopes data (data format: linear_fit_data)
        filter_criteria: a dictionary containing the criteria for filtering
    Returns:
        initial_rates_r_squared_mask: a dictionary (format: linear_fit_data_mask) indicating which chambers pass the R2 cutoff
    '''

    assert initial_rate_data['data_type'] == 'linear_fit_data', f"Data type {initial_rate_data['data_type']} not supported. Requires 'linear_fit_data' format."
    

    # first we grab the R2 values:
    initial_rates_r_squared = initial_rate_data['dep_vars']['r_squared']
    mask = initial_rates_r_squared >= r2_cutoff
    
    # Turn into db object:
    initial_rates_r_squared_mask = deepcopy(initial_rate_data)  # Create a copy of the initial rate data
    initial_rates_r_squared_mask['data_type'] = 'linear_fit_data_mask'  # Update the data type to indicate this is a mask
    initial_rates_r_squared_mask['dep_vars'] = { 'mask': mask }  # Update the dep_vars with the mask
    
    return initial_rates_r_squared_mask

def filter_initial_rates_positive_cutoff(initial_rate_data):
    '''
    Filter initial rates data.
    Inputs:
        slopes_data: a dictionary containing the slopes data (data format: linear_fit_data)
        filter_criteria: a dictionary containing the criteria for filtering
    Returns:
        initial_rates_slope_mask: (n_conc, n_chamb) a boolean array indicating which chambers have positive slopes
    '''

    assert initial_rate_data['data_type'] == 'linear_fit_data', f"Data type {initial_rate_data['data_type']} not supported. Requires 'linear_fit_data' format."

    # Positive slope cutoff:
    initial_rates_slope = initial_rate_data['dep_vars']['slope']
    mask = initial_rates_slope > 0

    # Turn into db object:
    initial_rates_slope_mask = deepcopy(initial_rate_data)  # Create a copy of the initial rate data
    initial_rates_slope_mask['data_type'] = 'linear_fit_data_mask'  # Update the data type to indicate this is a mask
    initial_rates_slope_mask['dep_vars'] = { 'mask': mask }  # Update the dep_vars with the mask

    return initial_rates_slope_mask

def filter_expression_cutoff(expression_data, initial_rate_data, expression_cutoff):
    '''
    Filter initial rates data based on expression threshold.
    Inputs:
        expression_data: a dictionary containing the enzyme concentration data (data format: concentration_data)
        initial_rate_data: a dictionary with fit initial rate values (only used to obtain the shape of (n_conc, n_chambers) for output)
        expression_cutoff: the expression threshold for filtering
    Returns:
        expression_mask: (n_chamb,) a boolean array indicating which chambers pass the expression threshold
    '''

    assert expression_data['data_type'] == 'concentration_data', f"Data type {expression_data['data_type']} not supported. Requires 'concentration_data' format."
    assert initial_rate_data['data_type'] == 'linear_fit_data', f"Data type {initial_rate_data['data_type']} not supported. Requires 'linear_fit_data' format."

    # Get the expression values from the initial rates data
    expression_values = expression_data['dep_vars']['concentration'] # (n_chamb,)
    mask = expression_values >= expression_cutoff

    # Shape into n_conc, n_chambers
    n_conc, n_chambers = initial_rate_data['dep_vars']['slope'].shape
    mask = np.tile(mask, (n_conc, 1))

    # turn into DB format:
    expression_mask = deepcopy(initial_rate_data)  # Create a copy of the initial rate data
    expression_mask['data_type'] = 'linear_fit_data_mask'  # Update the data type to indicate this is a mask
    expression_mask['dep_vars'] = { 'mask': mask }  # Update the dep_vars with the mask
    
    return expression_mask # (n_conc, n_chambers)

def filter_standard_curve_r2_cutoff(standard_curve_fit_data, initial_rate_data, r2_cutoff):
    '''
    Filter standard curve data based on R2 cutoff.
    Inputs:
        standard_curve_fit_data: a dictionary containing the standard curve fit data (data format: linear_fit_data)
        initial_rate_data: a dictionary with fit initial rate values (only used to obtain the shape of (n_conc, n_chambers) for output)
        r2_cutoff: the R2 cutoff for the standard curve
    Returns:
        r2_cutoff_mask: (n_chamb,) a boolean array indicating which chambers pass the R2 cutoff
    '''

    assert standard_curve_fit_data['data_type'] == 'linear_fit_data', f"Data type {standard_curve_fit_data['data_type']} not supported. Requires 'linear_fit_data' format."
    assert initial_rate_data['data_type'] == 'linear_fit_data', f"Data type {initial_rate_data['data_type']} not supported. Requires 'linear_fit_data' format."

    # Get the R2 values from the standard curve fit data
    r_squared = standard_curve_fit_data['dep_vars']['r_squared'] # (n_chamb,)
    mask = r_squared >= r2_cutoff

    # Shape into n_conc, n_chambers
    n_conc, n_chambers = initial_rate_data['dep_vars']['slope'].shape
    mask = np.tile(mask, (n_conc, 1)) # (n_conc, n_chambers)

    # turn into DB format:
    r_squared_mask = deepcopy(initial_rate_data)  # Create a copy of the initial rate data
    r_squared_mask['data_type'] = 'linear_fit_data_mask'  # Update the data type to indicate this is a mask
    r_squared_mask['dep_vars'] = { 'mask': mask }  # Update the dep_vars with the mask

    return r_squared_mask