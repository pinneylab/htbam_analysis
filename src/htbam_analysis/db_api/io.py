from htbam_analysis.db_api.exceptions import HtbamDBException
from htbam_analysis.db_api.csv_processing import CSV_DATA_LABELS, parse_concentration
from htbam_analysis.db_api.csv_processing import process_dataframe_kinetics, process_dataframe_binding
from htbam_analysis.db_api.data import Data2D, Data3D, Data4D, IndepVars, Meta

from pathlib import Path
from copy import copy
from typing import Union

import pandas as pd
import numpy as np
import pint

def verify_file_exists(file_path: str) -> None:
        '''
        Verifies that a file exists at the given path.
        Returns an informative Error message if not.

            Parameters:
                    file_path (str): Path to the file

            Returns:
                    None
        '''

        # exists?
        if Path(file_path).is_file():
            return True

        # if not, check if the parent file even exists
        parent_file_exists = False
        parent_file_contents = []

        parent_file = Path(file_path).parent

        while not parent_file_exists and parent_file != Path('/'):
            if Path(parent_file).exists():
                parent_file_exists = True
                parent_file_contents = [str(f) for f in Path(parent_file).iterdir() if f.is_file()]
            else:
                parent_file = parent_file.parent
        
        if not parent_file_exists:
            raise HtbamDBException(f"File {file_path} does not exist. We cannot find any files matching the path provided")
        else:
            raise HtbamDBException(f"File {file_path} does not exist. We found the parent file {parent_file} but it does not contain the file you requested.\n \
                                   We found the following files in the parent directory:\n" + "\n".join(parent_file_contents))

def load_run_from_csv(
    csv_path: str,
    run_type: str,
    conc_unit: pint.Unit,
    time_unit: pint.Unit,
    concentration_col: str,
    *,
    signal_col: str = None,
    image_type_col: str = None,
    post_wash_prey_type: str = None,
    post_wash_bait_type: str = None,
) -> Union[Data3D, Data4D]:
    '''
    Loads a run from a CSV file, and processes it into a dict of numpy arrays.

    Arguments:
        csv_path: The path to the CSV file
        run_type: The type of run (kinetics, standard curve, etc.)
        conc_unit: The unit for the concentration (e.g. 'nM', 'uM', etc.)
        time_unit: The unit for the time (e.g. 's', 'min', etc.)
    Returns:
        A dict of numpy arrays in the 'kinetics' or 'binding' format.
    '''
    ### Load CSV
    df = pd.read_csv(csv_path)

    ### Pre-process CSV
    ### TODO: Unify standard curve and kinetics CSV formats on microscope, so we don't have to juggle here.
    L = copy(CSV_DATA_LABELS) # shorthand for labels dict.
    # Add units for time, and stdcurve concentration labels:
    L['time'] += f"{time_unit:~}"
    L['standardcurve_concentration'] += f"{conc_unit:~}".replace('µ', 'u') # cleans up uM

    # Handle legacy vs new time column names
    if 'time' in df.columns:
        L['time'] = 'time'
    elif 'time_s' in df.columns:
        L['time'] = 'time_s'
        
    if L['time'] not in df.columns:
        df[L['time']] = 0

    # Parse concentration using the user-specified column
    if concentration_col in df.columns:
        df[L['concentration']] = df[concentration_col].apply(lambda x: parse_concentration(x, conc_unit))
    else:
        raise HtbamDBException(f"Concentration column '{concentration_col}' not found in {csv_path}. Columns available: {df.columns.tolist()}")

    # Create unique Chamber_IDs as "x,y"
    df[L['chamber_IDs']] = df[L['chamber_x']].astype(str) + ',' + df[L['chamber_y']].astype(str)
    
    ### Sort
    # Sort the df first by chamber_id (using x and y to keep order correct), then by concentration, then by time
    # Warning: Modifying in-place here.
    df = df.sort_values(by=[L['chamber_x'], L['chamber_y'], L['concentration'], L['time']])

    ### Process data into numpy arrays:
    # N.F. I think we can have different functions to do this. Ideally we should keep data in the generally flexible "kinetics" format.
    # This format should work for kinetics (multiple concentrations and timepoints), inhibition, and standard curves (single timepoint).
    data_processing_functions = {
        'kinetics': process_dataframe_kinetics,
        'binding':  process_dataframe_binding
    }

    if run_type not in data_processing_functions:
        raise HtbamDBException(f"Unknown run_type '{run_type}'. Expected one of {list(data_processing_functions.keys())}.")

    if run_type == 'binding':
        L_bind = copy(CSV_DATA_LABELS)
        signal_col = signal_col or L_bind['binding_signal']
        image_type_col = image_type_col or L_bind['binding_image_type']
        if signal_col not in df.columns:
            raise HtbamDBException(
                f"Signal column '{signal_col}' not found in {csv_path}. Columns available: {df.columns.tolist()}"
            )
        if image_type_col not in df.columns:
            raise HtbamDBException(
                f"Image type column '{image_type_col}' not found in {csv_path}. Columns available: {df.columns.tolist()}"
            )
        run_data = process_dataframe_binding(
            df,
            time_unit,
            conc_unit,
            signal_col=signal_col,
            image_type_col=image_type_col,
            post_wash_prey_type=post_wash_prey_type,
            post_wash_bait_type=post_wash_bait_type,
        )
    else:
        run_data = data_processing_functions[run_type](df, time_unit, conc_unit)

    return run_data

from htbam_analysis.db_api.units import units

def load_button_quant_from_csv(csv_path: str) -> Data2D:
    '''
    Loads a button quant run from a dedicated button_quant CSV file, and processes it into a Data2D object.
    
    Arguments:
        csv_path: The path to the CSV file
    Returns:
        A Data2D object for button quant.
    '''
    df = pd.read_csv(csv_path)

    L = copy(CSV_DATA_LABELS)
    
    # Check for new vs legacy button quant column names
    bq_col = 'summed_button_BGsub' if 'summed_button_BGsub' in df.columns else L['button_quant_sum']
    
    if bq_col not in df.columns:
        raise HtbamDBException(f"Button quant column '{bq_col}' not found in {csv_path}")

    # Create unique Chamber_IDs as "x,y"
    df[L['chamber_IDs']] = df[L['chamber_x']].astype(str) + ',' + df[L['chamber_y']].astype(str)
    
    # Extract arrays
    chamber_ids = df[L['chamber_IDs']].to_numpy()
    sample_ids = df[L['sample_IDs']].to_numpy()
    button_quant = df[bq_col].to_numpy() * units.RFU
    
    button_quant_sum = button_quant[..., np.newaxis]
    
    # We initialize IndepVars with dummy arrays of correct dims for variables we don't have
    indep_vars = IndepVars(
        concentration=np.empty((0,)) * units.uM, 
        chamber_IDs=chamber_ids, 
        sample_IDs=sample_ids, 
        button_quant_sum=button_quant, 
        time=np.empty((0, 0)) * units.s
    )
    
    meta = Meta()

    button_quant_data = Data2D(
        indep_vars=indep_vars,
        meta=meta,
        dep_var=np.array(button_quant_sum.m),
        dep_var_type=["luminance"],
        dep_var_units=[units.RFU],
    )
    
    return button_quant_data