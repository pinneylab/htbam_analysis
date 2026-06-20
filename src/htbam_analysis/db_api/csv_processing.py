import numpy as np
import pint
from copy import copy

from htbam_analysis.db_api.units import units

# These are the CSV columns that correspond with our human-readable labels.
CSV_DATA_LABELS = {
    'concentration': 'concentration',   # we will construct this from raw_concentration
    'raw_concentration': 'series_index',
    'chamber_IDs': 'chamber_IDs',                 # we construct this from chamber_x, chamber_y
    'sample_IDs': 'id',
    'chamber_x': 'x',
    'chamber_y': 'y',
    'time': 'time_',                              # We'll add the units (usually s) later, so it looks like 'time_s'.
    'luminance': 'sum_chamber',
    'button_quant_sum': 'summed_button_BGsub_Button_Quant',
    'standardcurve_concentration': 'concentration_', # We'll add the units (usually uM) later, so it looks like 'concentration_uM'.
                                                     # We need this here because on the microscope, the concentration is named differently for standard experiments. We should change it to be uniform.
    'binding_signal': 'summed_button_BGsub',
    'binding_image_type': 'binding_image_type',
    'post_wash_prey_type': 'post_wash_prey',
    'post_wash_bait_type': 'post_wash_bait',
}

def process_dataframe_kinetics(df, time_unit: pint.Unit, conc_unit: pint.Unit):
    '''
    Turn a pre-processed experiment dataframe, and create a dict of numpy arrays in the 'RFU_data' format.
    
    Arguments:
        df: DataFrame of experiment data

    Returns:
        data_3d: Data3D object, with metadata, indep_vars, and dep_vars.
    '''
    L = copy(CSV_DATA_LABELS) # shorthand for labels dict.
    
    if 'time' in df.columns:
        L['time'] = 'time'
    else:
        L['time'] += f"{time_unit:~}"
        
    # The dataframe already has standard 'concentration' parsed by io.py
    L['concentration'] = 'concentration'
    # Chamber_IDs(length n_chambers)
    chamber_ids = df[L["chamber_IDs"]].unique()      # n_chambers

    # Get sample IDs (length n_chambers)
    chamber_to_sample_map = df.set_index(L["chamber_IDs"])[L['sample_IDs']].to_dict() # Create a mapping of Chamber_IDs to Sample_IDs
    sample_ids = np.array([chamber_to_sample_map[chamber] for chamber in chamber_ids]) # Map the Sample_IDs to the unique Chamber_IDs

    # Get button quant (length n_chambers)
    if L['button_quant_sum'] in df.columns:
        chamber_to_button_quant_map = df.set_index(L["chamber_IDs"])[L['button_quant_sum']].to_dict() # Create a mapping of Chamber_IDs to Button_Quant
        button_quant = np.array([chamber_to_button_quant_map[chamber] for chamber in chamber_ids]) # Map the Button_Quant to the unique Chamber_IDs
    else: 
        button_quant = np.nan * np.ones(len(chamber_ids))  # If no button quant, fill with NaNs

    # Chamber_IDs(length n_concentrations)
    concentrations = df[L['concentration']].unique() # n_concentrations

    # Time array (n_concentrations, n_timepoints). The values are time in seconds.
    time_array = np.array( list(
                                df[df[L["chamber_IDs"]] == df[L["chamber_IDs"]][0]] # Get just the first chamber
                                .groupby(L['concentration'])[L['time']]                                  # Group by the concentration column, and get just the time values.
                                .apply(list)) )                                                        # And convert the times for each concentration to a list.
                                                                                                              # Then we convert to a list of lists, and then to a numpy array.
    # RFU array (n_concentrations, n_timepoints, n_chambers)
    RFU_list_by_conc = []
    for conc_index, concentration in enumerate(concentrations):
        time_values = time_array[conc_index]
        #print(time_values)
        RFU_list_by_time = []
        for time_index, time in enumerate(time_values):
            # Get the RFU values for this concentration and time
            rfu_values = df[(df[L['concentration']] == concentration) & (df[L['time']] == time)][L['luminance']].to_list()
            RFU_list_by_time.append(rfu_values)
        # Append the RFU values for this concentration to the list
        RFU_list_by_conc.append(RFU_list_by_time)
    # Convert the list to a numpy array
    RFU_array = np.array(RFU_list_by_conc)
    # Expand by 1 axis so it's (n_concentrations, n_timepoints, n_chambers, 1)
    RFU_array = RFU_array[..., np.newaxis]
    
    ### Adding units:
    concentrations = concentrations * conc_unit
    button_quant = button_quant * units.RFU
    time_array = time_array * time_unit
    
    ### Create the data object:
    from htbam_analysis.db_api.data import Data4D, IndepVars, Meta
    # Independent variables:
    indep_vars = IndepVars(concentrations, chamber_ids, sample_ids, button_quant, time_array)
    # Meta data:
    meta = Meta()  # Currently empty, but can be extended in the future.
    # 3D Data object:
    data_4d = Data4D(indep_vars=indep_vars, 
                     meta=meta,
                     dep_var=RFU_array, 
                     dep_var_type=['luminance'],
                     dep_var_units=[units.RFU])

    return data_4d

def process_dataframe_binding(
    df,
    time_unit: pint.Unit,
    conc_unit: pint.Unit,
    signal_col: str = None,
    image_type_col: str = None,
    post_wash_prey_type: str = None,
    post_wash_bait_type: str = None,
):
    '''
    Turn a long-format binding dataframe into a Data3D object.

    Expects one row per chamber/concentration/image-type with button signal in
    `signal_col` and image label in `image_type_col`. Pivots post-wash prey and
    bait rows, then computes fluorescence_ratio = post_wash_prey / post_wash_bait.
    '''
    L = copy(CSV_DATA_LABELS)
    signal_col = signal_col or L['binding_signal']
    image_type_col = image_type_col or L['binding_image_type']
    post_wash_prey_type = post_wash_prey_type or L['post_wash_prey_type']
    post_wash_bait_type = post_wash_bait_type or L['post_wash_bait_type']

    if signal_col not in df.columns:
        raise ValueError(f"Signal column '{signal_col}' not found in binding dataframe.")
    if image_type_col not in df.columns:
        raise ValueError(f"Image type column '{image_type_col}' not found in binding dataframe.")

    L['concentration'] = 'concentration'

    required_types = {post_wash_prey_type, post_wash_bait_type}
    present_types = set(df[image_type_col].dropna().unique())
    missing_types = required_types - present_types
    if missing_types:
        raise ValueError(
            f"Binding dataframe missing required image types: {sorted(missing_types)}. "
            f"Found: {sorted(present_types)}"
        )

    subset = df[df[image_type_col].isin(required_types)].copy()
    wide = subset.pivot_table(
        index=[L['chamber_IDs'], L['concentration'], L['sample_IDs']],
        columns=image_type_col,
        values=signal_col,
        aggfunc='first',
    ).reset_index()
    wide.columns.name = None

    chamber_ids = wide[L["chamber_IDs"]].unique()
    chamber_to_sample_map = wide.set_index(L["chamber_IDs"])[L['sample_IDs']].to_dict()
    sample_ids = np.array([chamber_to_sample_map[chamber] for chamber in chamber_ids])

    if L['button_quant_sum'] in df.columns:
        bq_subset = df.drop_duplicates(subset=[L['chamber_IDs']], keep='first')
        chamber_to_button_quant_map = bq_subset.set_index(L["chamber_IDs"])[L['button_quant_sum']].to_dict()
        button_quant = np.array([chamber_to_button_quant_map.get(chamber, np.nan) for chamber in chamber_ids])
    else:
        button_quant = np.nan * np.ones(len(chamber_ids))

    concentrations = wide[L['concentration']].unique()

    ratio_list_by_conc = []
    for concentration in concentrations:
        rows = wide[wide[L['concentration']] == concentration]
        prey_vals = rows[post_wash_prey_type].to_numpy(dtype=float)
        bait_vals = rows[post_wash_bait_type].to_numpy(dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(bait_vals == 0, np.nan, prey_vals / bait_vals)
        ratio_list_by_conc.append(ratios)

    ratio_array = np.array(ratio_list_by_conc)[..., np.newaxis]

    concentrations = concentrations * conc_unit
    button_quant = button_quant * units.RFU

    from htbam_analysis.db_api.data import Data3D, IndepVars, Meta
    indep_vars = IndepVars(
        concentrations,
        chamber_ids,
        sample_ids,
        button_quant,
        np.empty((0, 0)) * time_unit,
    )
    meta = Meta()
    data_3d = Data3D(
        indep_vars=indep_vars,
        meta=meta,
        dep_var=ratio_array,
        dep_var_type=['fluorescence_ratio'],
        dep_var_units=[units.dimensionless],
    )

    return data_3d

def parse_concentration(conc_str: str, unit: pint.Unit) -> float:
    '''
    Currently, we're storing substrate concentration as a string in the kinetics data.
    This will be changed in the future to store as a float + unit as a string. For now,
    we will parse jankily.

    Arguments:
        conc_str: The concentration string to parse
        unit: The unit to remove from the string, as a pint.Unit

    Returns:
        The concentration as a float
    '''
    if isinstance(conc_str, (float, int)):
        return float(conc_str)
        
    #first, remove the unit and everything following
    unit_str = f"{unit:~}".replace('µ', 'u')
    conc = str(conc_str).split(unit_str)[0]
    #concentration number uses underscore as decimal point. Here, we replace and convert to a float:
    conc = float(conc.replace("_", "."))
    return conc
