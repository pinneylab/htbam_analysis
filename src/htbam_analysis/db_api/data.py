from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy
import pint

@dataclass
class IndepVars:
    concentration: pint.Quantity     # (n_conc,)
    chamber_IDs: np.ndarray       # (n_chamb,)   # These two are dimensionless
    sample_IDs: np.ndarray        # (n_chamb,)   # since they're just labels
    button_quant_sum: pint.Quantity  # (n_chamb,)
    time: pint.Quantity              # (n_conc, n_time)

    def __post_init__(self):
        for var in self.concentration, self.chamber_IDs, self.sample_IDs, self.button_quant_sum:
            if var.ndim != 1:
                raise ValueError(f"Expected 1D array, got {var.shape} for {var}")
        if self.time.ndim != 2:
            raise ValueError(f"time must be 2D, got {self.time.shape}")

@dataclass
class Meta:
    # TODO: flesh this out
    based_on: list[str] = field(default_factory=list)  # e.g. ['previous_run_1', previous_run_2'] if fit/masked from previus data
    description: str = field(default='')
    applied_masks: list[str] = field(default_factory=list)  # e.g. ['saved_mask_1', 'saved_mask_2'] if applied to this data
    # If it's a curve fit:
    fit_type: str = field(default='')  # e.g. 'linear', 'MM', etc.
    # If it's a mask:
    mask_type: str = field(default='')  # e.g. 'r_squared', 'positive_slope', etc.
    mask_cutoff: float = field(default=0.0)  # e.g. 0.9 for R2 cutoff

@dataclass
class Data4D:
    indep_vars: IndepVars

    dep_var: np.ndarray           # (n_conc, n_time, n_chamb, n_values)
    dep_var_type: list[str]       # e.g. ['luminance'] or ['slopes', 'intercepts']
    dep_var_units: list[pint.Unit]  # e.g. [units.RFU] or [units.RFU / units.s, units.RFU]
    
    meta: Meta = field(default_factory=Meta)

    def __post_init__(self):
        # make a full copy so original IndepVars isn’t shared
        self.indep_vars = deepcopy(self.indep_vars)
        self.indep_vars.__post_init__()  # validate copied indep vars
        if self.dep_var.ndim != 4:
            raise ValueError(f"dep_var must be 4D, got {self.dep_var.shape}")

@dataclass
class Data3D:
    indep_vars: IndepVars

    dep_var: np.ndarray           # (n_conc, n_chamb, n_values)
    dep_var_type: list[str]       # e.g. ['luminance'] or ['slopes', 'intercepts']
    dep_var_units: list[pint.Unit]  # e.g. [units.RFU] or [units.RFU / units.s, units.RFU]

    meta: Meta = field(default_factory=Meta)

    def __post_init__(self):
        self.indep_vars = deepcopy(self.indep_vars)
        self.indep_vars.__post_init__()
        if self.dep_var.ndim != 3:
            raise ValueError(f"dep_var must be 3D, got {self.dep_var.shape}")

@dataclass
class Data2D:
    indep_vars: IndepVars
    
    dep_var: np.ndarray           # (n_chambers, n_values)
    dep_var_type: list[str]       # e.g. ['luminance'] or ['slopes', 'intercepts']
    dep_var_units: list[pint.Unit]  # e.g. [units.RFU] or [units.RFU / units.s, units.RFU]
    
    meta: Meta = field(default_factory=Meta)

    def __post_init__(self):
        self.indep_vars = deepcopy(self.indep_vars)
        self.indep_vars.__post_init__()
        if self.dep_var.ndim != 2:
            raise ValueError(f"dep_var must be 2D, got {self.dep_var.shape}")
