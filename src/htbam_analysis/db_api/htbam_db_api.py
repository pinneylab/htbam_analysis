from abc import ABC, abstractmethod, abstractproperty
from htbam_analysis.db_api.data import Data2D, Data3D, Data4D, Meta
import pandas as pd
from typing import List, Literal, Optional, Tuple
import numpy as np
import json
from pathlib import Path
import re
from copy import deepcopy
import pint

from htbam_analysis.db_api.exceptions import HtbamDBException
from htbam_analysis.db_api.io import verify_file_exists, load_run_from_csv, load_button_quant_from_csv
from htbam_analysis.db_api.units import units

class AbstractHtbamDBAPI(ABC):
    def __init__(self):
        pass

    # @abstractmethod
    # def get_standard_data(self, standard_name: str) -> Tuple[List[float], np.ndarray]:
    #     raise NotImplementedError

    # @abstractmethod
    # def get_run_assay_data(self, run_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     raise NotImplementedError
    
    # @abstractmethod
    # def create_analysis(self, run_name: str):
    #     raise NotImplementedError

class LocalHtbamDBAPI(AbstractHtbamDBAPI):

    def __init__(
        self,
        standard_curve_data_path: Optional[str] = None,
        standard_name: Optional[str] = None,
        standard_substrate: Optional[str] = None,
        standard_units: Optional[pint.Unit] = None,
        standard_concentration_col: Optional[str] = None,
        kinetic_data_path: Optional[str] = None,
        kinetic_name: Optional[str] = None,
        kinetic_substrate: Optional[str] = None,
        kinetic_units: Optional[pint.Unit] = None,
        kinetic_concentration_col: Optional[str] = None,
        time_units: Optional[pint.Unit] = None,
        button_quant_data_path: Optional[str] = None,
    ):
        super().__init__()
        self._init_json_dict()

        legacy_args = [
            standard_curve_data_path, standard_name, standard_units, standard_concentration_col,
            kinetic_data_path, kinetic_name, kinetic_units, kinetic_concentration_col,
            time_units, button_quant_data_path,
        ]
        if any(arg is not None for arg in legacy_args):
            if not all(arg is not None for arg in legacy_args):
                raise HtbamDBException(
                    "Legacy LocalHtbamDBAPI initialization requires all standard, kinetic, "
                    "and button_quant arguments to be provided."
                )
            self.load_run(
                standard_name,
                standard_curve_data_path,
                'kinetics',
                conc_unit=standard_units,
                time_unit=time_units,
                concentration_col=standard_concentration_col,
            )
            self.load_run(
                kinetic_name,
                kinetic_data_path,
                'kinetics',
                conc_unit=kinetic_units,
                time_unit=time_units,
                concentration_col=kinetic_concentration_col,
            )
            self.load_run('button_quant', button_quant_data_path, 'button_quant')

    def load_run(
        self,
        run_name: str,
        csv_path: str,
        run_type: Literal["kinetics", "binding", "button_quant"],
        *,
        conc_unit: pint.Unit = None,
        time_unit: pint.Unit = None,
        concentration_col: str = None,
        signal_col: str = None,
        image_type_col: str = None,
        post_wash_prey_type: str = None,
        post_wash_bait_type: str = None,
    ) -> None:
        '''
        Load a run from CSV and add it to the database.
        '''
        verify_file_exists(csv_path)

        if run_type in ('kinetics', 'binding'):
            if conc_unit is None or time_unit is None or concentration_col is None:
                raise HtbamDBException(
                    f"run_type '{run_type}' requires conc_unit, time_unit, and concentration_col."
                )
            run_data = load_run_from_csv(
                csv_path,
                run_type,
                conc_unit,
                time_unit,
                concentration_col,
                signal_col=signal_col,
                image_type_col=image_type_col,
                post_wash_prey_type=post_wash_prey_type,
                post_wash_bait_type=post_wash_bait_type,
            )
        elif run_type == 'button_quant':
            run_data = load_button_quant_from_csv(csv_path)
        else:
            raise HtbamDBException(f"Unknown run_type '{run_type}'.")

        self.add_run(run_name, run_data)


    def _init_json_dict(self) -> None:
        '''
        Populates an initial dictionary with chamber specific metadata.

        Parameters:
                None

        Returns:
                None
        ''' 
        self._json_dict = dict()
        self._json_dict["metadata"] = dict() # Will contain chamber_IDs, sample_IDs as 1D numpy arrays of shape (n_chambers, )
        self._json_dict["runs"] = dict()


    def __repr__(self) -> str:
        '''
        Returns a string representation of the object.

        Returns:
                str: A string representation of the object
        '''
        def recursive_string(d: dict, indent: int, width=5) -> str:
            s = ""

            # How many keys are in the dictionary?
            num_keys = len(d)

            # If there are more than 20 keys, we're in a data-dense region. Let's only show the first 5.
            if num_keys > 20: 
                truncate=True 
            else: 
                truncate=False

            for i, (key, value) in enumerate(d.items()):
                if i > 5 and truncate:
                    s += "\t" * indent + "...\n"
                    break
                s += "\t" * indent + str(key) + ": "
                if isinstance(value, dict):
                    s += "\n" + recursive_string(value, indent + 1)
                else:
                    data_string = ""
                    data_string += str(type(value)) + " "
                    if isinstance(value, np.ndarray):
                        data_string += str(value.shape) + " "
                    value_string = str(value)
                    value_string = value_string.replace("\n", " ").replace("\t", " ")
                    if len(value_string) > 30:
                        data_string += value_string[:30] + "..."
                    else:
                        data_string += value_string
                    s += f"{data_string}\n"
            s += "\t"*indent + '}\n'
            return s
        
        return recursive_string(self._json_dict, 0)

    ### GETTERS & SETTERS
    def add_run(self, run_name: str, run_data) -> None:
        '''
        Adds a run to the database.

                Parameters:
                        run_name (str): Name of the run
                        run_data (Data4D, Data3D, etc): Data for the run

                Returns:
                        None
        '''
        # Check if the format matches one of the allowed dataclasses:
        if not isinstance(run_data, (Data2D, Data3D, Data4D)):
            raise HtbamDBException(f"Run data must be of type Data2D, Data3D, or Data4D. Got {type(run_data)}")
        
        # Add to the database:
        self._json_dict['runs'][run_name] = run_data
        return
    
    def get_run(self, run_name: str) -> dict:
        '''
        Gets a run from the database.

                Parameters:
                        run_name (str): Name of the run

                Returns:
                        dict: Data for the run
        '''
        if run_name not in self._json_dict['runs'].keys():
            raise HtbamDBException(f"Run {run_name} not found in database.")
        
        return self._json_dict['runs'][run_name]
    
    def get_metadata(self, name: str) -> dict:
        '''
        Gets metadata from the database.

                Parameters:
                        name (str): Name of the metadata

                Returns:
                        dict: Metadata
        '''
        # TODO: unused
        if name not in self._json_dict['metadata'].keys():
            raise HtbamDBException(f"Metadata {name} not found in database.")
        
        return self._json_dict['metadata'][name]
    
    def get_run_names(self):
        '''
        Gets the names of all runs in the database.
            
        Parameters:
            None
        Returns:
            list: List of run names
        '''
        # TODO: This should also return the run names of the runs in db_conn
        return [key for key in self._json_dict['runs'].keys()]

    def set_metadata(self, name: str, value: str) -> None:
        '''
        Sets metadata in the database.

                Parameters:
                        name (str): Name of the metadata
                        value (str): Value of the metadata

                Returns:
                        None
        '''
        # TODO: unused
        self._json_dict['metadata'][name] = value

         
    # def export_json(self):
    #     '''This writes the database to file, as a dict -> json'''
    #     with open('db.json', 'w') as fp:
    #         json.dump(self._json_dict, fp, indent=4)
