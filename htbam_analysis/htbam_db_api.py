from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
from typing import List, Tuple
import numpy as np
import json

class AbstractHtbamDBAPI(ABC):
    def __init__(self):
        pass

    # @abstractmethod
    # def get_standard_data(self, standard_name: str) -> Tuple[List[float], np.ndarray]:
    #     raise NotImplementedError

def _squeeze_df(df: pd.DataFrame, grouping_index: str, squeeze_targets: List[str]):
    '''
    Squeezes a dataframe along a given column to de-tidy the target data into lists.

            Parameters:
                    grouping_index (str): Aggregation column name
                    squeeze_targets ([str]): List of columns to reduce values to lists

            Returns:
                    sqeeuzed_df (pd.DataFrame): DF with columns == [grouping_index, *squeeze_targets]
    '''

    squeeze_func = lambda x : pd.Series([x[grouping_index].values[0]] + [x[col].tolist() for col in squeeze_targets], 
                                        index=[grouping_index] + squeeze_targets)
    return df.groupby(grouping_index).apply(squeeze_func)



class LocalHtbamDBAPI(AbstractHtbamDBAPI):
   

    def __init__(self, standard_curve_data_path: str, standard_name: str, standard_type: str, standard_units: str, kinetic_data_path: str, kinetic_name: str, kinetic_type: str, kinetic_units: str):
        super().__init__()
        
        self._standard_data = pd.read_csv(standard_curve_data_path)
        self._standard_name = standard_name
        self._standard_type = standard_type
        self._standard_units = standard_units
        self._standard_data['indices'] = self._standard_data.x.astype('str') + ',' + self._standard_data.y.astype('str')
        
        self._kinetic_data = pd.read_csv(kinetic_data_path)
        self._kinetic_name = kinetic_name
        self._kinetic_type = kinetic_type
        self._kinetic_units = kinetic_units
        self._kinetic_data['indices'] = self._kinetic_data.x.astype('str') + ',' + self._kinetic_data.y.astype('str')


        self._init_json_dict()
        self._load_std_data()
        self._load_kinetic_data()
        self._load_button_quant_data()


    def _init_json_dict(self) -> None:
        '''
        Populates an initial dictionary with chamber specific metadata.

                Parameters:
                        None

                Returns:
                        None
        '''        
        unique_chambers = self._standard_data[['id','x_center_chamber', 'y_center_chamber', 'radius_chamber', 
        'xslice', 'yslice', 'indices']].drop_duplicates(subset=["indices"]).set_index("indices")
        self._json_dict = {"chamber_metadata": unique_chambers.to_dict("index")}


    def _load_std_data(self) -> None:
        '''
        Populates an json_dict with standard curve data with the following schema:
        {standard_run_#: {
            name: str,
            type: str,
            conc_unit: str,
            assays: {
                1: {
                    conc: float,
                    _s: [0],time
                    chambers: {
                        1,1: {
                            sum_chamber: [...],
                            std_chamber: [...]
                        },
                        ...
                        }}}}

                Parameters:
                        None

                Returns:
                        None
        '''
        i = 0    
        std_assay_dict = {}
        for prod_conc, subset in self._standard_data.groupby("concentration_uM"):
            squeezed = _squeeze_df(subset, grouping_index="indices", squeeze_targets=['sum_chamber', 'std_chamber'])
            squeezed["time_s"] = pd.Series([[0]]*len(squeezed), index=squeezed.index) #this tomfoolery is used to create a list with a single value, 0, for the standard curve assays.
            std_assay_dict[i] = {
                "conc": prod_conc,
                "time_s": squeezed.iloc[0]["time_s"],
                "chambers": squeezed.drop(columns=["time_s", "indices"]).to_dict("index")}
            i += 1

        std_run_num = len([key for key in self._json_dict if "standard_" in key])
        self._json_dict["runs"] = {f"standard_{std_run_num}": {
            "name": self._standard_name,
            "type": self._standard_type,
            "conc_unit": self._standard_units,
            "assays": std_assay_dict
            }
        }

    def _load_kinetic_data(self) -> None:
        '''
        Populates an json_dict with kinetic data with the following schema:
        {kinetics_run_#: {
            name: str,
            type: str,
            conc_unit: str,
            assays: {
                1: {
                    conc: float,
                    time_s: [0],
                    chambers: {
                        1,1: {
                            sum_chamber: [...],
                            std_chamber: [...]
                        },
                        ...
                        }}}}

                Parameters:
                        None

                Returns:
                        None
        
                        
        {kintetcs: {
            substrate_conc: {
                time_s: [0,1, ...],
                chambers: {
                    1,1: {
                        sum_chamber: [...],
                        std_chamber: [...]
                    },
                    ...
                    }}}}

                Parameters:
                        None

                Returns:
                        None
        '''    
        # kin_dict = {}
        # for sub_conc, subset in self._kinetic_data.groupby("series_index"):
        #     squeezed = _squeeze_df(subset, grouping_index="indices", squeeze_targets=["time_s", "sum_chamber", "std_chamber"])
        #     kin_dict[sub_conc] = {
        #         "time_s": squeezed.iloc[0]["time_s"],
        #         "chambers": squeezed.drop(columns=["time_s", "indices"]).to_dict("index")}
            
        # self._json_dict["kinetics"] = kin_dict

        def parse_concentration(conc_str: str):
            '''
            Currently, we're storing substrate concentration as a string in the kinetics data.
            This will be changed in the future to store as a float + unit as a string. For now,
            we will parse jankily.
            '''
            print('Warning: parsing concentration from string')
            #first, remove the unit and everything following
            conc = conc_str.split(self._kinetic_units)[0]
            #concentration number uses underscore as decimal point. Here, we replace and convert to a float:
            conc = float(conc.replace("_", "."))
            return conc
            

        i = 0    
        kin_dict = {}
        for sub_conc, subset in self._kinetic_data.groupby("series_index"):
            squeezed = _squeeze_df(subset, grouping_index="indices", squeeze_targets=["time_s",'sum_chamber', 'std_chamber'])
            #squeezed["time_s"] = 0
            kin_dict[i] = {
                "conc": parse_concentration(sub_conc),
                "time_s": squeezed.iloc[0]["time_s"],
                "chambers": squeezed.drop(columns=["time_s", "indices"]).to_dict("index")}
            i += 1

        kinetics_run_num = len([key for key in self._json_dict if "kinetics_" in key])
        if "runs" not in self._json_dict:
            self._json_dict["runs"] = {}
        self._json_dict["runs"][f"kinetics_{kinetics_run_num}"] = {
            "name": self._kinetic_name,
            "type": self._kinetic_type,
            "conc_unit": self._kinetic_units,
            "assays": kin_dict
        }

    def _load_button_quant_data(self) -> None:
        '''
        Populates an json_dict with kinetic data with the following schema:
        {button_quant: {
         
            1,1: {
                sum_chamber: [...],
                std_chamber: [...]
            },
            ...
            }}

                Parameters:
                        None

                Returns:
                        None
        '''    
        unique_buttons = self._kinetic_data[["summed_button_Button_Quant","summed_button_BGsub_Button_Quant",
        "std_button_Button_Quant", "indices"]].drop_duplicates(subset=["indices"]).set_index("indices")

        self._json_dict["button_quant"] = unique_buttons.to_dict("index")  

    def __repr__(self) -> str:
        def recursive_string(d: dict, indent: int, width=5) -> str:
            s = "\t"*indent + '{\n'
            for i, (key, value) in enumerate(d.items()):
                if i == width:
                    s += "\t"*indent +"...\n"
                    break
                s += "\t"*indent + f"{key}: "
                if isinstance(value, dict):
                    s += "\n" + recursive_string(value, indent+1)
                else:
                    s += f"{value}\n"
            s += "\t"*indent + '}\n'
            return s
        
        return recursive_string(self._json_dict, 0)
    
    def export_json(self):
        '''This writes the database to file, as a dict -> json'''
        with open('db.json', 'w') as fp:
            json.dump(self._json_dict, fp, indent=4)