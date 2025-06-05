# Utilities
from typing import List, Tuple
import re
from copy import deepcopy
from pathlib import Path
import csv
import os
from tqdm import tqdm

# Numerical & Scientific
import numpy as np
import pandas as pd

# HTBAM
from htbam_db_api.htbam_db_api import AbstractHtbamDBAPI, HtbamDBException
from htbam_analysis.analysis.plotting import plot_chip
from htbam_analysis.analysis.fitting import fit_luminance_vs_time, fit_luminance_vs_concentration

# Plotting
import seaborn as sns


class HTBAMExperiment:
    def __init__(self, db_connection: AbstractHtbamDBAPI):
        self._db_conn = db_connection
        print("\nConnected to database.")
        print("Experiment found with the following runs:")
        print(self._db_conn.get_run_names())
        self._run_data = {}
    
    def __repr__(self):
        '''
        Returns a string representation of the HTBAMExperiment object.
        Output:
            str: a string representation of the HTBAMExperiment object.
        '''
        return str(self._db_conn)
    
    ### GETTERS & SETTERS ###
    def get_run_names(self) -> List[str]:
        '''
        Returns the names of all runs in the database.
        Output:
            List[str]: a list of run names.
        '''
        return self._db_conn.get_run_names()
    
    def get_run(self, run_name: str) -> dict:
        '''
        Returns the data for the given run.
        Input:
            run_name (str): the name of the run to be analyzed.
        Output:
            dict: a dictionary containing the data for the given run.
        '''
        if run_name not in self._run_data.keys():
            print("Existing run data not found. Fetching from database.")
            run_data = self._db_conn.get_run(run_name)
            self._run_data[run_name] = run_data

        return self._run_data[run_name]
    
    def set_run(self, run_name: str, run_data: dict):
        '''
        Sets the data for the given run.
        Input:
            run_name (str): the name of the run to be analyzed.
            run_data (dict): a dictionary containing the data for the given run.
        Output:
            None
        '''
        self._run_data[run_name] = run_data

    ### MASKING ###
    def apply_mask(self, run_name: str, dep_variables: list, save_as: str, mask_names: list):
        '''
        TODO: We should pass in the names of the masks to retrieve from the DB. For now, we pass in masks.
        Applies a mask to a dataset, where False => NaN
        Input:
            run_name (str): the name of the run we will be masking
            dep_variable (list): the name of the dependent variable that will be masked (e.g. ['slopes', 'intercepts'] )
            save_as (str): the name of the new, masked database object which will be saved.
            mask_names (list): list of the names of the saved masks to apply.
        Output:
            Saves a new database object with masked data.
        '''

        run_data = self.get_run(run_name)

        masked_run_data = deepcopy(run_data)

        mask_list = []
        for mask_name in mask_names:
            mask_run = self.get_run(mask_name)
            assert 'mask' in mask_run['dep_vars'].keys(), 'No \'mask\' value found in mask[\'dep_vars\']'
            mask_list.append( mask_run['dep_vars']['mask'])

        for dep_var in dep_variables:
            for mask in mask_list:
                assert  masked_run_data['dep_vars'][dep_var].shape == mask.shape
                
                # Turn Falses into NaNs
                masked_run_data['dep_vars'][dep_var] = np.where(mask, masked_run_data['dep_vars'][dep_var], np.nan)

        self.set_run(save_as, masked_run_data)


    ### PLOTTING ###
    def plot_standard_curve_chip(self, analysis_name: str, experiment_name: str):
        '''
        Plot a full chip with raw data and std curve slopes.

        Parameters:
            analysis_name (str): the name of the analysis to be plotted.
            experiment_name (str): the name of the raw experiment data to be plotted.

        Returns:
            None
        '''
        #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
        
        experiment_data = self.get_run(experiment_name) # Raw data from experiment (to show datapoints)
        analysis_data = self.get_run(analysis_name)     # Analysis data (to show slopes/intercepts)

        #slopes_to_plot = self._db_conn.get_analysis(run_name, 'linear_regression', 'slope')
        #intercepts_to_plot = self._db_conn.get_analysis(run_name, 'linear_regression', 'intercept')
        slopes_to_plot = analysis_data['dep_vars']['slope']          # (n_chambers)
        intercepts_to_plot = analysis_data['dep_vars']['intercept']  # (n_chambers)

        luminance = experiment_data['dep_vars']['luminance'] # (n_chambers, n_timepoints, n_conc)
        concentration = experiment_data['indep_vars']['concentration'] # (n_conc,)
        
       
        #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
        #chamber_names_dict = self._db_conn.get_chamber_name_dict()
        chamber_names = experiment_data['indep_vars']['chamber_IDs'] # (n_chambers,)
        sample_names =  experiment_data['indep_vars']['sample_IDs'] # (n_chambers,)

        # Create dictionary mapping chamber_id -> sample_name:
        sample_names_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            sample_names_dict[chamber_id] = sample_names[i]

        # Create dictionary mapping chamber_id -> slopes:
        slopes_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            slopes_dict[chamber_id] = slopes_to_plot[i]

        #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
        # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.
        def plot_chamber_slopes(chamber_id, ax):
            #parameters:

            x_data = concentration
            y_data = luminance[:, -1, chamber_names == chamber_id] # using last timepoint
            
            m = slopes_to_plot[chamber_names == chamber_id]
            b = intercepts_to_plot[chamber_names == chamber_id]
            
            #make a simple matplotlib plot
            ax.scatter(x_data, y_data)
            if not (np.isnan(m) or np.isnan(b)):
                #return False, no_update, no_update
                ax.plot(x_data, m*np.array(x_data) + b)
                ax.set_title(f'{chamber_id}: {sample_names_dict[chamber_id]}')
                ax.set_xlabel('Concentration')
                ax.set_ylabel('Luminance (RFU)')
            return ax
        
        plot_chip(slopes_dict, sample_names_dict, graphing_function=plot_chamber_slopes, title='Standard Curve: Slope')

    def plot_initial_rates_chip(self, analysis_name: str, experiment_name: str, skip_start_timepoint: bool = True):
        '''
        Plot a full chip with raw data and fit initial rates.

        Parameters:
            analysis_name (str): the name of the analysis to be plotted.
            experiment_name (str): the name of the experiment to be plotted.
            skip_start_timepoint (bool): whether to skip the first timepoint in the analysis (Sometimes are unusually low). Default is True.

        Returns:
            None
        '''
        #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
        
        experiment_data = self.get_run(experiment_name) # Raw data from experiment (to show datapoints)
        analysis_data = self.get_run(analysis_name)     # Analysis data (to show slopes/intercepts)
        
        slopes_to_plot = analysis_data['dep_vars']['slope']          # (n_conc, n_chambers)
        intercepts_to_plot = analysis_data['dep_vars']['intercept']  # (n_conc, n_chambers)

        luminance = experiment_data['dep_vars']['luminance'] # (n_chambers, n_timepoints, n_conc)
        concentration = experiment_data['indep_vars']['concentration'] # (n_conc,)
        time_data = experiment_data['indep_vars']['time'] # (n_conc, n_timepoints)

        # If skip_start_timepoint is True, we'll skip the first timepoint in the analysis
        if skip_start_timepoint:
            luminance = luminance[:, 1:, :] # (n_chambers, n_timepoints-1, n_conc)
            time_data = time_data[:, 1:] # (n_conc, n_timepoints-1)
            #slopes_to_plot = slopes_to_plot[:, 1:]
        
       
        #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
        #chamber_names_dict = self._db_conn.get_chamber_name_dict()
        chamber_names = experiment_data['indep_vars']['chamber_IDs'] # (n_chambers,)
        sample_names =  experiment_data['indep_vars']['sample_IDs'] # (n_chambers,)

        # Create dictionary mapping chamber_id -> sample_name:
        sample_names_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            sample_names_dict[chamber_id] = sample_names[i]

        # Create dictionary mapping chamber_id -> mean slopes:
        slopes_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            slopes_dict[chamber_id] = np.nanmean(slopes_to_plot[:, i])

        #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
        # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.

        def plot_chamber_initial_rates(chamber_id, ax):#, time_to_plot=time_to_plot):
            #N.B. Every so often, slope and line colors don't match up. Not sure why.
            
            #convert from 'x,y' to integer index in the array:
            #data_index = list(self._run_data[run_name]["chamber_idxs"]).index(chamber_id)
            x_data = time_data # same for all chambers              (n_timepoints, n_conc)
            y_data = luminance[:, :, chamber_names == chamber_id]  #(n_timepoints, n_conc)
        
            m = slopes_to_plot[:, chamber_names == chamber_id]
            b = intercepts_to_plot[:, chamber_names == chamber_id]
            
            colors = sns.color_palette('husl', n_colors=y_data.shape[0])

            for i in range(y_data.shape[0]): #over each concentration:

                ax.scatter(x_data[i], y_data[i,:].flatten(), color=colors[i], alpha=0.3) # raw data
                ax.plot(x_data[i], m[i]*x_data[i] + b[i], color=colors[i], alpha=1, linewidth=2, label=f'{concentration[i]}')  # fitted line

            ax.legend()
            return ax
        
        plot_chip(slopes_dict, sample_names_dict, graphing_function=plot_chamber_initial_rates, title='Kinetics: Initial Rates')

    def plot_initial_rates_vs_concentration_chip(self,
                                                 analysis_name: str,
                                                 model_fit_name: str = None,
                                                 x_log: bool = False,
                                                 y_log: bool = False):
        """
        Plot initial rates (slopes) vs substrate concentration for each chamber.
        Parameters:
            analysis_name (str): the name of the analysis data object to be plotted.
            model_fit_name (str): the name of the model data object to be plotted (contains MM or IC50 fits, etc).
            x_log (bool): whether to plot x-axis on a logarithmic scale. Default is False.
            y_log (bool): whether to plot y-axis on a logarithmic scale. Default is False.
        """
        # get raw experiment (for concentration) and analysis (for slopes)
        analysis_data   = self.get_run(analysis_name)

        slopes = analysis_data['dep_vars']['slope']           # Y: (n_conc, n_chambers)
        conc   = analysis_data['indep_vars']['concentration'] # X: (n_conc,)

        # If model_fit_name is provided, we can also plot the fitted model:
        if model_fit_name is not None:
            model_fit_data = self.get_run(model_fit_name)
            slopes_pred = model_fit_data['dep_vars']['y_pred']  # (n_conc, n_chambers)
            r_squared = model_fit_data['dep_vars']['r_squared']  # (n_chambers,)

        # chamber/sample info
        chambers = analysis_data['indep_vars']['chamber_IDs']
        samples  = analysis_data['indep_vars']['sample_IDs']

        # dicts for plot_chip
        sample_names = {cid: samples[i] for i, cid in enumerate(chambers)}
        mean_rates   = {cid: np.nanmean(slopes[:, i])
                        for i, cid in enumerate(chambers)}

        # per‐chamber inset: rate vs conc
        def plot_rates_vs_conc(cid, ax):
            idx = chambers == cid
            y = slopes[:, idx].flatten()
            x = conc
            ax.scatter(x, y, alpha=0.7)
            if model_fit_name is not None:
                y_pred = slopes_pred[:, idx].flatten()
                ax.plot(x, y_pred, color='red', label=f'R² = {r_squared[idx][0]:.2f}')
                ax.legend()
            ax.set_title(f"{cid}: {sample_names[cid]}")
            ax.set_xlabel("Concentration")
            ax.set_ylabel("Initial Rate")
            # set log scale if requested
            if x_log:
                ax.set_xscale('log')
            if y_log:
                ax.set_yscale('log')
            return ax

        plot_chip(mean_rates,
                  sample_names,
                  graphing_function=plot_rates_vs_conc,
                  title="Initial Rates vs Concentration")

    def plot_enzyme_concentration_chip(self, analysis_name: str, units:str, skip_start_timepoint: bool = True):
        '''
        Plot a full chip with raw data and fit initial rates.

        Parameters:
            analysis_name (str): the name of the analysis to be plotted.
            experiment_name (str): the name of the experiment to be plotted.
            skip_start_timepoint (bool): whether to skip the first timepoint in the analysis (Sometimes are unusually low). Default is True.

        Returns:
            None
        '''
        #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
        analysis_data = self.get_run(analysis_name)     # Analysis data (to show slopes/intercepts)
        
        concentration = analysis_data['dep_vars']['concentration'] # (n_conc,)
        
       
        #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
        #chamber_names_dict = self._db_conn.get_chamber_name_dict()
        chamber_names = analysis_data['indep_vars']['chamber_IDs'] # (n_chambers,)
        sample_names =  analysis_data['indep_vars']['sample_IDs'] # (n_chambers,)

        # Create dictionary mapping chamber_id -> sample_name:
        sample_names_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            sample_names_dict[chamber_id] = sample_names[i]

        # Create dictionary mapping chamber_id -> mean slopes:
        conc_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            conc_dict[chamber_id] = np.nanmean(concentration[i])

        #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
        # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.
        
        plot_chip(conc_dict, sample_names_dict, title=f'Enzyme Concentration ({units})')

    def plot_mask_chip(self, mask_name: str):
        '''
        Plot a full chip with raw data and fit initial rates.

        Parameters:
            mask_name (str): the name of the mask to be plotted.

        Returns:
            None
        '''
        #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
        mask_data = self.get_run(mask_name)     # Analysis data (to show slopes/intercepts)
        
        mask = mask_data['dep_vars']['mask'] # (n_conc, n_chambers,)

        # We want to plot the number of concentrations that pass the mask in each well.
        # so, we'll sum across the concentration dimension, leaving an (n_chambers,) array

        passed_conc = np.sum(mask, axis=0)

        #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
        chamber_names = mask_data['indep_vars']['chamber_IDs'] # (n_chambers,)
        sample_names =  mask_data['indep_vars']['sample_IDs'] # (n_chambers,)

        # Create dictionary mapping chamber_id -> sample_name:
        sample_names_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            sample_names_dict[chamber_id] = sample_names[i]

        # Create dictionary mapping chamber_id -> mean slopes:
        conc_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            conc_dict[chamber_id] = passed_conc[i]

        #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
        # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.
        
        plot_chip(conc_dict, sample_names_dict, title=f'# Concentrations that pass filter: {mask_name}')