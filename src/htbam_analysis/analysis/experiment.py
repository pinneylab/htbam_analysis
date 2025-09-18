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
from htbam_db_api.data import Data4D, Data3D, Data2D, Meta
from htbam_analysis.analysis.plot import plot_chip
#from htbam_analysis.analysis.fit import fit_luminance_vs_time, fit_luminance_vs_concentration

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
        """
        Applies boolean masks (DataND) to specified dependent vars in a Data3D/4D run.
        """
        run_data = self.get_run(run_name)
        dtype = type(run_data)

        # target shape for mask (all axes except dep-var axis)
        target_shape = run_data.dep_var.shape[:-1]  # e.g., (n_conc, n_time, n_chambers) or (n_conc, n_chambers)

        # combine all masks into a single boolean array shaped like target_shape
        combined = None
        for m in mask_names:
            current_mask = self.get_run(m)
            assert current_mask.dep_var_type == ["mask"], "mask must be DataND with dep_var_type ['mask']"
            mat = current_mask.dep_var[..., 0].astype(bool)  # drop mask channel, ensure bool

            # # Make mask broadcastable to data shape:
            # if mat.shape == target_shape:
            #     pass
            # elif len(target_shape) == 3 and mat.shape == (target_shape[0], target_shape[2]):
            #     # mask lacks time axis -> expand over time
            #     mat = np.broadcast_to(mat[:, None, :], target_shape)
            # else:
            #     raise ValueError(f"Incompatible mask shape {mat.shape} for data shape {target_shape}")

            combined = mat if combined is None else (combined & mat)

        # IMPORTANT: upcast to float so NaN is representable
        new_dep_var = run_data.dep_var.astype(float, copy=True)

        # mask each requested dep var
        for dv in dep_variables:
            idx = run_data.dep_var_type.index(dv)
            new_dep_var[..., idx] = np.where(combined, new_dep_var[..., idx], np.nan)

        meta = Meta(
            based_on=[run_name] + mask_names,
            applied_masks=mask_names,
        )
        masked = dtype(
            indep_vars=run_data.indep_vars,
            dep_var=new_dep_var,
            dep_var_type=run_data.dep_var_type,
            meta=meta
        )
        self.set_run(save_as, masked)


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

        slope_idx = analysis_data.dep_var_type.index('slope')          # index of slope in dep_vars
        intercept_idx = analysis_data.dep_var_type.index('intercept')  # index of intercept in dep_vars
        r_squared_idx = analysis_data.dep_var_type.index('r_squared')  # index of r_squared in dep_vars
        
        # Extract slopes and intercepts from analysis data
        slopes_to_plot = analysis_data.dep_var[..., slope_idx]          # (n_chambers,)
        intercepts_to_plot = analysis_data.dep_var[..., intercept_idx]  # (n_chambers,)
        r_squared = analysis_data.dep_var[..., r_squared_idx]  # (n_chambers,)

        # Extract luminance and concentration from experiment data
        luminance_idx = experiment_data.dep_var_type.index('luminance')  # index of luminance in dep_vars
        luminance = experiment_data.dep_var[..., luminance_idx]  # (n_chambers, n_timepoints, n_conc)
        concentration = experiment_data.indep_vars.concentration # (n_conc,)
        
        #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
        chamber_names = experiment_data.indep_vars.chamber_IDs # (n_chambers,)
        sample_names =  experiment_data.indep_vars.sample_IDs # (n_chambers,)

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
        
        # Extract slopes and intercepts from analysis data
        slopes_idx = analysis_data.dep_var_type.index('slope')          # index of slope in dep_vars
        intercepts_idx = analysis_data.dep_var_type.index('intercept')  # index of intercept in dep_vars
        r_squared_idx = analysis_data.dep_var_type.index('r_squared')  # index of r_squared in dep_vars

        # Extract slopes and intercepts from analysis data
        slopes_to_plot = analysis_data.dep_var[..., slopes_idx]          # (n_chambers, n_conc)
        intercepts_to_plot = analysis_data.dep_var[..., intercepts_idx]  # (n_chambers, n_conc)
        r_squared = analysis_data.dep_var[..., r_squared_idx]          # (n_chambers, n_conc)

        # Extract product_concentration (Y) from experiment data
        product_conc_idx = experiment_data.dep_var_type.index('concentration')  # index of luminance in dep_vars
        product_conc = experiment_data.dep_var[..., product_conc_idx]  # (n_chambers, n_timepoints, n_conc)
        substrate_conc = experiment_data.indep_vars.concentration # (n_conc,)
        time_data = experiment_data.indep_vars.time # (n_conc, n_timepoints)

        # If skip_start_timepoint is True, we'll skip the first timepoint in the analysis
        if skip_start_timepoint:
            product_conc = product_conc[:, 1:, :] # (n_chambers, n_timepoints-1, n_conc)
            time_data = time_data[:, 1:] # (n_conc, n_timepoints-1)
            #slopes_to_plot = slopes_to_plot[:, 1:]
       
        #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
        #chamber_names_dict = self._db_conn.get_chamber_name_dict()
        chamber_names = experiment_data.indep_vars.chamber_IDs # (n_chambers,)
        sample_names =  experiment_data.indep_vars.sample_IDs # (n_chambers,)
        
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
            y_data = product_conc[:, :, chamber_names == chamber_id]  #(n_timepoints, n_conc)
        
            m = slopes_to_plot[:, chamber_names == chamber_id]
            b = intercepts_to_plot[:, chamber_names == chamber_id]
            
            colors = sns.color_palette('husl', n_colors=y_data.shape[0])

            for i in range(y_data.shape[0]): #over each substrate concentration:

                ax.scatter(x_data[i], y_data[i,:].flatten(), color=colors[i], alpha=0.3) # raw data
                ax.plot(x_data[i], m[i]*x_data[i] + b[i], color=colors[i], alpha=1, linewidth=2, label=f'{substrate_conc[i]}')  # fitted line

            ax.legend()
            return ax
        
        plot_chip(slopes_dict, sample_names_dict, graphing_function=plot_chamber_initial_rates, title='Kinetics: Initial Rates')

    def plot_initial_rates_vs_concentration_chip(self,
                                                 analysis_name: str,
                                                 model_fit_name: str = None,
                                                 model_pred_data_name: str = None,
                                                 x_log: bool = False,
                                                 y_log: bool = False):
        """
        Plot initial rates vs substrate concentration for each chamber.
        Optionally overlay fitted curve from `model_pred_data_name` and
        annotate fit parameters from `model_fit_name`.
        """
        analysis: Data3D = self.get_run(analysis_name)
        si = analysis.dep_var_type.index("slope")
        slopes = analysis.dep_var[..., si]               # (n_conc, n_chambers)
        conc   = analysis.indep_vars.concentration       # (n_conc,)

        if model_pred_data_name:
            mf_pred: Data3D = self.get_run(model_pred_data_name)
            yi = mf_pred.dep_var_type.index("y_pred")
            preds = mf_pred.dep_var[..., yi]           # (n_conc, n_chambers)

        if model_fit_name:
            mf_fit: Data2D = self.get_run(model_fit_name)
            fit_types = mf_fit.dep_var_type           # e.g. ["v_max","K_m","r_squared"]
            fit_vals = mf_fit.dep_var                # shape (n_chamb, len(fit_types))

        chambers = analysis.indep_vars.chamber_IDs        # (n_chambers,)
        samples  = analysis.indep_vars.sample_IDs         # (n_chambers,)
        sample_names = {cid: samples[i] for i, cid in enumerate(chambers)}
        mean_rates   = {cid: np.nanmean(slopes[:, i]) for i, cid in enumerate(chambers)}

        def plot_rates_vs_conc(cid, ax):
            idx = (chambers == cid)
            x   = conc
            y   = slopes[:, idx].flatten()
            ax.scatter(x, y, alpha=0.7)

            if model_pred_data_name:
                y_p = preds[:, idx].flatten()
                ax.plot(x, y_p, color="red", label="model")

            if model_fit_name:
                # extract this chamber's fit row
                chamb_idx = np.where(chambers == cid)[0][0]
                vals = fit_vals[chamb_idx]
                txt = "".join(f"{nm}={v:.2f}\n" for nm,v in zip(fit_types, vals))
                ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                        va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

            if model_pred_data_name or model_fit_name:
                ax.legend()
            ax.set_title(f"{cid}: {sample_names[cid]}")
            ax.set_xlabel("Concentration")
            ax.set_ylabel("Initial Rate")
            if x_log: ax.set_xscale("log")
            if y_log: ax.set_yscale("log")
            return ax

        plot_chip(mean_rates, sample_names,
                  graphing_function=plot_rates_vs_conc,
                  title="Initial Rates vs Concentration")

    def plot_MM_chip(self,
                    analysis_name: str,
                    model_fit_name: str,
                    model_pred_data_name: str = None,
                    x_log: bool = False,
                    y_log: bool = False):
        """
        Plot MM values, with inset initial rates vs substrate concentration for each chamber.
        Optionally overlay fitted curve from `model_pred_data_name` and
        annotate fit parameters from `model_fit_name`.
        """
        analysis: Data3D = self.get_run(analysis_name)
        si = analysis.dep_var_type.index("slope")
        slopes = analysis.dep_var[..., si]               # (n_conc, n_chambers)
        conc   = analysis.indep_vars.concentration       # (n_conc,)

        mf_fit: Data2D = self.get_run(model_fit_name)
        fit_types = mf_fit.dep_var_type           # e.g. ["v_max","K_m","r_squared"]
        fit_vals = mf_fit.dep_var                # shape (n_chamb, len(fit_types))

        mm_idx = mf_fit.dep_var_type.index("v_max")
        mms = mf_fit.dep_var[..., mm_idx]    # (n_chambers,)

        if model_pred_data_name:
            mf_pred: Data3D = self.get_run(model_pred_data_name)
            yi = mf_pred.dep_var_type.index("y_pred")
            preds = mf_pred.dep_var[..., yi]           # (n_conc, n_chambers)

        chambers = analysis.indep_vars.chamber_IDs        # (n_chambers,)
        samples  = analysis.indep_vars.sample_IDs         # (n_chambers,)
        sample_names = {cid: samples[i] for i, cid in enumerate(chambers)}
        #mean_rates   = {cid: np.nanmean(slopes[:, i]) for i, cid in enumerate(chambers)}
        mms_to_plot = {cid: mms[i] for i, cid in enumerate(chambers)}

        def plot_rates_vs_conc(cid, ax):
            idx = (chambers == cid)
            x   = conc
            y   = slopes[:, idx].flatten()
            ax.scatter(x, y, alpha=0.7, label="current well")

            if model_pred_data_name:
                # show envelope of model fits for all wells with this sample
                sample = sample_names[cid]
                same_idxs = [i for i, s in enumerate(samples) if s == sample]
                y_all = preds[:, same_idxs]               # (n_conc, n_same)
                y_min = np.nanmin(y_all, axis=1)
                y_max = np.nanmax(y_all, axis=1)
                ax.fill_between(x, y_min, y_max, color="gray", alpha=0.3, label='other well fits')
                # then overplot this chamber’s model fit
                y_p = preds[:, idx].flatten()
                ax.plot(x, y_p, color="red", label="current well fit")

            if model_fit_name:
                # extract this chamber's fit row
                chamb_idx = np.where(chambers == cid)[0][0]
                vals = fit_vals[chamb_idx]
                txt = "".join(f"{nm}={v:.2f}\n" for nm,v in zip(fit_types, vals))
                ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                        va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            
            if model_pred_data_name or model_fit_name:
                ax.legend()
            ax.set_title(f"{cid}: {sample_names[cid]}")
            ax.set_xlabel("Concentration")
            ax.set_ylabel("Initial Rate")
            if x_log: ax.set_xscale("log")
            if y_log: ax.set_yscale("log")
            return ax

        plot_chip(mms_to_plot, sample_names,
                  graphing_function=plot_rates_vs_conc,
                  title="Initial Rates vs Concentration")
        
    
    def plot_ic50_chip(self,
                    analysis_name: str,
                    model_fit_name: str,
                    model_pred_data_name: str = None,
                    x_log: bool = False,
                    y_log: bool = False):
        """
        Plot ic50 values, with inset initial rates vs substrate concentration for each chamber.
        Optionally overlay fitted curve from `model_pred_data_name` and
        annotate fit parameters from `model_fit_name`.
        """
        analysis: Data3D = self.get_run(analysis_name)
        si = analysis.dep_var_type.index("slope")
        slopes = analysis.dep_var[..., si]               # (n_conc, n_chambers)
        conc   = analysis.indep_vars.concentration       # (n_conc,)

        mf_fit: Data2D = self.get_run(model_fit_name)
        fit_types = mf_fit.dep_var_type           # e.g. ["v_max","K_m","r_squared"]
        fit_vals = mf_fit.dep_var                # shape (n_chamb, len(fit_types))

        ic50s_idx = mf_fit.dep_var_type.index("ic50")
        ic50s = mf_fit.dep_var[..., ic50s_idx]    # (n_chambers,)

        if model_pred_data_name:
            mf_pred: Data3D = self.get_run(model_pred_data_name)
            yi = mf_pred.dep_var_type.index("y_pred")
            preds = mf_pred.dep_var[..., yi]           # (n_conc, n_chambers)

        chambers = analysis.indep_vars.chamber_IDs        # (n_chambers,)
        samples  = analysis.indep_vars.sample_IDs         # (n_chambers,)
        sample_names = {cid: samples[i] for i, cid in enumerate(chambers)}
        #mean_rates   = {cid: np.nanmean(slopes[:, i]) for i, cid in enumerate(chambers)}
        ic50s_to_plot = {cid: ic50s[i] for i, cid in enumerate(chambers)}

        def plot_rates_vs_conc(cid, ax):
            idx = (chambers == cid)
            x   = conc
            y   = slopes[:, idx].flatten()
            ax.scatter(x, y, alpha=0.7, label="current well")

            if model_pred_data_name:
                # show envelope of model fits for all wells with this sample
                sample = sample_names[cid]
                same_idxs = [i for i, s in enumerate(samples) if s == sample]
                y_all = preds[:, same_idxs]               # (n_conc, n_same)
                y_min = np.nanmin(y_all, axis=1)
                y_max = np.nanmax(y_all, axis=1)
                ax.fill_between(x, y_min, y_max, color="gray", alpha=0.3, label='other well fits')
                # then overplot this chamber’s model fit
                y_p = preds[:, idx].flatten()
                ax.plot(x, y_p, color="red", label="current well fit")

            if model_fit_name:
                # extract this chamber's fit row
                chamb_idx = np.where(chambers == cid)[0][0]
                vals = fit_vals[chamb_idx]
                txt = "".join(f"{nm}={v:.2f}\n" for nm,v in zip(fit_types, vals))
                ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                        va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            
            if model_pred_data_name or model_fit_name:
                ax.legend()
            ax.set_title(f"{cid}: {sample_names[cid]}")
            ax.set_xlabel("Concentration")
            ax.set_ylabel("Initial Rate")
            if x_log: ax.set_xscale("log")
            if y_log: ax.set_yscale("log")
            return ax

        plot_chip(ic50s_to_plot, sample_names,
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
        #plotting variable: We'll plot by enzyme concentration. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
        analysis_data = self.get_run(analysis_name)     # Analysis data (to show slopes/intercepts)
        
        plotting_var = 'concentration'

        # Verify we have the variable:
        if plotting_var not in analysis_data.dep_var_type:
            raise ValueError(f"'{plotting_var}' not found in analysis data. Available variables: {analysis_data.dep_vars_types}")
        else:
            plotting_var_index = analysis_data.dep_var_type.index(plotting_var)

        concentration = analysis_data.dep_var[..., plotting_var_index]  # (n_chambers, n_conc, 1)
       
        #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
        #chamber_names_dict = self._db_conn.get_chamber_name_dict()
        chamber_names = analysis_data.indep_vars.chamber_IDs # (n_chambers,)
        sample_names =  analysis_data.indep_vars.sample_IDs # (n_chambers,)

        # Create dictionary mapping chamber_id -> sample_name:
        sample_names_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            sample_names_dict[chamber_id] = sample_names[i]

        # Create dictionary mapping chamber_id -> mean slopes:
        conc_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            conc_dict[chamber_id] = np.nanmean(concentration[i])
        
        plot_chip(conc_dict, sample_names_dict, title=f'Enzyme Concentration ({units})')

    def plot_mask_chip(self, mask_name: str):
        '''
        Plot a full chip with raw data and fit initial rates.

        Parameters:
            mask_name (str): the name of the mask to be plotted. (Data3D or Data2D)

        Returns:
            None
        '''
        #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
        mask_data = self.get_run(mask_name)     # Analysis data (to show slopes/intercepts)
        
        dtype = type(mask_data)
        assert dtype in [Data3D, Data2D], "mask_data must be of type Data3D or Data2D."

        mask_idx = mask_data.dep_var_type.index('mask')

        # If we're using a data2D
        mask = mask_data.dep_var[..., mask_idx] # (n_conc, n_chambers,)

        # We want to plot the number of concentrations that pass the mask in each well.
        # so, we'll sum across the concentration dimension, leaving an (n_chambers,) array

        #passed_conc = np.sum(mask, axis=0)
        #print(mask.shape)

        #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
        chamber_names = mask_data.indep_vars.chamber_IDs # (n_chambers,)
        sample_names =  mask_data.indep_vars.sample_IDs # (n_chambers,)

        # Create dictionary mapping chamber_id -> sample_name:
        sample_names_dict = {}
        for i, chamber_id in enumerate(chamber_names):
            sample_names_dict[chamber_id] = sample_names[i]

        # Create dictionary mapping chamber_id -> mean slopes:
        mask_sum = {}
        for i, chamber_id in enumerate(chamber_names):
            if dtype == Data3D:
                mask_sum[chamber_id] = mask[:, i].sum()  # sum across concentrations for each chamber
            elif dtype == Data2D:
                mask_sum[chamber_id] = mask[i].sum()

        #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
        # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.
        
        plot_chip(mask_sum, sample_names_dict, title=f'# Concentrations that pass filter: {mask_name}')

    ### DATA EXPORT ###
    def export_run_data_raw(self, run_name: str, output_dir: str = None):
        '''
        Export per-chamber data to a CSV file, for Data2D objects.
        Input:
            run_name (str): the name of the run to be exported.
            output_dir (str): the directory to save the CSV file. If None, uses current directory.
        Output:
            None
        '''
        run_data = self.get_run(run_name)

        assert isinstance(run_data, Data2D), "run_data must be of type Data2D."

        if output_dir is None:
            output_dir = os.getcwd()
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Define the output file path
        output_file = Path(output_dir) / f"{run_name}_data.csv"

        chamber_IDs = run_data.indep_vars.chamber_IDs
        sample_IDs = run_data.indep_vars.sample_IDs
        
        # Save the data to a CSV file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['chamber_ID', 'sample_ID'] + run_data.dep_var_type
            writer.writerow(header)
            for i in range(run_data.dep_var.shape[0]):
                row = [chamber_IDs[i], sample_IDs[i]] + run_data.dep_var[i].tolist()
                writer.writerow(row)
        
        print(f"Run data exported to {output_file}")

    def export_run_data_processed(self, run_name: str, output_dir: str = None):
        '''
        Export per-sample data to a CSV file, for Data2D objects.
        Input:
            run_name (str): the name of the run to be exported.
            output_dir (str): the directory to save the CSV file. If None, uses current directory.
        Output:
            None
        '''
        run_data = self.get_run(run_name)
        assert isinstance(run_data, Data2D), "run_data must be of type Data2D."
        
        sample_IDs = run_data.indep_vars.sample_IDs
        chamber_IDs = run_data.indep_vars.chamber_IDs

        # get the mean, std, and count for each value across each sample
        sample_list = np.unique(sample_IDs)
        sample_data = {sample: [] for sample in sample_list}
        for i, sample in enumerate(sample_list):
            sample_mask = (sample_IDs == sample)
            sample_data[sample] = {
                'chamber_IDs': chamber_IDs[sample_mask],
                'mean': np.nanmean(run_data.dep_var[sample_mask], axis=0),
                'std': np.nanstd(run_data.dep_var[sample_mask], axis=0),
                'count': np.sum(~np.isnan(run_data.dep_var[sample_mask]), axis=0)
            }

        if output_dir is None:
            output_dir = os.getcwd()
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Define the output file path
        output_file = Path(output_dir) / f"{run_name}_processed_data.csv"
        # Save the data to a CSV file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['sample_ID', 'chamber_IDs'] + run_data.dep_var_type
            writer.writerow(header)
            for sample, data in sample_data.items():
                row = [sample, ','.join(data['chamber_IDs'].tolist())] + data['mean'].tolist()
                writer.writerow(row)
        print(f"Processed run data exported to {output_file}")