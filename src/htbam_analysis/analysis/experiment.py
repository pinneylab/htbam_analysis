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
#import scipy

# HTBAM
from htbam_db_api.htbam_db_api import AbstractHtbamDBAPI, HtbamDBException
from htbam_analysis.analysis.plotting import plot_chip
from htbam_analysis.analysis.fitting import fit_luminance_vs_time, fit_luminance_vs_concentration

# Need these?
#from sklearn.linear_model import LinearRegression
#import inspect
import seaborn as sns

#from scipy.optimize import curve_fit

#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages


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

    ### PLOTTING ###
    def plot_standard_curve_chip(self, analysis_name: str, experiment_name):
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

    def plot_initial_rates_chip(self, analysis_name: str, experiment_name, skip_start_timepoint: bool = True):
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

        #slopes_to_plot = self._db_conn.get_analysis(run_name, 'linear_regression', 'slope')
        #intercepts_to_plot = self._db_conn.get_analysis(run_name, 'linear_regression', 'intercept')
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
            slopes_dict[chamber_id] = np.mean(slopes_to_plot[:, i])

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


    ######################
    ### N.F: Everything under here, I'm not using as of 5/16/25 in my refactor (will probably use some of it later)
    ######################

    # def get_concentration_units(self, run_name: str) -> str:
    #     '''
    #     Returns the units of concentration for the given run.
    #     Input:
    #         run_name (str): the name of the run to be analyzed.
    #     Output:
    #         str: the units of concentration for the given run.
    #     '''
    #     return self._db_conn.get_concentration_units(run_name)

    # def fit_standard_curve(self, run_name: str, low_conc_idx: int = 0, high_conc_idx: int = None):
    #     '''
    #     Fits a standard curve to the data in the given run.
    #     Input:
    #         run_name (str): the name of the run to be analyzed.
    #     Output:
    #         None
    #     '''
    #     if run_name not in self._run_data.keys():
    #         print("Existing run data not found. Fetching from database.")
    #         chamber_idxs, luminance_data, conc_data, _ = self._db_conn.get_run_assay_data(run_name)
    #         self._run_data[run_name] = {'chamber_idxs': chamber_idxs, 'luminance_data': luminance_data, 'conc_data': conc_data}

    #     print(f"Standard curve data found for run \"{run_name}\" with:")
    #     luminance_shape = self._run_data[run_name]['luminance_data'].shape
    #     print(f"\t-- {luminance_shape[0]} time points.\n\t-- {luminance_shape[1]} chambers.\n\t-- {luminance_shape[2]} concentrations.")
        
    #     if high_conc_idx is None:
    #         high_conc_idx = len(self._run_data[run_name]["conc_data"])

    #     print("\nFitting standard curve...")
    #     for i, idx in tqdm(list(enumerate(self._run_data[run_name]["chamber_idxs"]))):
    #         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(self._run_data[run_name]["conc_data"][low_conc_idx:high_conc_idx], 
    #                                                                              self._run_data[run_name]["luminance_data"][:,i][:,low_conc_idx:high_conc_idx])
    #         results_dict = {'slope': slope, 'intercept': intercept, 'r_value': r_value, 'r2':r_value**2, 'p_value': p_value, 'std_err': std_err}
    #         self._db_conn.add_analysis(run_name, 'linear_regression', idx,results_dict)

    # def fit_initial_rates(self, run_name: str, 
    #                       standard_run_name: str, 
    #                       substrate_conc: float = None,
    #                       max_rxn_perc: int = 15,
    #                       starting_timepoint_index: int = 0,
    #                       max_rxn_time = np.inf):
    #     '''
    #     Fits a standard curve to the data in the given run.
    #     Input:
    #         run_name (str): the name of the run to be analyzed.
    #     Output:
    #         None
    #     '''
    #     if run_name not in self._run_data.keys():
    #         print("Existing run data not found. Fetching from database.")
    #         chamber_idxs, luminance_data, conc_data, time_data = self._db_conn.get_run_assay_data(run_name)
    #         self._run_data[run_name] = {'chamber_idxs': chamber_idxs, 
    #                                     'luminance_data': luminance_data, 
    #                                     'conc_data': conc_data, 
    #                                     'time_data': time_data}

    #     print(f"Activity data found for run \"{run_name}\" with:")
    #     luminance_shape = self._run_data[run_name]['luminance_data'].shape
    #     print(f"\t-- {luminance_shape[0]} time points.\n\t-- {luminance_shape[1]} chambers.\n\t-- {luminance_shape[2]} concentrations.")

    #     if standard_run_name not in self._run_data.keys():
    #         raise ValueError(f"Standard curve data not found for run \"{standard_run_name}\". Please fit the standard curve first.")
        
    #     print(f"Using standard curve data from run \"{standard_run_name}\" to convert luminance data to concentration data.")

    #     std_slopes = np.array(list(self._db_conn.get_analysis(standard_run_name, 'linear_regression', 'slope').values()))

    #     #calculate product concentration by dividing every chamber intensity by the slope of the standard curve for that chamber
    #     self._run_data[run_name]["product_concentration"] = self._run_data[run_name]["luminance_data"] / std_slopes[np.newaxis, :, np.newaxis]

    #     chamber_dim = len(self._run_data[run_name]["chamber_idxs"])
    #     conc_dim = len(self._run_data[run_name]["conc_data"])
    #     time_dim = len(self._run_data[run_name]["time_data"])

    #     arr = np.empty((chamber_dim, conc_dim))
    #     arr[:] = np.nan
            
    #     #make an array of initial slopes for each chamber: should be (#chambers , #concentrations) = (1792 x 11)
    #     # initial_slopes = arr.copy()
    #     # initial_slopes_R2 = arr.copy()
    #     # initial_slopes_intercepts = arr.copy()
    #     # reg_idx_arr = np.zeros((chamber_dim, conc_dim, time_dim)).astype(bool)

    #     time_series = self._run_data[run_name]["time_data"]#[:,0][:, np.newaxis]
    #     if substrate_conc is None:
    #         substrate_concs = self._run_data[run_name]["conc_data"]
    #     else:
    #         substrate_concs = np.array([substrate_conc for _ in range(conc_dim)])
    #     #print(substrate_concs)
    #     product_thresholds = substrate_concs * max_rxn_perc / 100
    #     product_thresholds = product_thresholds[:, np.newaxis]
    #     two_point_fits = 0 
    #     for i, chamber_idx in tqdm(list(enumerate(self._run_data[run_name]["chamber_idxs"]))):

    #         #use the kinetics package to calculate the slopes for this chamber at each substrate concentration.
    #         product_conc_array = self._run_data[run_name]["product_concentration"][:,i,:].T

    #         # which product concentration is below the threshold?
    #         zeroed_product_conc_array = deepcopy(product_conc_array)
          
    #         zeroed_product_conc_array = zeroed_product_conc_array - zeroed_product_conc_array[:,starting_timepoint_index][:, np.newaxis]
            
    #         rxn_threshold_mask = zeroed_product_conc_array < product_thresholds
            
    #         time_allowed_mask = time_series < max_rxn_time
            
    #         rxn_threshold_mask[:,:starting_timepoint_index] = 0

    #         rxn_threshold_mask = np.logical_and(rxn_threshold_mask, time_allowed_mask.T)

    #         slopes = np.zeros_like(self._run_data[run_name]["conc_data"])
    #         intercepts = np.zeros_like(self._run_data[run_name]["conc_data"])
    #         scores = np.zeros_like(self._run_data[run_name]["conc_data"])

    #         for j, mask in enumerate(rxn_threshold_mask):
    #             if mask.sum() < 2:
    #                 two_point_fits += 1
    #                 pass
    #                 #print(f'Chamber {chamber_idx} Concentration {conc_data[i]} uM has less than 2 points')
    #             else:
    #                 lin_reg = LinearRegression()
    #                 lin_reg.fit(time_series[:,j][mask].reshape(-1,1), product_conc_array[j,:][mask])
    #                 #print(time_series[mask], product_conc_array[i,:][mask])
    #                 slope, intercept, score = lin_reg.coef_, lin_reg.intercept_, lin_reg.score(time_series[:,j][mask].reshape(-1,1), product_conc_array[j,:][mask])
    #                 #print(f'Concentration {conc_data[i]} uM has slope {slope} and intercept {intercept} with R2 {score}')
    #                 slopes[j] = slope
    #                 intercepts[j] = intercept
    #                 scores[j] = score
    #         results_dict = {'slopes': slopes, 'intercepts': intercepts, 'r_values': scores,  'mask': rxn_threshold_mask}
            
    #         self._db_conn.add_analysis(run_name, 'linear_regression', chamber_idx, results_dict)
    #         # initial_slopes[i] = slopes
    #         # initial_slopes_intercepts[i] = intercepts
    #         # initial_slopes_R2[i] = scores
    #         # reg_idx_arr[i] = rxn_threshold_mask
    #     print(f'{two_point_fits} reactions had less than 2 points for fitting')
    
    # def plot_progress_curves(self, run_name: str, chamber_idx: str, export_path: str = None):
        
    #     idx = list(self._run_data[run_name]["chamber_idxs"]).index(chamber_idx) 
    #     print(idx)
    #     name = self._db_conn.get_chamber_name_dict()[chamber_idx]
    #     prod_conc = self._run_data[run_name]["product_concentration"][:,idx,:].T
    #     time_data = self._run_data[run_name]["time_data"].T

    #     if export_path is not None:
    #         df = pd.DataFrame()
    #         for i, conc in enumerate(self._run_data[run_name]["conc_data"]):
    #             df[f'{conc}_time'] = time_data[i]
    #             df[f'{conc}_conc'] = prod_conc[i]
    #         df.to_csv(export_path, index=False)
    #     fig, ax = plt.subplots()
    #     for i in range(prod_conc.shape[0]):
    #         ax.scatter(time_data[i], prod_conc[i], label=f'{self._run_data[run_name]["conc_data"][i]}')
    #     ax.legend()
    #     ax.set_title(f'Progress Curves for {chamber_idx}: {name}')
    #     ax.set_xlabel('Time (s)')
    #     ax.set_ylabel('Product Concentration (uM)')
        
    # def subtract_background_rate(self, run_name: str, background_pattern: str):
    #     chamber_name_to_id_dict = self._db_conn.get_chamber_name_to_id_dict()
    #     names = []
    #     for key in chamber_name_to_id_dict.keys():
    #         if re.match(background_pattern, key):
    #             names.append(key)
    #     ids = []
    #     for name in names:
    #         ids.extend(chamber_name_to_id_dict[name])

    #     rates = []
    #     for id in ids:
    #         initial_rates = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')[id]
    #         rates.append(initial_rates)
    #     background_rate_arr = np.array(rates).T
    #     #print(background_rate_arr.shape)
    #     #print(np.median(background_rate_arr, axis=1))
    #     # lower quartile

    #     lower_quartile = np.percentile(background_rate_arr, 25, axis=1)
    #     print(lower_quartile)
    #     initial_rates_dict = deepcopy(self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes'))
    #     for key in initial_rates_dict.keys():
    #         initial_rates_dict[key] = initial_rates_dict[key] - lower_quartile
    #         self._db_conn.add_analysis(run_name, 'bgsub_linear_regression', key, {"slopes": initial_rates_dict[key]})
    #     return

    # def plot_initial_rates_chip(self, run_name: str, time_to_plot=0.3, subtract_zeroth_point=False):

    #     initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')

    #     initial_rate_intercepts_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'intercepts')
    #     initial_rate_masks_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'mask') 
    #     intial_rate_arr = np.array(list(initial_rates_dict.values()))
    #     initial_rate_intercepts_arr = np.array(list(initial_rate_intercepts_dict.values()))
    #     initial_rate_masks_arr = np.array(list(initial_rate_masks_dict.values()))

    #     initial_rates_to_plot = {i: np.nanmax(j) for i, j in initial_rates_dict.items()}


    #     chamber_names_dict = self._db_conn.get_chamber_name_dict()


    #     #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
    #     # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.
    #     def plot_chamber_initial_rates(chamber_id, ax, time_to_plot=time_to_plot):
    #         #N.B. Every so often, slope and line colors don't match up. Not sure why.
            
    #         #convert from 'x,y' to integer index in the array:
    #         data_index = list(self._run_data[run_name]["chamber_idxs"]).index(chamber_id)
    #         x_data = self._run_data[run_name]["time_data"][:,0]
    #         y_data = self._run_data[run_name]["product_concentration"][:,data_index,:].T
            
    #         #plot only first X% of time:
    #         max_time = np.nanmax(x_data)
    #         time_to_plot = max_time*time_to_plot
    #         time_idxs_to_plot = x_data < time_to_plot
    #         x_data = x_data[time_idxs_to_plot]
    #         y_data = y_data[:, time_idxs_to_plot]
            
    #         # TODO: add option to subtract zeroth point(s) from all points
    #         #get slope from the analysis:
    #         current_chamber_slopes = intial_rate_arr[data_index,:]
    #         #calculate y-intercept by making sure it intersects first point:
    #         current_chamber_intercepts = initial_rate_intercepts_arr[data_index,:]
    #         # get regressed point mask:
    #         current_chamber_reg_mask = initial_rate_masks_arr[data_index,:][:,:len(x_data)]
            
    #         colors = sns.color_palette('husl', n_colors=y_data.shape[0])

    #         for i in range(y_data.shape[0]): #over each concentration:
                
    #             if subtract_zeroth_point:
    #                 try:
    #                     ax.scatter(x_data, y_data[i,:] - y_data[i,0], color=colors[i], alpha=0.3)
    #                     ax.scatter(x_data[current_chamber_reg_mask[i]],
    #                                y_data[i, current_chamber_reg_mask[i]] - y_data[i, current_chamber_reg_mask[i]][0], color=colors[i], alpha=1, s=50, label=f'{self._run_data[run_name]["conc_data"][i]}')
    #                 except:
    #                     pass
    #             else:
    #                 ax.scatter(x_data, y_data[i,:], color=colors[i], alpha=0.3)
    #                 ax.scatter(x_data[current_chamber_reg_mask[i]], y_data[i, current_chamber_reg_mask[i]], color=colors[i], alpha=1, s=50, label=f'{self._run_data[run_name]["conc_data"][i]}')

    #             m = current_chamber_slopes[i]
    #             b = current_chamber_intercepts[i] if not subtract_zeroth_point else 0
    #             if not (np.isnan(m) or np.isnan(b)):
    #                 #return False, no_update, no_update
    #                 ax.plot(x_data, m*np.array(x_data) + b, color=colors[i])
    #         ax.legend()
    #         return ax

    #     ### PLOT THE CHIP: now, we plot
    #     plot_chip(initial_rates_to_plot, chamber_names_dict, graphing_function=plot_chamber_initial_rates, title='Kinetics: Initial Rates (Max)')

    # def compute_enzyme_concentration(self, run_name: str, egfp_slope):
    #     #make numpy array of all button_quants with[ subtracted backgrounds:
    #     button_quant_no_background = [] #we will soon turn this into a numpy array
    #     for chamber_idx in self._run_data[run_name]['chamber_idxs']:
    #         next_button_quant = self._db_conn.get_button_quant_data(chamber_idx)
    #         button_quant_no_background.append(next_button_quant)
    #     button_quant_no_background = np.array(button_quant_no_background)

    #     # use eGFP standard curve to convert between button quant and eGFP concentration
    #     self._run_data[run_name]["enzyme_concentration"] = button_quant_no_background / egfp_slope    #in units of EGFP_SLOPE_CONC_UNITS

    # def filter_initial_rates(self,
    #                         kinetic_run_name: str, 
    #                         standard_run_name: str,
    #                         standard_curve_r2_cutoff: float = 0.98,
    #                         expression_threshold: float = 1.0,
    #                         initial_rate_R2_threshold: float = 0.0, 
    #                         positive_initial_slope_filter: bool = True,
    #                         multiple_exposures: bool = False,
    #                         background_subtraction: bool = False,):
        
    #     initial_slopes = np.array(list(self._db_conn.get_analysis(kinetic_run_name, 'bgsub_linear_regression' if background_subtraction else 'linear_regression', 'slopes').values()))
    #     enzyme_concentration = self._run_data[kinetic_run_name]["enzyme_concentration"]
    #     chamber_count = len(self._run_data[kinetic_run_name]["chamber_idxs"])
    #     conc_count = len(self._run_data[kinetic_run_name]["conc_data"])
    #     standard_r2_values = np.array(list(self._db_conn.get_analysis(standard_run_name, 'linear_regression', 'r2').values()))
    #     initial_rate_r2_values = np.array(list(self._db_conn.get_analysis(kinetic_run_name, 'linear_regression', 'r_values').values()))
        
    #     #print(initial_slopes)
    #     ### Make filters ###
    #     filters = []
    #     filter_r2 = np.ones_like(initial_slopes)
    #    # print(filter_r2)

    #     # STANDARD CURVE FILTER #
    #     # overwrite all chambers (rows) with r^2 values below the threshold with NaNs:
    #     _count = 0
    #     for i in range(len(self._run_data[kinetic_run_name]["chamber_idxs"])):
    #         if standard_r2_values[i] < standard_curve_r2_cutoff:
    #             _count +=1
    #             filter_r2[i, :] = np.nan
    #     print('Pearson r^2 filter: {}/{} chambers pass'.format(chamber_count-_count, chamber_count))
    #     filter_names = ['filter_r2']
    #     filter_values = [standard_curve_r2_cutoff]
    #     filters.append(filter_r2)

    #     # ENZYME EXPRESSION FILTER #
    #     # overwrite all chambers (rows) with enzyme expression below the threshold with NaNs:
    #     #Double check the expression units match the EGFP units:
    #     #assert expression_threshhold_units == EGFP_SLOPE_CONC_UNITS, 'Error, enzyme expression and EGFP standard curve units do not match!'
        
    #     filter_enzyme_expression = np.ones_like(initial_slopes)
    #     _count = 0
    #     for i in range(len(self._run_data[kinetic_run_name]["chamber_idxs"])):
    #         if enzyme_concentration[i] < expression_threshold:
    #             _count +=1
    #             filter_enzyme_expression[i,:] = np.nan
    #     print('Enzyme expression filter: {}/{} chambers pass'.format(chamber_count-_count, chamber_count))
    #     filters.append(filter_enzyme_expression)
    #     filter_names.append('filter_enzyme_expression')
    #     filter_values.append(expression_threshold)
    #     #TODO: track units! 

    #     # INITIAL RATE FIT FILTER #
    #     # overwrite just the assays per chamber (single values) with initial rate fit R^2 values below the threshold with NaNs:
    #     filter_initial_rate_R2 = np.ones_like(initial_slopes)
    #     _count = 0
    #     for i in range(len(self._run_data[kinetic_run_name]["chamber_idxs"])):            
    #         _chamber_count = 0
    #         for j in range(conc_count):
    #             if initial_rate_r2_values[i,j] < initial_rate_R2_threshold:
    #                 _chamber_count +=1
    #                 filter_initial_rate_R2[i,j] = np.nan
    #         if conc_count - _chamber_count < 7:
    #             _count +=1
    #     # TODO This shouldnt be base on sume hard coded cutoff, we should figure out a more flexible way of setting the min number of chambers
    #     # but for now, we'll just use 7 as a cutoff
    #     print('Initial Rate R^2 filter: {}/{} chambers pass with 10 or more slopes. NOTE: this doesnt mean chambers with <10 are excluded. Will make more clear in the future.'.format(chamber_count-_count, chamber_count))
    #     filters.append(filter_initial_rate_R2)
    #     filter_names.append('filter_initial_rate_R2')
    #     filter_values.append(initial_rate_R2_threshold)

    #     if positive_initial_slope_filter:
    #         # POSITIVE INITIAL SLOPE FILTER #
    #         # overwrite just the assays per chamber (single values) with initial slopes below zero with NaNs:
    #         filter_positive_initial_slope = np.ones_like(initial_slopes)
    #         _count = 0
    #         for i in range(len(self._run_data[kinetic_run_name]["chamber_idxs"])):
    #             _chamber_count = 0
    #             for j in range(conc_count):
    #                 if initial_slopes[i,j] < 0:
    #                     _chamber_count +=1
    #                     filter_positive_initial_slope[i,j] = np.nan
    #             if conc_count - _chamber_count < 7:
    #                 _count +=1
    #         print('Positive Initial Slope filter: {}/{} chambers pass with 10 or more slopes. NOTE: this doesnt mean chambers with <10 are excluded. Will make more clear in the future.'.format(chamber_count-_count, chamber_count))
    #         filters.append(filter_positive_initial_slope)
    #     filter_names.append('filter_positive_initial_slope')
    #     filter_values.append(positive_initial_slope_filter)

    #     ### manually flagged wells ###
    #     #TODO: implement

    #     ### TODO: make visualization!
    #     # chamber_idxs, luminance_data, conc_data, time_data

    #     filtered_initial_slopes = deepcopy(initial_slopes)
    #     for filter in filters: filtered_initial_slopes *= filter

    #     assay_dict = {"filters": filters}
    #     assay_dict.update({filter_names[i]: filter_values[i] for i in range(len(filter_names))})

    #     # #initialize the dictionary
       
    #     assay_data = {}
    #     for i in range(conc_count):
    #         assay_data[i] = {
    #             'substrate_conc': self._run_data[kinetic_run_name]["conc_data"][i],
    #             'chambers': {}
    #         }
    #         for j, chamber_idx in enumerate(self._run_data[kinetic_run_name]["chamber_idxs"]):
    #             assay_data[i]['chambers'][chamber_idx] = {
    #                 'slope': filtered_initial_slopes[j,i],
    #                 'r2': initial_rate_r2_values[j,i]
    #             }

    #     assay_dict["assays"] = assay_data
    #     self._db_conn.add_filtered_assay(kinetic_run_name, 'filtered_initial_rates', assay_dict)

    # def plot_filtered_initial_rates_chip(self, run_name: str, time_to_plot=0.3):
    #     ###N.B.: May be some bug here, because some of the filtered-out chambers are still showing slopes.
    #     # I think they should have all nans...?

    #     initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')
    #     initial_rate_intercepts_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'intercepts')
    #     initial_rate_masks_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'mask') 
    #     initial_rate_arr = np.array(list(initial_rates_dict.values()))
    #     initial_rate_intercepts_arr = np.array(list(initial_rate_intercepts_dict.values()))
    #     initial_rate_masks_arr = np.array(list(initial_rate_masks_dict.values()))


    #     filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
    #     filtered_initial_slopes = deepcopy(initial_rate_arr)
    #     for filter in filters: filtered_initial_slopes *= filter
        
    #     chamber_names_dict = self._db_conn.get_chamber_name_dict()

    #     #Let's plot as before:
    #     #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
    #     # TODO: I don't think this is what freitas intended, but will probably work for now. revisit.
    #     filtered_initial_rates_to_plot = {i: np.nanmax(j) for i, j in initial_rates_dict.items()}


    #     #chamber_names: Same as before.

    #     #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
    #     # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.

    #     def plot_chamber_filtered_initial_rates(chamber_id, ax, time_to_plot=time_to_plot):
    #         #N.B. Every so often, slope and line colors don't match up. Not sure why.
    #         #parameters: what amount of total time to plot? First 20%?
        
    #         #convert from 'x,y' to integer index in the array:
    #         data_index = list(self._run_data[run_name]["chamber_idxs"]).index(chamber_id)
    #         x_data = self._run_data[run_name]["time_data"][:,0]
    #         y_data = self._run_data[run_name]["product_concentration"][:,data_index,:].T
            
    #         #plot only first X% of time:
    #         max_time = np.nanmax(x_data)
    #         time_to_plot = max_time*time_to_plot
    #         time_idxs_to_plot = x_data < time_to_plot
    #         x_data = x_data[time_idxs_to_plot]
    #         y_data = y_data[:, time_idxs_to_plot]
            
    #         #get slope from the analysis:
    #         current_chamber_slopes = filtered_initial_slopes[data_index,:]
    #         #calculate y-intercept by making sure it intersects first point:
    #         current_chamber_intercepts = initial_rate_intercepts_arr[data_index,:]
    #         # get regressed point mask:
    #         current_chamber_reg_mask = initial_rate_masks_arr[data_index,:][:,:len(x_data)]
            
    #         colors = sns.color_palette('husl', n_colors=y_data.shape[0])

    #         #print(y_data.shape[0])
    #         for i in range(y_data.shape[0]): #over each concentration:
                
    #             ax.scatter(x_data, y_data[i,:], color=colors[i], alpha=0.3)
    #             ax.scatter(x_data[current_chamber_reg_mask[i]], y_data[i, current_chamber_reg_mask[i]], color=colors[i], alpha=1, s=50)
                
    #             m = current_chamber_slopes[i]
    #             b = current_chamber_intercepts[i]
    #             if not (np.isnan(m) or np.isnan(b)):
    #                 #return False, no_update, no_update
    #                 ax.plot(x_data, m*np.array(x_data) + b, color=colors[i])
    #         return ax

            

    #     ### PLOT THE CHIP: now, we plot
    #     plot_chip(filtered_initial_rates_to_plot, chamber_names_dict, graphing_function=plot_chamber_filtered_initial_rates, title='Kinetics: Filtered Initial Rates (Max)')
    #     print('{}/1792 wells pass our filters.'.format( 
    #         np.sum([np.any(~np.isnan(filtered_initial_slopes[i,:])) for i in range(len(self._run_data[run_name]["chamber_idxs"]))]) ) )

    # def combine_filtered_assays(self, run_names_to_combine: List[str]):
    #     new_run_name = self._db_conn.combine_runs(run_names_to_combine)
    #     if new_run_name not in self._run_data.keys():
    #         self._run_data[new_run_name] = {"conc_data": None}
    #     for run_name in run_names_to_combine:
    #         if self._run_data[new_run_name]["conc_data"] is None:
    #            self._run_data[new_run_name]["conc_data"] = self._run_data[run_name]["conc_data"]
    #         else:
    #             self._run_data[new_run_name]["conc_data"] = np.concatenate((self._run_data[new_run_name]["conc_data"], self._run_data[run_name]["conc_data"]))

    #     self._run_data[new_run_name]["enzyme_concentration"] = self._run_data[run_names_to_combine[0]]["enzyme_concentration"]
    #     self._run_data[new_run_name]["chamber_idxs"] = self._run_data[run_names_to_combine[0]]["chamber_idxs"] 
    #     print(f"New run \"{new_run_name}\" created by combining runs {run_names_to_combine}.")

    # def remove_run(self, run_name: str):
    #     if run_name in self._run_data.keys():
    #         del self._run_data[run_name]
    #     self._db_conn.remove_run(run_name)
        
    # def fit_ic50s(self, run_name: str, inhibition_model = None):

    #     if inhibition_model is None:
    #         # default model
    #         def inhibition_model(x, r_max, r_min, ic50):
    #             return r_min + (r_max-r_min)/(1+(x/ic50))
        
    #     arg_list = str(inspect.signature(inhibition_model)).strip("()").replace(" ", "").split(",")

    #     # get data
    #     initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')
    #     initial_rate_arr = np.array(list(initial_rates_dict.values()))
    #     chamber_idxs = list(initial_rates_dict.keys())
      
    #     filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
    #     filtered_initial_slopes = deepcopy(initial_rate_arr)
    #     for filter in filters: filtered_initial_slopes *= filter

    #     #Here, we calculate the IC50 for each chamber.
    #     ic50_array = np.array([])
    #     ic50_error_array = np.array([])
    #     fit_params = []
    #     fit_params_errs = []
    #     _count = 0
    #     for i in range(len(chamber_idxs)):
    #         current_slopes = filtered_initial_slopes[i, :]

    #         if np.all(np.isnan(current_slopes)) or np.all(current_slopes == 0) or np.nanmax(current_slopes) == 0:
    #             #print('Chamber {} has no slopes!'.format(chamber_idxs[i]))
    #             ic50_array = np.append(ic50_array, np.nan)
    #             ic50_error_array = np.append(ic50_error_array, np.nan)
    #             fit_params.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             fit_params_errs.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             _count += 1
    #             continue

    #         #get indices of non-nan values:
    #         non_nan_idxs = np.where(~np.isnan(current_slopes))[0]
            
    #         current_slopes = current_slopes[non_nan_idxs]
    #         current_concs = self._run_data[run_name]["conc_data"][non_nan_idxs]

    #         if len(current_slopes) < 3:
    #             #print('Chamber {} has fewer than 3 slopes!'.format(chamber_idxs[i]))
    #             ic50_array = np.append(ic50_array, np.nan)
    #             ic50_error_array = np.append(ic50_error_array, np.nan)
    #             fit_params.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             fit_params_errs.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             _count += 1
    #             continue

    #         max_normed_slopes = current_slopes / np.nanmax(current_slopes)
    #         #kinetics.fit_and_plot_micheaelis_menten(current_slopes, current_slopes, current_concs, enzyme_concentration_converted_units[i], 'uM', 'MM for first chamber!')
    #         #K_i, std_err = kinetics.fit_inhibition_constant(max_normed_slopes, max_normed_slopes, current_concs, enzyme_concentration_converted_units[i], 'uM', 'MM for first chamber!')
            
    #         try:
    #             p_opt, p_cov = curve_fit(inhibition_model, current_concs, max_normed_slopes)
    #             param_dict = {i:j for i,j in zip(arg_list[1:], p_opt)}
    #             if "ic50" not in param_dict.keys():
    #                 raise ValueError("Inhibition model must have an 'ic50' parameter.")
                
    #             ic50 = param_dict["ic50"]
    #             fit_err = np.sqrt(np.diag(p_cov))
    #             param_error_dict = {i:j for i,j in zip(arg_list[1:], fit_err)}
            
    #             ic50_err = param_error_dict["ic50"]

    #             ic50_array = np.append(ic50_array, ic50)
    #             ic50_error_array = np.append(ic50_error_array, ic50_err)
    #             fit_params.append(param_dict)
    #             fit_params_errs.append(param_error_dict)
    #         except RuntimeError:
    #             ic50_array = np.append(ic50_array, np.nan)
    #             ic50_error_array = np.append(ic50_error_array, np.nan)
    #             fit_params.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             fit_params_errs.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             _count += 1


            
    #     # chamber_idxs, luminance_data, conc_data, time_data

    #     for i, chamber_idx in enumerate(chamber_idxs):
    #         self._db_conn.add_analysis(run_name, 'ic50_raw', chamber_idx,  {'ic50': ic50_array[i], 
    #                                                                         'ic50_error': ic50_error_array[i],
    #                                                                         'fit_params': fit_params[i],
    #                                                                         'fit_params_errs': fit_params_errs[i]} )
    #     print(f'{_count} chambers had fewer than 3 slopes and were not fit.')

    # def filter_ic50s(self, 
    #                  run_name,
    #                  z_score_threshold_ic50 = 1.5,
    #                  z_score_threshold_expression = 1.5,
    #                  save_intermediate_data = False):

    #     ic50_array = np.array(list(self._db_conn.get_analysis(run_name, 'ic50_raw', 'ic50').values()))
    #     enzyme_concentration = self._run_data[run_name]["enzyme_concentration"]

    #     #Get chamber ids from metadata:
       
    #     #Get chamber ids from metadata:
    #     chamber_names_dict = self._db_conn.get_chamber_name_to_id_dict()

    #     #Get average k_cat, k_M, and v_max for each sample:
    #     sample_names = np.array([])
    #     sample_ic50 = np.array([])
    #     sample_ic50_error = np.array([])
    #     sample_ic50_replicates = []

    #     # #Get z-scores for each well (used to filter in the next step!)
    #     # ic50_zscores = np.array([])
    #     # enzyme_concentration_zscores = np.array([])

    #     export_list1=[]
    #     #For each sample, 
    #     for name, ids in chamber_names_dict.items():
            

    #         ### GATHER MM PARAMETERS OF REPLICATES FOR EACH SAMPLE: ###
    #         #get indices of idxs in chamber_idxs:
    #         idxs = [list(self._run_data[run_name]["chamber_idxs"]).index(x) for x in ids]

    #         #get values for those indices:
    #         ic50s = ic50_array[idxs]

    #         #keep track of which wells we exclude later:
    #         ic50_replicates = np.array(ids)

    #         #if any of these is all nans, just continue to avoid errors:
    #         if np.all(np.isnan(ic50s)):
    #             print('No values from sample {}, all pre-filtered.'.format(name))
    #             continue

    #         ### FILTER OUT OUTLIERS: ###
    #         #calculate z-score for each value:
    #         ic50_zscore = (ic50s - np.nanmean(ic50s))/np.nanstd(ic50s)

    #         #also, get z-score of enzyme expression for each well:
    #         enzyme_concentration_zscore = (enzyme_concentration[idxs] - np.nanmean(enzyme_concentration[idxs]))/np.nanstd(enzyme_concentration[idxs]) #in units of 'substrate_conc_unit' 

    #         #First, for enzyme expression outliers, set the value to NaN to be filtered in the final step:
    #         ic50s[np.abs(enzyme_concentration_zscore) > z_score_threshold_expression] = np.nan

    #         #filter out values with z-score > threshhold:
    #         ic50s = ic50s[np.abs(ic50_zscore) < z_score_threshold_ic50]

    #         #do the same for the replicates ids:
    #         ic50_replicates = ic50_replicates[np.abs(ic50_zscore) < z_score_threshold_ic50]

    #         #remove nan values from all (nan values are due to both no experimental data, and z-score filtering)
    #         ic50_replicates = ic50_replicates[~np.isnan(ic50s)]
    #         ic50s = ic50s[~np.isnan(ic50s)]

    #         if len(ic50s) < 3:
    #             print('Not enough replicates for sample {}. Skipping.'.format(name))
    #             continue
            
    #         #get average values:
    #         sample_names = np.append(sample_names, name)
    #         sample_ic50 = np.append(sample_ic50, np.mean(ic50s))
    #         sample_ic50_error = np.append(sample_ic50_error,np.std(ic50s))
            
    #         #keep track of replicates:
    #         sample_ic50_replicates.append(ic50_replicates)

    #         if save_intermediate_data:
    #             temp_list1 = []
    #             temp_list1.append(name)
    #             for ic50 in ic50s:
    #                 temp_list1.append(ic50)
    #             export_list1.append(temp_list1)

    #     if save_intermediate_data:   
    #         df2 = pd.DataFrame(export_list1)
    #         df2.to_csv('ic50_file_intermediate.csv')      
    #     for i, sample_name in enumerate(sample_names):
    #         self._db_conn.add_sample_analysis(run_name, 'ic50_filtered', sample_name, {'ic50': sample_ic50[i], 'ic50_error': sample_ic50_error[i], 'ic50_replicates': sample_ic50_replicates[i]})
          
    #     print('Average number of replicates per sample post-filtering: {}'.format(int(np.round(np.mean([len(i) for i in sample_ic50_replicates]), 0))))


    # def fit_mm(self, run_name: str, enzyme_conc_conversion: float = 1.0, mm_model = None,
    #            background_subtraction: bool = False):

    #     if mm_model is None:
    #         # default model
    #         def mm_model(x, v_max, K_m):
    #             return v_max*x/(K_m + x)
        
    #     arg_list = str(inspect.signature(mm_model)).strip("()").replace(" ", "").split(",")

    #     # get data
    #     initial_rates_dict = self._db_conn.get_analysis(run_name, 'bgsub_linear_regression' if background_subtraction else 'linear_regression', 'slopes')
    #     initial_rate_arr = np.array(list(initial_rates_dict.values()))
    #     chamber_idxs = list(initial_rates_dict.keys())
      
    #     filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
    #     filtered_initial_slopes = deepcopy(initial_rate_arr)
    #     for filter in filters: filtered_initial_slopes *= filter

    #     enzyme_concentrations = deepcopy(self._run_data[run_name]["enzyme_concentration"])

    #     #Here, we calculate the IC50 for each chamber.
    #     kcat_array = np.array([])
    #     Km_array = np.array([])
    #     fit_params = []
    #     fit_params_errs = []
    #     _count = 0
    #     for i in range(len(chamber_idxs)):
    #         current_slopes = filtered_initial_slopes[i, :]

    #         if np.all(np.isnan(current_slopes)) or np.all(current_slopes == 0):
    #             #print('Chamber {} has no slopes!'.format(chamber_idxs[i]))
    #             kcat_array = np.append(kcat_array, np.nan)
    #             Km_array = np.append(Km_array, np.nan)
    #             fit_params.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             fit_params_errs.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             _count += 1
    #             continue

    #         #get indices of non-nan values:
    #         non_nan_idxs = np.where(~np.isnan(current_slopes))[0]
            
    #         current_slopes = current_slopes[non_nan_idxs]
    #         current_concs = self._run_data[run_name]["conc_data"][non_nan_idxs]

    #         # NOTE: this should be optimized
    #         if len(current_slopes) < 5:
    #             #print('Chamber {} has fewer than 3 slopes!'.format(chamber_idxs[i]))
    #             kcat_array = np.append(kcat_array, np.nan)
    #             Km_array = np.append(Km_array, np.nan)
    #             fit_params.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             fit_params_errs.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
    #             _count += 1
    #             continue

    #         #kinetics.fit_and_plot_micheaelis_menten(current_slopes, current_slopes, current_concs, enzyme_concentration_converted_units[i], 'uM', 'MM for first chamber!')
    #         #K_i, std_err = kinetics.fit_inhibition_constant(max_normed_slopes, max_normed_slopes, current_concs, enzyme_concentration_converted_units[i], 'uM', 'MM for first chamber!')
    #         p_opt, p_cov = curve_fit(mm_model, current_concs, current_slopes)
    #         param_dict = {i:j for i,j in zip(arg_list[1:], p_opt)}

    #         if ("v_max" not in param_dict.keys()) or ("K_m" not in param_dict.keys()):
    #             raise ValueError("MM model must have 'v_max' and 'K_m' parameters.")
            
    #         v_max = param_dict["v_max"]
    #         fit_err = np.sqrt(np.diag(p_cov))
    #         param_error_dict = {i:j for i,j in zip(arg_list[1:], fit_err)}

          
    #         kcat_array = np.append(kcat_array, v_max/ (enzyme_concentrations[i] * enzyme_conc_conversion))
            
    #         Km_array = np.append(Km_array, param_dict["K_m"])
    #         fit_params.append(param_dict)
    #         fit_params_errs.append(param_error_dict)
    #     # chamber_idxs, luminance_data, conc_data, time_data

    #     for i, chamber_idx in enumerate(chamber_idxs):
    #         self._db_conn.add_analysis(run_name, 'mm_raw', chamber_idx,  {'kcat': kcat_array[i], 
    #                                                                         'K_m': Km_array[i],
    #                                                                         'fit_params': fit_params[i],
    #                                                                         'fit_params_errs': fit_params_errs[i]} )
    #     print(f'{_count} chambers had fewer than 5 slopes and were not fit.')

    # def filter_mm(self, 
    #                 run_name,
    #                 z_score_threshold_mm = 1.5,
    #                 z_score_threshold_expression = 1.5,
    #                 km_fold_over_substrate = 5,
    #                 save_intermediate_data = False):

    #     try:
    #         self._db_conn.remove_analysis(run_name, 'mm_filtered')
    #         print('Removed old filtered MM data.')
    #     except HtbamDBException:
    #         print('No old filtered MM data to remove.')
    #     kcat_array = np.array(list(self._db_conn.get_analysis(run_name, 'mm_raw', 'kcat').values()))
    #     Km_array = np.array(list(self._db_conn.get_analysis(run_name, 'mm_raw', 'K_m').values()))
    #     enzyme_concentration = self._run_data[run_name]["enzyme_concentration"]
       
    #     #Get chamber ids from metadata:
    #     chamber_names_dict = self._db_conn.get_chamber_name_to_id_dict()

    #     #Get average k_cat, k_M, and v_max for each sample:
    #     sample_names = np.array([])
    #     sample_kcat = np.array([])
    #     sample_kcat_std = np.array([])
    #     sample_K_m = np.array([])
    #     sample_K_m_std = np.array([])
    #     sample_mm_replicates = []

    #     export_list1=[]

    #     conc_data = self._run_data[run_name]["conc_data"]
        
    #     #For each sample, 
    #     for name, ids in chamber_names_dict.items():
            
            

    #         ### GATHER MM PARAMETERS OF REPLICATES FOR EACH SAMPLE: ###
    #         #get indices of idxs in chamber_idxs:
    #         idxs = [list(self._run_data[run_name]["chamber_idxs"]).index(x) for x in ids]

    #         #get values for those indices:
    #         kcats = kcat_array[idxs]
    #         K_ms = Km_array[idxs]

    #         #keep track of which wells we exclude later:
    #         mm_replicates = np.array(ids)

    #         #if any of these is all nans, just continue to avoid errors:
    #         if np.all(np.isnan(kcats)) or np.all(np.isnan(K_ms)):
    #             print('No values from sample {}, all pre-filtered.'.format(name))
    #             continue

    #         ### FILTER OUT OUTLIERS: ###
    #         #calculate z-score for each value:
    #         kcat_zscore = (kcats - np.nanmean(kcats))/np.nanstd(kcats)
    #         K_m_zscore = (K_ms - np.nanmean(K_ms))/np.nanstd(K_ms)

    

    #         #also, get z-score of enzyme expression for each well:
    #         enzyme_concentration_zscore = (enzyme_concentration[idxs] - np.nanmean(enzyme_concentration[idxs]))/np.nanstd(enzyme_concentration[idxs]) #in units of 'substrate_conc_unit' 

    #         z_score_mask = np.logical_and(np.abs(enzyme_concentration_zscore) < z_score_threshold_expression, np.abs(kcat_zscore) < z_score_threshold_mm, np.abs(K_m_zscore) < z_score_threshold_mm)
            
    #         # Km ceiling mask
    #         z_score_mask = np.logical_and(z_score_mask, K_ms < (km_fold_over_substrate * max(conc_data)))
          

    #         #First, for enzyme expression outliers, set the value to NaN to be filtered in the final step:
    #         kcats[~z_score_mask] = np.nan
    #         K_ms[~z_score_mask] = np.nan
    #         mm_replicates[~z_score_mask] = np.nan
           

    #         #remove nan values from all (nan values are due to both no experimental data, and z-score filtering)
    #         nan_mask = np.logical_and(~np.isnan(kcats), ~np.isnan(K_ms))
    #         kcats = kcats[nan_mask]
    #         K_ms = K_ms[nan_mask]
    #         mm_replicates = mm_replicates[nan_mask]
          


    #         if len(kcats) < 3:
    #             print('Not enough replicates for sample {}. Skipping.'.format(name))
    #             continue
            
    #         #get average values:
    #         sample_names = np.append(sample_names, name)
    #         sample_kcat = np.append(sample_kcat, np.mean(kcats))
    #         sample_kcat_std = np.append(sample_kcat_std, np.std(kcats))
    #         sample_K_m = np.append(sample_K_m, np.mean(K_ms))
    #         sample_K_m_std = np.append(sample_K_m_std, np.std(K_ms))
            
    #         #keep track of replicates:
    #         sample_mm_replicates.append(mm_replicates)

    #         if save_intermediate_data:
    #             temp_list1 = []
    #             temp_list1.append(name)
    #             for kcat in kcats:
    #                 temp_list1.append(kcat)
    #             export_list1.append(temp_list1)

    #     if save_intermediate_data:   
    #         df2 = pd.DataFrame(export_list1)
    #         df2.to_csv('mm_file_intermediate.csv')      
    #     for i, sample_name in enumerate(sample_names):
          
    #         self._db_conn.add_sample_analysis(run_name, 'mm_filtered', sample_name, {'kcat': sample_kcat[i], 'kcat_std': sample_kcat_std[i],
    #                                                                                  'K_m': sample_K_m[i] , 'K_m_std': sample_K_m_std[i],
    #                                                                                  'mm_replicates': sample_mm_replicates[i]})
          
    #     print('Average number of replicates per sample post-filtering: {}'.format(int(np.round(np.mean([len(i) for i in sample_mm_replicates]), 0))))

    # def plot_filtered_ic50(self, run_name: str, inhibition_model = None, show_average_fit: bool = False):
        
    #     if inhibition_model is None:
    #         # default model
    #         def inhibition_model(x, r_max, r_min, ic50):
    #             return r_min + (r_max-r_min)/(1+(x/ic50))
            
    #     #first, fill it with NaNs as a placeholder:
    #     ic50_to_plot = {chamber_idx: np.nan for chamber_idx in self._run_data[run_name]["chamber_idxs"]}

    #     chamber_name_to_id_dict = self._db_conn.get_chamber_name_to_id_dict()

    #     initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')
    #     initial_rate_arr = np.array(list(initial_rates_dict.values()))
    #     chamber_idxs = list(initial_rates_dict.keys())
      
    #     filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
    #     filtered_initial_slopes = deepcopy(initial_rate_arr)
    #     for filter in filters: filtered_initial_slopes *= filter
    #     sample_ic50_dict = self._db_conn.get_sample_analysis_dict(run_name, 'ic50_filtered')
    #     #then, fill in the values we have:
    #     for name, values in sample_ic50_dict.items():
    #         for chamber_idx in chamber_name_to_id_dict[name]:

    #             ic50_to_plot[chamber_idx] = values['ic50']
        
    #     chamber_names_dict = self._db_conn.get_chamber_name_dict()

    #     #plotting function: We'll generate an MM subplot for each chamber.
    #     def plot_chamber_ic50(chamber_id, ax):

    #         #get the substrate concentrations that match with each initial rate:
    #         substrate_concs = self._run_data[run_name]["conc_data"]

    #         ### PLOT MEAN KI FIT###
    #         #find the name of the chamber:
    #         chamber_name = chamber_names_dict[chamber_id]
    #         #first, find all chambers with this name:
    #         #if there's no data, just skip!
    #         if chamber_name not in sample_ic50_dict.keys():
    #             return ax
    #         chamber_id_list = sample_ic50_dict[chamber_name]['ic50_replicates']

    #         #convert to array indices:
    #         chamber_id_list = [list(chamber_idxs).index(x) for x in chamber_id_list]

    #         #get the initial rates for each chamber:
    #         initial_slopes = deepcopy(filtered_initial_slopes[chamber_id_list,:])
        
    #         normed_initial_slopes = initial_slopes / np.nanmax(initial_slopes, axis=1)[: , np.newaxis]

    #         #get average
    #         initial_slopes_avg = np.nanmean(normed_initial_slopes, axis=0)
    #         #get error bars
    #         initial_slopes_std = np.nanstd(normed_initial_slopes, axis=0)

    #         x_data = substrate_concs
    #         y_data = initial_slopes_avg
           
    #         #plot with error bars:
    #         ax.errorbar(x_data, y_data, yerr=initial_slopes_std,  ls="None", capsize=3, color='orange')
    #         ax.scatter(x_data, y_data, color='orange', s=100)

    #         if show_average_fit:
    #             all_fit_params = self._db_conn.get_analysis(run_name, 'ic50_raw', 'fit_params')
    #             r_max_mins = [(all_fit_params[idx]["r_max"], all_fit_params[idx]["r_min"]) for idx in sample_ic50_dict[chamber_name]['ic50_replicates']]
    #             r_max_avg = np.mean([x[0] for x in r_max_mins][0])
    #             r_min_avg = np.mean([x[1] for x in r_max_mins][0])
    #             x_logspace = np.logspace(np.log10(np.nanmin(x_data[1:])), np.log10(np.nanmax(x_data)), 100)
    #             ax.plot(x_logspace, inhibition_model(x_logspace, r_max=r_max_avg, r_min=r_min_avg, ic50=sample_ic50_dict[chamber_name]['ic50'],), color='orange', label='Average Fit')
    #             ax.fill_between(x_logspace, inhibition_model(x_logspace, r_max=r_max_avg, r_min=r_min_avg, ic50=sample_ic50_dict[chamber_name]['ic50'] - sample_ic50_dict[chamber_name]['ic50_error'],),
    #                                         inhibition_model(x_logspace,  r_max=r_max_avg, r_min=r_min_avg, ic50=sample_ic50_dict[chamber_name]['ic50'] + sample_ic50_dict[chamber_name]['ic50_error'],),
    #                                         color='orange', alpha=0.3)
            
    #         ### PLOT INDIVIDUAL K_i VALUES ###
    #         chamber_initial_slopes = filtered_initial_slopes[list(chamber_idxs).index(chamber_id), :]
    #         chamber_normed_initial_slopes = chamber_initial_slopes/ np.nanmax(chamber_initial_slopes)
    #         x_data = substrate_concs
    #         y_data = chamber_normed_initial_slopes

    #         fit_params = self._db_conn.get_analysis(run_name, 'ic50_raw', 'fit_params')[chamber_id]

    #         #plot with error bars:
    #         ax.scatter(x_data, y_data, color='blue',zorder=3, label='Chamber')
    #         x_logspace = np.logspace(np.log10(np.nanmin(x_data[1:])), np.log10(np.nanmax(x_data)), 100)
        
    #         ax.plot(x_logspace, inhibition_model(x_logspace, **fit_params), color='blue',zorder=3, label='Chamber Fit')

    #         ax.set_xscale('log')
    #         ax.text(0.03, 0.15, '$IC_{50}$' + ': {:.3f} '.format(fit_params['ic50']) +'blah', transform=ax.transAxes, fontsize=12,)
    #         ax.text(0.02,0.05, ' $\overline{IC_{50}}$: ' + '{:.3f}'.format(sample_ic50_dict[chamber_name]["ic50"]) 
    #                 + '$\pm$ {:.3f}'.format(sample_ic50_dict[chamber_name]["ic50_error"]) + ' blah', transform=ax.transAxes, fontsize=12,) 
              
    #         ax.legend()
            
    #         return ax


    #     ### PLOT THE CHIP: now, we plot
    #     plot_chip(ic50_to_plot, chamber_names_dict, graphing_function=plot_chamber_ic50, title='Filtered IC50s')

    # def plot_filtered_mm(self, run_name: str,
    #                      enzyme_conc_conversion: float = 1.0, 
    #                      show_average_fit: bool = False,
    #                      mm_model = None,
    #                      background_subtraction: bool = False):
        
    #     if mm_model is None:
    #         # default model
    #         def mm_model(x, v_max, K_m):
    #             return v_max*x/(K_m + x)
    #     elif show_average_fit:
    #         raise ValueError("Cannot show average fit with custom model.")
            
    #     #first, fill it with NaNs as a placeholder:
    #     mm_to_plot = {chamber_idx: np.nan for chamber_idx in self._run_data[run_name]["chamber_idxs"]}

    #     chamber_name_to_id_dict = self._db_conn.get_chamber_name_to_id_dict()

    #     enzyme_concentrations = deepcopy(self._run_data[run_name]["enzyme_concentration"])

    #     initial_rates_dict = self._db_conn.get_analysis(run_name, 'bgsub_linear_regression' if background_subtraction else 'linear_regression', 'slopes')
    #     initial_rate_arr = np.array(list(initial_rates_dict.values()))
    #     chamber_idxs = list(initial_rates_dict.keys())
      
    #     filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
    #     filtered_initial_slopes = deepcopy(initial_rate_arr)
    #     for filter in filters: filtered_initial_slopes *= filter

    #     sample_mm_dict = self._db_conn.get_sample_analysis_dict(run_name, 'mm_filtered')

    #     #then, fill in the values we have:
    #     for name, values in sample_mm_dict.items():
    #         for chamber_idx in chamber_name_to_id_dict[name]:
    #             cham_kcat = deepcopy(self._db_conn.get_analysis(run_name, 'mm_raw', 'kcat')[chamber_idx])

    #             mm_to_plot[chamber_idx] = cham_kcat
        
    #     chamber_names_dict = self._db_conn.get_chamber_name_dict()

    #     #plotting function: We'll generate an MM subplot for each chamber.
    #     def plot_chamber_mm(chamber_id, ax):

    #         #get the substrate concentrations that match with each initial rate:
    #         substrate_concs = self._run_data[run_name]["conc_data"]

    #         ## TODO: upper / lower bound for mean MM plotting
    #         ### PLOT MEAN KI FIT###
    #         #find the name of the chamber:
    #         chamber_name = chamber_names_dict[chamber_id]

            
    #         #first, find all chambers with this name:
    #         #if there's no data, just skip!
    #         if chamber_name not in sample_mm_dict.keys():
    #             return ax
    #         chamber_id_list = sample_mm_dict[chamber_name]['mm_replicates']
    #         #convert to array indices:
    #         chamber_id_list = [list(chamber_idxs).index(x) for x in chamber_id_list]

    #         #get the initial rates for each chamber:
    #         initial_slopes = deepcopy(filtered_initial_slopes[chamber_id_list,:])
        
    #         normed_initial_slopes = initial_slopes / (enzyme_concentrations[chamber_id_list][: , np.newaxis] * enzyme_conc_conversion)

           
    #         #get average
    #         initial_slopes_avg = np.nanmean(normed_initial_slopes, axis=0)
    #         #get error bars
    #         initial_slopes_std = np.nanstd(normed_initial_slopes, axis=0)

    #         x_data = substrate_concs
    #         y_data = initial_slopes_avg

    #         #plot with error bars:
    #         ax.errorbar(x_data, y_data, yerr=initial_slopes_std,  ls="None", capsize=3, color='orange')
    #         ax.scatter(x_data, y_data, color='orange', s=100)
          
    #         if show_average_fit:
    #             x_linspace = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
    #             ax.plot(x_linspace, mm_model(x_linspace, sample_mm_dict[chamber_name]['kcat'], sample_mm_dict[chamber_name]["K_m"]), color='orange', label='Average Fit')
    #             ax.fill_between(x_linspace, mm_model(x_linspace, sample_mm_dict[chamber_name]['kcat'] - sample_mm_dict[chamber_name]['kcat_std'], sample_mm_dict[chamber_name]['K_m']),
    #                                         mm_model(x_linspace, sample_mm_dict[chamber_name]['kcat'] + sample_mm_dict[chamber_name]['kcat_std'], sample_mm_dict[chamber_name]['K_m']),
    #                                         color='orange', alpha=0.3)

    #         ### PLOT INDIVIDUAL K_i VALUES ###
    #         chamber_initial_slopes = deepcopy(filtered_initial_slopes[list(chamber_idxs).index(chamber_id), :])
           
    #         chamber_normed_initial_slopes = chamber_initial_slopes/ (enzyme_concentrations[list(chamber_idxs).index(chamber_id)] * enzyme_conc_conversion)
            
    #         x_data = substrate_concs
    #         y_data = chamber_normed_initial_slopes

    #         fit_params = deepcopy(self._db_conn.get_analysis(run_name, 'mm_raw', 'fit_params')[chamber_id])
                    
    #         fit_params['v_max'] = fit_params['v_max']/(enzyme_concentrations[list(chamber_idxs).index(chamber_id)]* enzyme_conc_conversion)
    #         #plot with error bars:
    #         ax.scatter(x_data, y_data, color='blue', label='Chamber', zorder=3)
    #         x_linspace = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
            
    #         ax.plot(x_linspace, mm_model(x_linspace, **fit_params), color='blue', label='Chamber Fit', zorder=3)

    #         ax.legend(loc='upper left')
    #         ax.text(0.7, 0.25, '$k_{cat}$' + ': {:.2f} '.format(fit_params['v_max']) +'$s^{-1}$', transform=ax.transAxes, fontsize=15,)
    #         ax.text(0.7, 0.18, '$K_M$: {:.2f} uM'.format(fit_params['K_m']), transform=ax.transAxes, fontsize=15)
    #         ax.text(0.55,0.10, ' $\overline{k_{cat}}$: ' + '{:.2f}'.format(sample_mm_dict[chamber_name]["kcat"]) 
    #                 + '$\pm$ {:.2f}'.format(sample_mm_dict[chamber_name]["kcat_std"]) + ' $s^{-1}$', transform=ax.transAxes, fontsize=15,) 
    #         ax.text(0.55,0.03, ' $\overline{K_M}$: ' + '{:.2f}'.format(sample_mm_dict[chamber_name]["K_m"]) 
    #                 + '$\pm$ {:.2f}'.format(sample_mm_dict[chamber_name]["K_m_std"]) + ' uM', transform=ax.transAxes, fontsize=15)           
    #         return ax


    #     ### PLOT THE CHIP: now, we plot
    #     plot_chip(mm_to_plot, chamber_names_dict, graphing_function=plot_chamber_mm, title='Filtered MM')


    # def export_ic50_result_csv(self, path_to_save:str, run_name:str):

    #     chamber_names_dict = self._db_conn.get_chamber_name_dict()
    #     sample_ic50_dict = self._db_conn.get_sample_analysis_dict(run_name, 'ic50_filtered')

    #     #Full CSV, showing data for each CHAMBER:
    #     output_csv_name = 'inhibition'

    #     with open(os.path.join(path_to_save, output_csv_name+'.csv'), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         #write header:
    #         writer.writerow(['id', 
    #                         'x,y',
    #                         'substrate_name', 
    #                         'assay_type', 
    #                         'replicates', 
    #                         'Ki', 
    #                         'Ki_mean_filtered', 
    #                         'Ki_stdev_filtered', 
    #                         'enzyme',])
    #         #write data for each chamber:
    #         for i, chamber_idx in enumerate(self._run_data[run_name]["chamber_idxs"]):
    #             sample_name = chamber_names_dict[chamber_idx]
    #             #get index in sample_names:
    #             if sample_name in sample_ic50_dict.keys():
    #                 sample_dict = sample_ic50_dict[sample_name]
    #                 row = [chamber_names_dict[chamber_idx], #id
    #                         chamber_idx, #x,y
    #                         sample_name, #substrate_name
    #                         "ic50_filtered", #assay_type
    #                         len(sample_dict["ic50_replicates"]), #replicates
    #                         self._db_conn.get_analysis(run_name, 'ic50_raw','ic50')[chamber_idx], #ic50
    #                         sample_dict["ic50"], #kcat_mean_filtered
    #                         sample_dict["ic50_error"], #kcat_stdev_filtered
    #                         self._run_data[run_name]["enzyme_concentration"][i], #enzyme
    #                         ]
    #             else:
    #                 row = [chamber_names_dict[chamber_idx], #id
    #                         chamber_idx, #x,y
    #                         sample_name, #substrate_name
    #                         "ic50_filtered", #assay_type
    #                         'NaN', #replicates
    #                         self._db_conn.get_analysis(run_name, 'ic50_raw', 'ic50')[chamber_idx], #ic50
    #                         'NaN', #K_i_mean_filtered
    #                         'NaN', #K_i_stdev_filtered
    #                         self._run_data[run_name]["enzyme_concentration"][i], #enzyme
    #                 ]
                
    #             writer.writerow(row)

    #         #Summary CSV, showing data for each SAMPLE:
    #     output_csv_name = 'inhibition_summary'

    #     with open(os.path.join(path_to_save, output_csv_name)+'_short.csv', 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         #write header:
    #         writer.writerow(['id', 
    #                         'substrate_name', 
    #                         'assay_type', 
    #                         'replicates', 
    #                         'Ki_mean_filtered', 
    #                         'Ki_stdev_filtered', 
    #                         'enzyme'])
    #         #write data:
    #         for name, sample_dict in sample_ic50_dict.items():
    #             row = [name,
    #                 name,
    #                 "ic50_filtered", 
    #                 len(sample_dict["ic50_replicates"]), 
    #                 sample_dict["ic50"], 
    #                 sample_dict["ic50_error"], 
    #                 "What should this be?",
    #                 ]
    #             writer.writerow(row)
    
    # def export_mm_result_csv(self, path_to_save:str, run_name:str):

    #     chamber_names_dict = self._db_conn.get_chamber_name_dict()
    #     sample_mm_dict = self._db_conn.get_sample_analysis_dict(run_name, 'mm_filtered')

    #     #Full CSV, showing data for each CHAMBER:
    #     output_csv_name = 'mm'

    #     with open(os.path.join(path_to_save, output_csv_name+'.csv'), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         #write header:
    #         writer.writerow(['id', 
    #                         'x,y',
    #                         'sample_name', 
    #                         'assay_type', 
    #                         'replicates', 
    #                         'kcat', 
    #                         'Km',
    #                         'kcat_mean_filtered', 
    #                         'kcat_stdev_filtered',
    #                         'Km_mean_filtered',
    #                         'Km_stdev_filtered', 
    #                         'enzyme',])
    #         #write data for each chamber:
    #         for i, chamber_idx in enumerate(self._run_data[run_name]["chamber_idxs"]):
    #             sample_name = chamber_names_dict[chamber_idx]
    #             #get index in sample_names:
    #             if sample_name in sample_mm_dict.keys():
    #                 sample_dict = sample_mm_dict[sample_name]
    #                 row = [chamber_names_dict[chamber_idx], #id
    #                         chamber_idx, #x,y
    #                         sample_name, #substrate_name
    #                         "mm_filtered", #assay_type
    #                         len(sample_dict["mm_replicates"]), #replicates
    #                         self._db_conn.get_analysis(run_name, 'mm_raw','kcat')[chamber_idx], #kcat
    #                         self._db_conn.get_analysis(run_name, 'mm_raw','K_m')[chamber_idx], #Km
    #                         sample_dict["kcat"], #kcat_mean_filtered
    #                         sample_dict["kcat_std"], #kcat_stdev_filtered
    #                         sample_dict["K_m"], #Km_mean_filtered
    #                         sample_dict["K_m_std"], #Km_stdev_filtered
    #                         self._run_data[run_name]["enzyme_concentration"][i], #enzyme
    #                         ]
    #             else:
    #                 row = [chamber_names_dict[chamber_idx], #id
    #                         chamber_idx, #x,y
    #                         sample_name, #substrate_name
    #                         "mm_filtered", #assay_type
    #                         'NaN', #replicates
    #                         self._db_conn.get_analysis(run_name, 'mm_raw', 'kcat')[chamber_idx], #ic50
    #                         self._db_conn.get_analysis(run_name, 'mm_raw', 'K_m')[chamber_idx], #ic50
    #                         'NaN', #kcat_mean_filtered
    #                         'NaN', #kcat_stdev_filtered
    #                         'NaN', #Km_mean_filtered
    #                         'NaN', #Km_stdev_filtered
    #                         self._run_data[run_name]["enzyme_concentration"][i], #enzyme
    #                 ]
                
    #             writer.writerow(row)

    #         #Summary CSV, showing data for each SAMPLE:
    #     output_csv_name = 'mm_summary'

    #     with open(os.path.join(path_to_save, output_csv_name)+'_short.csv', 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         #write header:
    #         writer.writerow(['id', 
    #                         'sample_name', 
    #                         'assay_type', 
    #                         'replicates', 
    #                         'kcat_mean_filtered',
    #                         'kcat_stdev_filtered',
    #                         'Km_mean_filtered',
    #                         'Km_stdev_filtered',
    #                         'enzyme'])
    #         #write data:
    #         for name, sample_dict in sample_mm_dict.items():
    #             row = [name,
    #                 name,
    #                 "mm_filtered", 
    #                 len(sample_dict["mm_replicates"]), 
    #                 sample_dict["kcat"], 
    #                 sample_dict["kcat_std"],
    #                 sample_dict["K_m"],
    #                 sample_dict["K_m_std"], 
    #                 "What should this be?",
    #                 ]
    #             writer.writerow(row)

    # def plot_initial_rate_distribution(self, run_name: str, patterns: List[str] = ["^Buffer*"]):
    #     chamber_name_to_id_dict = self._db_conn.get_chamber_name_to_id_dict()
    #     rate_arrs = []
    #     for pattern in patterns:
    #         names = []
    #         for key in chamber_name_to_id_dict.keys():
    #             if re.match(pattern, key):
    #                 names.append(key)
    #         ids = []
    #         for name in names:
    #             ids.extend(chamber_name_to_id_dict[name])

    #         rates = []
    #         for id in ids:
    #             initial_rates = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')[id]
    #             rates.append(initial_rates)
    #         rate_arrs.append(np.array(rates).T)
        
        
    #     conc_data = ["{:.1f}".format(i) for i in self._run_data[run_name]["conc_data"]]
    #     # make data frame with columns : [concentration, pattern, rate]
    #     dict_list = []
    #     for i, pattern in enumerate(patterns):
    #         for j, rate_replicates in enumerate(rate_arrs[i]):
    #             for rate in rate_replicates:
    #                 dict_list.append({'concentration': conc_data[j], 'pattern': pattern, 'rate': rate})
                
    #     df = pd.DataFrame(dict_list)

    #     # violin plot by pattern
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     sns.violinplot(x='concentration', y='rate', hue="pattern", data=df, ax=ax)
    #     ax.set_title('Initial Rate Distribution')
    #     ax.set_xlabel('Substrate')
    #     ax.set_ylabel(f'Initial Rate ({self.get_concentration_units(run_name)}/s)')
    #     plt.show()

    # def export_all_mm_curves(self, run_name: str,
        #                 save_path: str,
        #                 substrate_name: str,
        #                 substrate_conc_unit: str,
        #                 enzyme_conc_conversion: float = 1.0, 
        #                 show_average_fit: bool = False,
        #                 mm_model = None,
        #                 background_subtraction: bool = False,
        #                 ):

        # save_path = Path(save_path)

        # if mm_model is None:
        #     # default model
        #     def mm_model(x, v_max, K_m):
        #         return v_max*x/(K_m + x)
        # elif show_average_fit:
        #     raise ValueError("Cannot show average fit with custom model.")

        # enzyme_concentrations = deepcopy(self._run_data[run_name]["enzyme_concentration"])

        # initial_rates_dict = self._db_conn.get_analysis(run_name, 'bgsub_linear_regression' if background_subtraction else 'linear_regression', 'slopes')
        # initial_rate_arr = np.array(list(initial_rates_dict.values()))
        # chamber_idxs = list(initial_rates_dict.keys())
      
        # filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
        # filtered_initial_slopes = deepcopy(initial_rate_arr)
        # for filter in filters: filtered_initial_slopes *= filter

        # sample_mm_dict = self._db_conn.get_sample_analysis_dict(run_name, 'mm_filtered')
        # #  = dict()
        # # for k, v in unsorted_sample_mm_dict.items():
        # #     sample_mm_dict[k.lower()] = v  

        # chamber_names_dict = self._db_conn.get_chamber_name_dict()
        # # chamber_names_dict = {k: v.lower() for k, v in uppercase_chamber_names_dict.items()}

        # chamber_name_to_id_dict = self._db_conn.get_chamber_name_to_id_dict()
        # # chamber_name_to_id_dict = {k.lower(): v for k, v in uppercase_chamber_name_to_id_dict.items()}
        
        # # sort sample_mm_dict alphabetically
        # sample_mm_dict = dict(sorted(sample_mm_dict.items()))

    
        # def plot_chamber_mm(chamber_id, ax, title):

        #     #get the substrate concentrations that match with each initial rate:
        #     substrate_concs = self._run_data[run_name]["conc_data"]

        #     #find the name of the chamber:
        #     chamber_name = chamber_names_dict[chamber_id]

            
        #     #first, find all chambers with this name:
        #     #if there's no data, just skip!
        #     if chamber_name not in sample_mm_dict.keys():
        #         return ax
        #     chamber_id_list = sample_mm_dict[chamber_name]['mm_replicates']
        #     #convert to array indices:
        #     chamber_id_list = [list(chamber_idxs).index(x) for x in chamber_id_list]

        #     #get the initial rates for each chamber:
        #     initial_slopes = deepcopy(filtered_initial_slopes[chamber_id_list,:])
        
        #     normed_initial_slopes = initial_slopes / (enzyme_concentrations[chamber_id_list][: , np.newaxis] * enzyme_conc_conversion)

           
        #     #get average
        #     initial_slopes_avg = np.nanmean(normed_initial_slopes, axis=0)
        #     #get error bars
        #     initial_slopes_std = np.nanstd(normed_initial_slopes, axis=0)

        #     x_data = substrate_concs
        #     y_data = initial_slopes_avg

        #     #plot with error bars:
        #     ax.errorbar(x_data, y_data, yerr=initial_slopes_std,  ls="None", capsize=3, color='blue')
        #     ax.scatter(x_data, y_data, color='blue', s=50)

        #     avg_kcat = sample_mm_dict[chamber_name]['kcat']
        #     std_kcat = sample_mm_dict[chamber_name]['kcat_std']
        #     if show_average_fit:
        #         x_linspace = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
        #         ax.plot(x_linspace, mm_model(x_linspace, avg_kcat, sample_mm_dict[chamber_name]["K_m"]), color='blue')
        #         ax.fill_between(x_linspace, mm_model(x_linspace, avg_kcat - std_kcat, sample_mm_dict[chamber_name]['K_m']),
        #                                     mm_model(x_linspace, avg_kcat + std_kcat, sample_mm_dict[chamber_name]['K_m']),
        #                                     color='blue', alpha=0.3)

            
        #     ax.text(0.2,0.15, ' $\overline{k_{cat}}$: ' + '{:.2f}'.format(avg_kcat) 
        #             + '$\pm$ {:.2f}'.format(std_kcat) + ' $s^{-1}$', transform=ax.transAxes, fontsize=8,fontfamily="Helvetica") 
        #     ax.text(0.2,0.03, ' $\overline{K_M}$: ' + '{:.2f}'.format(sample_mm_dict[chamber_name]["K_m"]) 
        #             + '$\pm$ {:.2f}'.format(sample_mm_dict[chamber_name]["K_m_std"]) + ' ' + substrate_conc_unit, transform=ax.transAxes, fontsize=8,
        #             fontfamily="Helvetica")    
            
            

        #     ax.set_title(title, fontsize=9, fontfamily="Helvetica")
        #     ax.set_xlabel('$\it{[' + substrate_name + '] (' + substrate_conc_unit +')}$', fontfamily="Helvetica", fontsize=8, fontstyle="italic")
        #     for tick in ax.xaxis.get_major_ticks():
        #         tick.label.set_fontsize(8) 
        #         tick.label.set_fontfamily("Helvetica")
        #     for tick in ax.yaxis.get_major_ticks():
        #         tick.label.set_fontsize(8) 
        #         tick.label.set_fontfamily("Helvetica")
        #     ax.set_ylabel('$v_0/[E] (s^{-1})$', fontfamily="Helvetica", fontsize=8, fontstyle="italic")
        #     return ax
        
        # #with PdfPages('/Users/duncanmuir/Desktop/mm_curves_new.pdf') as pdf:
        # page_counter = 0
        # page_num = 1
        # fig, axs = plt.subplots(5,3, figsize=(7.24, 10))
        # axs = axs.flatten()
        # #org_lookup['id'] = org_lookup['id'].str.lower()

    
        # # fixed_sample_mm_dict = dict()
        # # for old_name, values in sample_mm_dict.items():
        # #     if old_name in org_lookup['org_name'].values:
        # #             new_name = old_name
        # #     elif old_name in org_lookup['id'].str.lower().values:
        # #         new_name = org_lookup.loc[(org_lookup['id'].str.lower() == old_name), 'org_name'].values[0]

        # #     else:
        # #         print("YIKES:", old_name) 
        # #     if new_name in org_swap_dict.keys():
        # #         new_name = org_swap_dict[new_name]
        # #     fixed_sample_mm_dict[new_name] = (old_name, values)
        # # fixed_sample_mm_dict = dict(sorted(fixed_sample_mm_dict.items()))
        
        
        # for name, _ in sample_mm_dict.items():
        #     #old_name, values = values
           
        #     plot_chamber_mm(chamber_name_to_id_dict[name][0], axs[page_counter], title=name)
        #     if page_counter == 14:
        #         plt.tight_layout()
        #         #pdf.savefig(fig)
        #         plt.savefig(save_path / f"mm_curve_page_{page_num}.png", dpi=300)
        #         plt.close(fig)
        #         page_counter = 0
        #         page_num += 1
        #         fig, axs = plt.subplots(5,3, figsize=(7.24, 10))
        #         axs = axs.flatten()
        #         continue
        #     page_counter += 1
        # plt.tight_layout()
        # plt.savefig(save_path / f"mm_curve_page_{page_num}.png", dpi=300)
        # plt.close(fig)