from typing import List, Tuple
import numpy as np
import pandas as pd
import csv
import json
import os
import scipy
from tqdm import tqdm
from htbam_db_api.htbam_db_api import AbstractHtbamDBAPI
from htbam_analysis.analysis.plotting import plot_chip
from sklearn.linear_model import LinearRegression
import inspect
import seaborn as sns
from copy import deepcopy
from scipy.optimize import curve_fit


class HTBAMExperiment:
    def __init__(self, db_connection: AbstractHtbamDBAPI):
        self._db_conn = db_connection
        print("\nConnected to database.")
        print("Experiment found with the following runs:")
        print(self._db_conn.get_run_names())
        self._run_data = {}

    def get_concentration_units(self, run_name: str) -> str:
        '''
        Returns the units of concentration for the given run.
        Input:
            run_name (str): the name of the run to be analyzed.
        Output:
            str: the units of concentration for the given run.
        '''
        return self._db_conn.get_concentration_units(run_name)

    def fit_standard_curve(self, run_name: str):
        '''
        Fits a standard curve to the data in the given run.
        Input:
            run_name (str): the name of the run to be analyzed.
        Output:
            None
        '''
        if run_name not in self._run_data.keys():
            print("Existing run data not found. Fetching from database.")
            chamber_idxs, luminance_data, conc_data, _ = self._db_conn.get_run_assay_data(run_name)
            self._run_data[run_name] = {'chamber_idxs': chamber_idxs, 'luminance_data': luminance_data, 'conc_data': conc_data}

        print(f"Standard curve data found for run \"{run_name}\" with:")
        luminance_shape = self._run_data[run_name]['luminance_data'].shape
        print(f"\t-- {luminance_shape[0]} time points.\n\t-- {luminance_shape[1]} chambers.\n\t-- {luminance_shape[2]} concentrations.")
        
        print("\nFitting standard curve...")
        for i, idx in tqdm(list(enumerate(self._run_data[run_name]["chamber_idxs"]))):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(self._run_data[run_name]["conc_data"], self._run_data[run_name]["luminance_data"][:,i])
            results_dict = {'slope': slope, 'intercept': intercept, 'r_value': r_value, 'r2':r_value**2, 'p_value': p_value, 'std_err': std_err}
            self._db_conn.add_analysis(run_name, 'linear_regression', idx,results_dict)

    def plot_standard_curve_chip(self, run_name: str):
        #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
        slopes_to_plot = self._db_conn.get_analysis(run_name, 'linear_regression', 'slope')
        intercepts_to_plot = self._db_conn.get_analysis(run_name, 'linear_regression', 'intercept')
        #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
        chamber_names_dict = self._db_conn.get_chamber_name_dict()

        #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
        # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.
        def plot_chamber_slopes(chamber_id, ax):
            #parameters:
            run_name = 'standard_0'
            analysis_name = 'linear_regression'
            
            #convert from 'x,y' to integer index in the array:
            data_index = list(self._run_data[run_name]["chamber_idxs"]).index(chamber_id)
            x_data = self._run_data[run_name]["conc_data"]
            y_data = self._run_data[run_name]["luminance_data"][:,data_index]
        
            #get slope from the analysis:
            slope = np.array(list(slopes_to_plot.values()))
            intercept = np.array(list(intercepts_to_plot.values()))
            
            m = slope[data_index]
            b = intercept[data_index]
            #make a simple matplotlib plot
            ax.scatter(x_data, y_data)
            if not (np.isnan(m) or np.isnan(b)):
                #return False, no_update, no_update
                ax.plot(x_data, m*np.array(x_data) + b)
            return ax
        print(slopes_to_plot)
        plot_chip(slopes_to_plot, chamber_names_dict, graphing_function=plot_chamber_slopes, title='Standard Curve: Slope')

    def fit_initial_rates(self, run_name: str, 
                          standard_run_name: str, 
                          substrate_conc: float = None,
                          max_rxn_perc: int = 15,
                          starting_timepoint_index: int = 0,
                          max_rxn_time = np.inf):
        '''
        Fits a standard curve to the data in the given run.
        Input:
            run_name (str): the name of the run to be analyzed.
        Output:
            None
        '''
        if run_name not in self._run_data.keys():
            print("Existing run data not found. Fetching from database.")
            chamber_idxs, luminance_data, conc_data, time_data = self._db_conn.get_run_assay_data(run_name)
            self._run_data[run_name] = {'chamber_idxs': chamber_idxs, 
                                        'luminance_data': luminance_data, 
                                        'conc_data': conc_data, 
                                        'time_data': time_data}

        print(f"Activity data found for run \"{run_name}\" with:")
        luminance_shape = self._run_data[run_name]['luminance_data'].shape
        print(f"\t-- {luminance_shape[0]} time points.\n\t-- {luminance_shape[1]} chambers.\n\t-- {luminance_shape[2]} concentrations.")

        if standard_run_name not in self._run_data.keys():
            raise ValueError(f"Standard curve data not found for run \"{standard_run_name}\". Please fit the standard curve first.")
        
        print(f"Using standard curve data from run \"{standard_run_name}\" to convert luminance data to concentration data.")

        std_slopes = np.array(list(self._db_conn.get_analysis(standard_run_name, 'linear_regression', 'slope').values()))

        #calculate product concentration by dividing every chamber intensity by the slope of the standard curve for that chamber
        self._run_data[run_name]["product_concentration"] = self._run_data[run_name]["luminance_data"] / std_slopes[np.newaxis, :, np.newaxis]

        chamber_dim = len(self._run_data[run_name]["chamber_idxs"])
        conc_dim = len(self._run_data[run_name]["conc_data"])
        time_dim = len(self._run_data[run_name]["time_data"])

        arr = np.empty((chamber_dim, conc_dim))
        arr[:] = np.nan
            
        #make an array of initial slopes for each chamber: should be (#chambers , #concentrations) = (1792 x 11)
        # initial_slopes = arr.copy()
        # initial_slopes_R2 = arr.copy()
        # initial_slopes_intercepts = arr.copy()
        # reg_idx_arr = np.zeros((chamber_dim, conc_dim, time_dim)).astype(bool)

        time_series = self._run_data[run_name]["time_data"][:,0][:, np.newaxis]
        if substrate_conc is None:
            substrate_concs = self._run_data[run_name]["conc_data"]
        else:
            substrate_concs = np.array([substrate_conc for _ in range(conc_dim)])
        #print(substrate_concs)
        product_thresholds = substrate_concs * max_rxn_perc / 100
        product_thresholds = product_thresholds[:, np.newaxis]

        two_point_fits = 0 
        for i, chamber_idx in tqdm(list(enumerate(self._run_data[run_name]["chamber_idxs"]))):

            #use the kinetics package to calculate the slopes for this chamber at each substrate concentration.
            product_conc_array = self._run_data[run_name]["product_concentration"][:,i,:].T 

            # which product concentration is below the threshold?
            rxn_threshold_mask = product_conc_array < product_thresholds
            #print(time_series.shape, product_conc_array.shape, rxn_threshold_mask.shape)
            time_allowed_mask = time_series < max_rxn_time
            rxn_threshold_mask[:,:starting_timepoint_index] = 0
            #print(rxn_threshold_mask, time_allowed_mask)
            rxn_threshold_mask = np.logical_and(rxn_threshold_mask, time_allowed_mask.T)
            slopes = np.zeros_like(self._run_data[run_name]["conc_data"])
            intercepts = np.zeros_like(self._run_data[run_name]["conc_data"])
            scores = np.zeros_like(self._run_data[run_name]["conc_data"])

            for j, mask in enumerate(rxn_threshold_mask):
                if mask.sum() < 2:
                    two_point_fits += 1
                    pass
                    #print(f'Chamber {chamber_idx} Concentration {conc_data[i]} uM has less than 2 points')
                else:
                    lin_reg = LinearRegression()
                    lin_reg.fit(time_series[mask], product_conc_array[j,:][mask])
                    #print(time_series[mask], product_conc_array[i,:][mask])
                    slope, intercept, score = lin_reg.coef_, lin_reg.intercept_, lin_reg.score(time_series[mask], product_conc_array[j,:][mask])
                    #print(f'Concentration {conc_data[i]} uM has slope {slope} and intercept {intercept} with R2 {score}')
                    slopes[j] = slope
                    intercepts[j] = intercept
                    scores[j] = score
            results_dict = {'slopes': slopes, 'intercepts': intercepts, 'r_values': scores,  'mask': rxn_threshold_mask}
           
            self._db_conn.add_analysis(run_name, 'linear_regression', chamber_idx, results_dict)
            # initial_slopes[i] = slopes
            # initial_slopes_intercepts[i] = intercepts
            # initial_slopes_R2[i] = scores
            # reg_idx_arr[i] = rxn_threshold_mask
        print(f'{two_point_fits} reactions had less than 2 points for fitting')


    def plot_initial_rates_chip(self, run_name: str, time_to_plot=0.3, subtract_zeroth_point=False):

        initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')

        initial_rate_intercepts_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'intercepts')
        initial_rate_masks_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'mask') 
        intial_rate_arr = np.array(list(initial_rates_dict.values()))
        initial_rate_intercepts_arr = np.array(list(initial_rate_intercepts_dict.values()))
        initial_rate_masks_arr = np.array(list(initial_rate_masks_dict.values()))

        initial_rates_to_plot = {i: np.nanmax(j) for i, j in initial_rates_dict.items()}


        chamber_names_dict = self._db_conn.get_chamber_name_dict()


        #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
        # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.
        def plot_chamber_initial_rates(chamber_id, ax, time_to_plot=time_to_plot):
            #N.B. Every so often, slope and line colors don't match up. Not sure why.
            
            #convert from 'x,y' to integer index in the array:
            data_index = list(self._run_data[run_name]["chamber_idxs"]).index(chamber_id)
            x_data = self._run_data[run_name]["time_data"][:,0]
            y_data = self._run_data[run_name]["product_concentration"][:,data_index,:].T
            
            #plot only first X% of time:
            max_time = np.nanmax(x_data)
            time_to_plot = max_time*time_to_plot
            time_idxs_to_plot = x_data < time_to_plot
            x_data = x_data[time_idxs_to_plot]
            y_data = y_data[:, time_idxs_to_plot]
            
            # TODO: add option to subtract zeroth point(s) from all points
            #get slope from the analysis:
            current_chamber_slopes = intial_rate_arr[data_index,:]
            #calculate y-intercept by making sure it intersects first point:
            current_chamber_intercepts = initial_rate_intercepts_arr[data_index,:]
            # get regressed point mask:
            current_chamber_reg_mask = initial_rate_masks_arr[data_index,:][:,:len(x_data)]
            
            colors = sns.color_palette('husl', n_colors=y_data.shape[0])

            for i in range(y_data.shape[0]): #over each concentration:
                
                if subtract_zeroth_point:
                    try:
                        ax.scatter(x_data, y_data[i,:] - y_data[i,0], color=colors[i], alpha=0.3)
                        ax.scatter(x_data[current_chamber_reg_mask[i]],
                                   y_data[i, current_chamber_reg_mask[i]] - y_data[i, current_chamber_reg_mask[i]][0], color=colors[i], alpha=1, s=50)
                    except:
                        pass
                else:
                    ax.scatter(x_data, y_data[i,:], color=colors[i], alpha=0.3)
                    ax.scatter(x_data[current_chamber_reg_mask[i]], y_data[i, current_chamber_reg_mask[i]], color=colors[i], alpha=1, s=50)

                m = current_chamber_slopes[i]
                b = current_chamber_intercepts[i] if not subtract_zeroth_point else 0
                if not (np.isnan(m) or np.isnan(b)):
                    #return False, no_update, no_update
                    ax.plot(x_data, m*np.array(x_data) + b, color=colors[i])
            return ax

        ### PLOT THE CHIP: now, we plot
        plot_chip(initial_rates_to_plot, chamber_names_dict, graphing_function=plot_chamber_initial_rates, title='Kinetics: Initial Rates (Max)')

    def compute_enzyme_concentration(self, run_name: str, egfp_slope):
        #make numpy array of all button_quants with[ subtracted backgrounds:
        button_quant_no_background = [] #we will soon turn this into a numpy array
        for chamber_idx in self._run_data[run_name]['chamber_idxs']:
            next_button_quant = self._db_conn.get_button_quant_data(chamber_idx)
            button_quant_no_background.append(next_button_quant)
        button_quant_no_background = np.array(button_quant_no_background)

        # use eGFP standard curve to convert between button quant and eGFP concentration
        self._run_data[run_name]["enzyme_concentration"] = button_quant_no_background / egfp_slope    #in units of EGFP_SLOPE_CONC_UNITS

    def filter_initial_rates(self,
                            kinetic_run_name: str, 
                            standard_run_name: str,
                            standard_curve_r2_cutoff: float = 0.98,
                            expression_threshold: float = 1.0,
                            initial_rate_R2_threshold: float = 0.0, 
                            positive_initial_slope_filter: bool = True,):
        
        initial_slopes = np.array(list(self._db_conn.get_analysis(kinetic_run_name, 'linear_regression', 'slopes').values()))
        enzyme_concentration = self._run_data[kinetic_run_name]["enzyme_concentration"]
        chamber_count = len(self._run_data[kinetic_run_name]["chamber_idxs"])
        conc_count = len(self._run_data[kinetic_run_name]["conc_data"])
        standard_r2_values = np.array(list(self._db_conn.get_analysis(standard_run_name, 'linear_regression', 'r2').values()))
        initial_rate_r2_values = np.array(list(self._db_conn.get_analysis(kinetic_run_name, 'linear_regression', 'r_values').values()))
        
        #print(initial_slopes)
        ### Make filters ###
        filters = []
        filter_r2 = np.ones_like(initial_slopes)
       # print(filter_r2)

        # STANDARD CURVE FILTER #
        # overwrite all chambers (rows) with r^2 values below the threshold with NaNs:
        _count = 0
        for i in range(len(self._run_data[kinetic_run_name]["chamber_idxs"])):
            if standard_r2_values[i] < standard_curve_r2_cutoff:
                _count +=1
                filter_r2[i, :] = np.nan
        print('Pearson r^2 filter: {}/{} chambers pass'.format(chamber_count-_count, chamber_count))
        filter_names = ['filter_r2']
        filter_values = [standard_curve_r2_cutoff]
        filters.append(filter_r2)

        # ENZYME EXPRESSION FILTER #
        # overwrite all chambers (rows) with enzyme expression below the threshold with NaNs:
        #Double check the expression units match the EGFP units:
        #assert expression_threshhold_units == EGFP_SLOPE_CONC_UNITS, 'Error, enzyme expression and EGFP standard curve units do not match!'
        
        filter_enzyme_expression = np.ones_like(initial_slopes)
        _count = 0
        for i in range(len(self._run_data[kinetic_run_name]["chamber_idxs"])):
            if enzyme_concentration[i] < expression_threshold:
                _count +=1
                filter_enzyme_expression[i,:] = np.nan
        print('Enzyme expression filter: {}/{} chambers pass'.format(chamber_count-_count, chamber_count))
        filters.append(filter_enzyme_expression)
        filter_names.append('filter_enzyme_expression')
        filter_values.append(expression_threshold)
        #TODO: track units! 

        # INITIAL RATE FIT FILTER #
        # overwrite just the assays per chamber (single values) with initial rate fit R^2 values below the threshold with NaNs:
        filter_initial_rate_R2 = np.ones_like(initial_slopes)
        _count = 0
        for i in range(len(self._run_data[kinetic_run_name]["chamber_idxs"])):            
            _chamber_count = 0
            for j in range(conc_count):
                if initial_rate_r2_values[i,j] < initial_rate_R2_threshold:
                    _chamber_count +=1
                    filter_initial_rate_R2[i,j] = np.nan
            if conc_count - _chamber_count < 7:
                _count +=1
        print('Initial Rate R^2 filter: {}/{} chambers pass with 10 or more slopes.'.format(chamber_count-_count, chamber_count))
        filters.append(filter_initial_rate_R2)
        filter_names.append('filter_initial_rate_R2')
        filter_values.append(initial_rate_R2_threshold)

        if positive_initial_slope_filter:
            # POSITIVE INITIAL SLOPE FILTER #
            # overwrite just the assays per chamber (single values) with initial slopes below zero with NaNs:
            filter_positive_initial_slope = np.ones_like(initial_slopes)
            _count = 0
            for i in range(len(self._run_data[kinetic_run_name]["chamber_idxs"])):
                _chamber_count = 0
                for j in range(conc_count):
                    if initial_slopes[i,j] < 0:
                        _chamber_count +=1
                        filter_positive_initial_slope[i,j] = np.nan
                if conc_count - _chamber_count < 7:
                    _count +=1
            print('Positive Initial Slope filter: {}/{} chambers pass with 10 or more slopes.'.format(chamber_count-_count, chamber_count))
            filters.append(filter_positive_initial_slope)
        filter_names.append('filter_positive_initial_slope')
        filter_values.append(positive_initial_slope_filter)

        ### manually flagged wells ###
        #TODO: implement

        ### TODO: make visualization!
        # chamber_idxs, luminance_data, conc_data, time_data

        filtered_initial_slopes = deepcopy(initial_slopes)
        for filter in filters: filtered_initial_slopes *= filter

        assay_dict = {"filters": filters}
        assay_dict.update({filter_names[i]: filter_values[i] for i in range(len(filter_names))})

        # #initialize the dictionary
       
        assay_data = {}
        for i in range(conc_count):
            assay_data[i] = {
                'substrate_conc': self._run_data[kinetic_run_name]["conc_data"][i],
                'chambers': {}
            }
            for j, chamber_idx in enumerate(self._run_data[kinetic_run_name]["chamber_idxs"]):
                assay_data[i]['chambers'][chamber_idx] = {
                    'slope': filtered_initial_slopes[j,i],
                    'r2': initial_rate_r2_values[j,i]
                }

        assay_dict["assays"] = assay_data
        self._db_conn.add_filtered_assay(kinetic_run_name, 'filtered_initial_rates', assay_dict)

    def plot_filtered_initial_rates_chip(self, run_name: str, time_to_plot=0.3):
        ###N.B.: May be some bug here, because some of the filtered-out chambers are still showing slopes.
        # I think they should have all nans...?

        initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')
        initial_rate_intercepts_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'intercepts')
        initial_rate_masks_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'mask') 
        initial_rate_arr = np.array(list(initial_rates_dict.values()))
        initial_rate_intercepts_arr = np.array(list(initial_rate_intercepts_dict.values()))
        initial_rate_masks_arr = np.array(list(initial_rate_masks_dict.values()))


        filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
        filtered_initial_slopes = deepcopy(initial_rate_arr)
        for filter in filters: filtered_initial_slopes *= filter
        
        chamber_names_dict = self._db_conn.get_chamber_name_dict()

        #Let's plot as before:
        #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
        # TODO: I don't think this is what freitas intended, but will probably work for now. revisit.
        filtered_initial_rates_to_plot = {i: np.nanmax(j) for i, j in initial_rates_dict.items()}


        #chamber_names: Same as before.

        #plotting function: We'll generate a subplot for each chamber, showing the raw data and the linear regression line.
        # to do this, we make a function that takes in the chamber_id and the axis object, and returns the axis object after plotting. Do NOT plot.show() in this function.

        def plot_chamber_filtered_initial_rates(chamber_id, ax, time_to_plot=time_to_plot):
            #N.B. Every so often, slope and line colors don't match up. Not sure why.
            #parameters: what amount of total time to plot? First 20%?
        
            #convert from 'x,y' to integer index in the array:
            data_index = list(self._run_data[run_name]["chamber_idxs"]).index(chamber_id)
            x_data = self._run_data[run_name]["time_data"][:,0]
            y_data = self._run_data[run_name]["product_concentration"][:,data_index,:].T
            
            #plot only first X% of time:
            max_time = np.nanmax(x_data)
            time_to_plot = max_time*time_to_plot
            time_idxs_to_plot = x_data < time_to_plot
            x_data = x_data[time_idxs_to_plot]
            y_data = y_data[:, time_idxs_to_plot]
            
            #get slope from the analysis:
            current_chamber_slopes = filtered_initial_slopes[data_index,:]
            #calculate y-intercept by making sure it intersects first point:
            current_chamber_intercepts = initial_rate_intercepts_arr[data_index,:]
            # get regressed point mask:
            current_chamber_reg_mask = initial_rate_masks_arr[data_index,:][:,:len(x_data)]
            
            colors = sns.color_palette('husl', n_colors=y_data.shape[0])

            #print(y_data.shape[0])
            for i in range(y_data.shape[0]): #over each concentration:
                
                ax.scatter(x_data, y_data[i,:], color=colors[i], alpha=0.3)
                ax.scatter(x_data[current_chamber_reg_mask[i]], y_data[i, current_chamber_reg_mask[i]], color=colors[i], alpha=1, s=50)
                
                m = current_chamber_slopes[i]
                b = current_chamber_intercepts[i]
                if not (np.isnan(m) or np.isnan(b)):
                    #return False, no_update, no_update
                    ax.plot(x_data, m*np.array(x_data) + b, color=colors[i])
            return ax

            

        ### PLOT THE CHIP: now, we plot
        plot_chip(filtered_initial_rates_to_plot, chamber_names_dict, graphing_function=plot_chamber_filtered_initial_rates, title='Kinetics: Filtered Initial Rates (Max)')
        print('{}/1792 wells pass our filters.'.format( 
            np.sum([np.any(~np.isnan(filtered_initial_slopes[i,:])) for i in range(len(self._run_data[run_name]["chamber_idxs"]))]) ) )


    def fit_ic50s(self, run_name: str, inhibition_model = None):

        if inhibition_model is None:
            # default model
            def inhibition_model(x, r_max, r_min, ic50):
                return r_min + (r_max-r_min)/(1+(x/ic50))
        
        arg_list = str(inspect.signature(inhibition_model)).strip("()").replace(" ", "").split(",")

        # get data
        initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')
        initial_rate_arr = np.array(list(initial_rates_dict.values()))
        chamber_idxs = list(initial_rates_dict.keys())
      
        filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
        filtered_initial_slopes = deepcopy(initial_rate_arr)
        for filter in filters: filtered_initial_slopes *= filter

        #Here, we calculate the IC50 for each chamber.
        ic50_array = np.array([])
        ic50_error_array = np.array([])
        fit_params = []
        fit_params_errs = []
        _count = 0
        for i in range(len(chamber_idxs)):
            current_slopes = filtered_initial_slopes[i, :]

            if np.all(np.isnan(current_slopes)) or np.all(current_slopes == 0) or np.nanmax(current_slopes) == 0:
                #print('Chamber {} has no slopes!'.format(chamber_idxs[i]))
                ic50_array = np.append(ic50_array, np.nan)
                ic50_error_array = np.append(ic50_error_array, np.nan)
                fit_params.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
                fit_params_errs.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
                _count += 1
                continue

            #get indices of non-nan values:
            non_nan_idxs = np.where(~np.isnan(current_slopes))[0]
            
            current_slopes = current_slopes[non_nan_idxs]
            current_concs = self._run_data[run_name]["conc_data"][non_nan_idxs]

            if len(current_slopes) < 3:
                #print('Chamber {} has fewer than 3 slopes!'.format(chamber_idxs[i]))
                ic50_array = np.append(ic50_array, np.nan)
                ic50_error_array = np.append(ic50_error_array, np.nan)
                fit_params.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
                fit_params_errs.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
                _count += 1
                continue

            max_normed_slopes = current_slopes / np.nanmax(current_slopes)
            #kinetics.fit_and_plot_micheaelis_menten(current_slopes, current_slopes, current_concs, enzyme_concentration_converted_units[i], 'uM', 'MM for first chamber!')
            #K_i, std_err = kinetics.fit_inhibition_constant(max_normed_slopes, max_normed_slopes, current_concs, enzyme_concentration_converted_units[i], 'uM', 'MM for first chamber!')
            p_opt, p_cov = curve_fit(inhibition_model, current_concs, max_normed_slopes)
            param_dict = {i:j for i,j in zip(arg_list[1:], p_opt)}

            if "ic50" not in param_dict.keys():
                raise ValueError("Inhibition model must have an 'ic50' parameter.")
            
            ic50 = param_dict["ic50"]
            fit_err = np.sqrt(np.diag(p_cov))
            param_error_dict = {i:j for i,j in zip(arg_list[1:], fit_err)}
        
            ic50_err = param_error_dict["ic50"]

            ic50_array = np.append(ic50_array, ic50)
            ic50_error_array = np.append(ic50_error_array, ic50_err)
            fit_params.append(param_dict)
            fit_params_errs.append(param_error_dict)
        # chamber_idxs, luminance_data, conc_data, time_data

        for i, chamber_idx in enumerate(chamber_idxs):
            self._db_conn.add_analysis(run_name, 'ic50_raw', chamber_idx,  {'ic50': ic50_array[i], 
                                                                            'ic50_error': ic50_error_array[i],
                                                                            'fit_params': fit_params[i],
                                                                            'fit_params_errs': fit_params_errs[i]} )
        print(f'{_count} chambers had fewer than 3 slopes and were not fit.')

    def filter_ic50s(self, 
                     run_name,
                     z_score_threshold_ic50 = 1.5,
                     z_score_threshold_expression = 1.5,
                     save_intermediate_data = False):

        ic50_array = np.array(list(self._db_conn.get_analysis(run_name, 'ic50_raw', 'ic50').values()))
        enzyme_concentration = self._run_data[run_name]["enzyme_concentration"]

        #Get chamber ids from metadata:
       
        #Get chamber ids from metadata:
        chamber_names_dict = self._db_conn.get_chamber_name_to_id_dict()

        #Get average k_cat, k_M, and v_max for each sample:
        sample_names = np.array([])
        sample_ic50 = np.array([])
        sample_ic50_error = np.array([])
        sample_ic50_replicates = []

        # #Get z-scores for each well (used to filter in the next step!)
        # ic50_zscores = np.array([])
        # enzyme_concentration_zscores = np.array([])

        export_list1=[]
        #For each sample, 
        for name, ids in chamber_names_dict.items():
            

            ### GATHER MM PARAMETERS OF REPLICATES FOR EACH SAMPLE: ###
            #get indices of idxs in chamber_idxs:
            idxs = [list(self._run_data[run_name]["chamber_idxs"]).index(x) for x in ids]

            #get values for those indices:
            ic50s = ic50_array[idxs]

            #keep track of which wells we exclude later:
            ic50_replicates = np.array(ids)

            #if any of these is all nans, just continue to avoid errors:
            if np.all(np.isnan(ic50s)):
                print('No values from sample {}, all pre-filtered.'.format(name))
                continue

            ### FILTER OUT OUTLIERS: ###
            #calculate z-score for each value:
            ic50_zscore = (ic50s - np.nanmean(ic50s))/np.nanstd(ic50s)

            #also, get z-score of enzyme expression for each well:
            enzyme_concentration_zscore = (enzyme_concentration[idxs] - np.nanmean(enzyme_concentration[idxs]))/np.nanstd(enzyme_concentration[idxs]) #in units of 'substrate_conc_unit' 

            #First, for enzyme expression outliers, set the value to NaN to be filtered in the final step:
            ic50s[np.abs(enzyme_concentration_zscore) > z_score_threshold_expression] = np.nan

            #filter out values with z-score > threshhold:
            ic50s = ic50s[np.abs(ic50_zscore) < z_score_threshold_ic50]

            #do the same for the replicates ids:
            ic50_replicates = ic50_replicates[np.abs(ic50_zscore) < z_score_threshold_ic50]

            #remove nan values from all (nan values are due to both no experimental data, and z-score filtering)
            ic50_replicates = ic50_replicates[~np.isnan(ic50s)]
            ic50s = ic50s[~np.isnan(ic50s)]

            if len(ic50s) < 3:
                print('Not enough replicates for sample {}. Skipping.'.format(name))
                continue
            
            #get average values:
            sample_names = np.append(sample_names, name)
            sample_ic50 = np.append(sample_ic50, np.mean(ic50s))
            sample_ic50_error = np.append(sample_ic50_error,np.std(ic50s))
            
            #keep track of replicates:
            sample_ic50_replicates.append(ic50_replicates)

            if save_intermediate_data:
                temp_list1 = []
                temp_list1.append(name)
                for ic50 in ic50s:
                    temp_list1.append(ic50)
                export_list1.append(temp_list1)

        if save_intermediate_data:   
            df2 = pd.DataFrame(export_list1)
            df2.to_csv('ic50_file_intermediate.csv')      
        for i, sample_name in enumerate(sample_names):
            self._db_conn.add_sample_analysis(run_name, 'ic50_filtered', sample_name, {'ic50': sample_ic50[i], 'ic50_error': sample_ic50_error[i], 'ic50_replicates': sample_ic50_replicates[i]})
          
        print('Average number of replicates per sample post-filtering: {}'.format(int(np.round(np.mean([len(i) for i in sample_ic50_replicates]), 0))))


    def fit_mm(self, run_name: str, enzyme_conc_conversion: float = 1.0, mm_model = None):

        if mm_model is None:
            # default model
            def mm_model(x, v_max, K_m):
                return v_max*x/(K_m + x)
        
        arg_list = str(inspect.signature(mm_model)).strip("()").replace(" ", "").split(",")

        # get data
        initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')
        initial_rate_arr = np.array(list(initial_rates_dict.values()))
        chamber_idxs = list(initial_rates_dict.keys())
      
        filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
        filtered_initial_slopes = deepcopy(initial_rate_arr)
        for filter in filters: filtered_initial_slopes *= filter

        enzyme_concentrations = deepcopy(self._run_data[run_name]["enzyme_concentration"])

        #Here, we calculate the IC50 for each chamber.
        kcat_array = np.array([])
        Km_array = np.array([])
        fit_params = []
        fit_params_errs = []
        _count = 0
        for i in range(len(chamber_idxs)):
            current_slopes = filtered_initial_slopes[i, :]

            if np.all(np.isnan(current_slopes)) or np.all(current_slopes == 0):
                #print('Chamber {} has no slopes!'.format(chamber_idxs[i]))
                kcat_array = np.append(kcat_array, np.nan)
                Km_array = np.append(Km_array, np.nan)
                fit_params.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
                fit_params_errs.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
                _count += 1
                continue

            #get indices of non-nan values:
            non_nan_idxs = np.where(~np.isnan(current_slopes))[0]
            
            current_slopes = current_slopes[non_nan_idxs]
            current_concs = self._run_data[run_name]["conc_data"][non_nan_idxs]

            # NOTE: this should be optimized
            if len(current_slopes) < 5:
                #print('Chamber {} has fewer than 3 slopes!'.format(chamber_idxs[i]))
                kcat_array = np.append(kcat_array, np.nan)
                Km_array = np.append(Km_array, np.nan)
                fit_params.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
                fit_params_errs.append({i:j for i,j in zip(arg_list[1:], [np.nan]*len(arg_list[1:]))})
                _count += 1
                continue

            #kinetics.fit_and_plot_micheaelis_menten(current_slopes, current_slopes, current_concs, enzyme_concentration_converted_units[i], 'uM', 'MM for first chamber!')
            #K_i, std_err = kinetics.fit_inhibition_constant(max_normed_slopes, max_normed_slopes, current_concs, enzyme_concentration_converted_units[i], 'uM', 'MM for first chamber!')
            p_opt, p_cov = curve_fit(mm_model, current_concs, current_slopes)
            param_dict = {i:j for i,j in zip(arg_list[1:], p_opt)}

            if ("v_max" not in param_dict.keys()) or ("K_m" not in param_dict.keys()):
                raise ValueError("MM model must have 'v_max' and 'K_m' parameters.")
            
            v_max = param_dict["v_max"]
            fit_err = np.sqrt(np.diag(p_cov))
            param_error_dict = {i:j for i,j in zip(arg_list[1:], fit_err)}

          
            kcat_array = np.append(kcat_array, v_max/ (enzyme_concentrations[i] * enzyme_conc_conversion))
            
            Km_array = np.append(Km_array, param_dict["K_m"])
            fit_params.append(param_dict)
            fit_params_errs.append(param_error_dict)
        # chamber_idxs, luminance_data, conc_data, time_data

        for i, chamber_idx in enumerate(chamber_idxs):
            self._db_conn.add_analysis(run_name, 'mm_raw', chamber_idx,  {'kcat': kcat_array[i], 
                                                                            'K_m': Km_array[i],
                                                                            'fit_params': fit_params[i],
                                                                            'fit_params_errs': fit_params_errs[i]} )
        print(f'{_count} chambers had fewer than 5 slopes and were not fit.')

    def filter_mm(self, 
                    run_name,
                    z_score_threshold_mm = 1.5,
                    z_score_threshold_expression = 1.5,
                    save_intermediate_data = False):

        kcat_array = np.array(list(self._db_conn.get_analysis(run_name, 'mm_raw', 'kcat').values()))
        Km_array = np.array(list(self._db_conn.get_analysis(run_name, 'mm_raw', 'K_m').values()))
        enzyme_concentration = self._run_data[run_name]["enzyme_concentration"]
       
        #Get chamber ids from metadata:
        chamber_names_dict = self._db_conn.get_chamber_name_to_id_dict()

        #Get average k_cat, k_M, and v_max for each sample:
        sample_names = np.array([])
        sample_kcat = np.array([])
        sample_kcat_std = np.array([])
        sample_K_m = np.array([])
        sample_K_m_std = np.array([])
        sample_mm_replicates = []

        export_list1=[]
        #For each sample, 
        for name, ids in chamber_names_dict.items():
            

            ### GATHER MM PARAMETERS OF REPLICATES FOR EACH SAMPLE: ###
            #get indices of idxs in chamber_idxs:
            idxs = [list(self._run_data[run_name]["chamber_idxs"]).index(x) for x in ids]

            #get values for those indices:
            kcats = kcat_array[idxs]
            K_ms = Km_array[idxs]

            #keep track of which wells we exclude later:
            mm_replicates = np.array(ids)

            #if any of these is all nans, just continue to avoid errors:
            if np.all(np.isnan(kcats)) or np.all(np.isnan(K_ms)):
                print('No values from sample {}, all pre-filtered.'.format(name))
                continue

            ### FILTER OUT OUTLIERS: ###
            #calculate z-score for each value:
            kcat_zscore = (kcats - np.nanmean(kcats))/np.nanstd(kcats)
            K_m_zscore = (K_ms - np.nanmean(K_ms))/np.nanstd(K_ms)

            #also, get z-score of enzyme expression for each well:
            enzyme_concentration_zscore = (enzyme_concentration[idxs] - np.nanmean(enzyme_concentration[idxs]))/np.nanstd(enzyme_concentration[idxs]) #in units of 'substrate_conc_unit' 

            z_score_mask = np.logical_and(np.abs(enzyme_concentration_zscore) < z_score_threshold_expression, np.abs(kcat_zscore) < z_score_threshold_mm, np.abs(K_m_zscore) < z_score_threshold_mm)
            
            #First, for enzyme expression outliers, set the value to NaN to be filtered in the final step:
            kcats[~z_score_mask] = np.nan
            K_ms[~z_score_mask] = np.nan
            mm_replicates[~z_score_mask] = np.nan

            #remove nan values from all (nan values are due to both no experimental data, and z-score filtering)
            nan_mask = np.logical_and(~np.isnan(kcats), ~np.isnan(K_ms))
            kcats = kcats[nan_mask]
            K_ms = K_ms[nan_mask]
            mm_replicates = mm_replicates[nan_mask]
          


            if len(kcats) < 3:
                print('Not enough replicates for sample {}. Skipping.'.format(name))
                continue
            
            #get average values:
            sample_names = np.append(sample_names, name)
            sample_kcat = np.append(sample_kcat, np.mean(kcats))
            sample_kcat_std = np.append(sample_kcat_std, np.std(kcats))
            sample_K_m = np.append(sample_K_m, np.mean(K_ms))
            sample_K_m_std = np.append(sample_K_m_std, np.std(K_ms))
            
            #keep track of replicates:
            sample_mm_replicates.append(mm_replicates)

            if save_intermediate_data:
                temp_list1 = []
                temp_list1.append(name)
                for kcat in kcats:
                    temp_list1.append(kcat)
                export_list1.append(temp_list1)

        if save_intermediate_data:   
            df2 = pd.DataFrame(export_list1)
            df2.to_csv('mm_file_intermediate.csv')      
        for i, sample_name in enumerate(sample_names):
            self._db_conn.add_sample_analysis(run_name, 'mm_filtered', sample_name, {'kcat': sample_kcat[i], 'kcat_std': sample_kcat_std[i],
                                                                                     'K_m': sample_K_m[i] , 'K_m_std': sample_K_m_std[i],
                                                                                     'mm_replicates': sample_mm_replicates[i]})
          
        print('Average number of replicates per sample post-filtering: {}'.format(int(np.round(np.mean([len(i) for i in sample_mm_replicates]), 0))))

    def plot_filtered_ic50(self, run_name: str, inhibition_model = None):
        
        if inhibition_model is None:
            # default model
            def inhibition_model(x, r_max, r_min, ic50):
                return r_min + (r_max-r_min)/(1+(x/ic50))
            
        #first, fill it with NaNs as a placeholder:
        ic50_to_plot = {chamber_idx: np.nan for chamber_idx in self._run_data[run_name]["chamber_idxs"]}

        chamber_name_to_id_dict = self._db_conn.get_chamber_name_to_id_dict()

        initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')
        initial_rate_arr = np.array(list(initial_rates_dict.values()))
        chamber_idxs = list(initial_rates_dict.keys())
      
        filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
        filtered_initial_slopes = deepcopy(initial_rate_arr)
        for filter in filters: filtered_initial_slopes *= filter
        sample_ic50_dict = self._db_conn.get_sample_analysis_dict(run_name, 'ic50_filtered')
        #then, fill in the values we have:
        for name, values in sample_ic50_dict.items():
            for chamber_idx in chamber_name_to_id_dict[name]:

                ic50_to_plot[chamber_idx] = values['ic50']
        
        chamber_names_dict = self._db_conn.get_chamber_name_dict()

        #plotting function: We'll generate an MM subplot for each chamber.
        def plot_chamber_ic50(chamber_id, ax):

            #get the substrate concentrations that match with each initial rate:
            substrate_concs = self._run_data[run_name]["conc_data"]

            ### PLOT MEAN KI FIT###
            #find the name of the chamber:
            chamber_name = chamber_names_dict[chamber_id]
            #first, find all chambers with this name:
            #if there's no data, just skip!
            if chamber_name not in sample_ic50_dict.keys():
                return ax
            chamber_id_list = sample_ic50_dict[chamber_name]['ic50_replicates']

            #convert to array indices:
            chamber_id_list = [list(chamber_idxs).index(x) for x in chamber_id_list]

            #get the initial rates for each chamber:
            initial_slopes = filtered_initial_slopes[chamber_id_list,:]
        
            normed_initial_slopes = initial_slopes / np.nanmax(initial_slopes, axis=1)[: , np.newaxis]

            #get average
            initial_slopes_avg = np.nanmean(normed_initial_slopes, axis=0)
            #get error bars
            initial_slopes_std = np.nanstd(normed_initial_slopes, axis=0)

            x_data = substrate_concs
            y_data = initial_slopes_avg

            #plot with error bars:
            ax.errorbar(x_data, y_data, yerr=initial_slopes_std,  fmt='o', label="Average")


            ### PLOT INDIVIDUAL K_i VALUES ###
            chamber_initial_slopes = filtered_initial_slopes[list(chamber_idxs).index(chamber_id), :]
            chamber_normed_initial_slopes = chamber_initial_slopes/ np.nanmax(chamber_initial_slopes)
            x_data = substrate_concs
            y_data = chamber_normed_initial_slopes

            fit_params = self._db_conn.get_analysis(run_name, 'ic50_raw', 'fit_params')[chamber_id]

            #plot with error bars:
            ax.scatter(x_data, y_data, color='green', s=100, label='Chamber')
            x_logspace = np.logspace(np.log10(np.nanmin(x_data[1:])), np.log10(np.nanmax(x_data)), 100)
        
            ax.plot(x_logspace, inhibition_model(x_logspace, **fit_params), color='green', label='Chamber Fit')

            ax.set_xscale('log')
            ax.legend()
            
            return ax


        ### PLOT THE CHIP: now, we plot
        plot_chip(ic50_to_plot, chamber_names_dict, graphing_function=plot_chamber_ic50, title='Filtered IC50s')

    def plot_filtered_mm(self, run_name: str,
                         enzyme_conc_conversion: float = 1.0, 
                         show_average_fit: bool = False,
                         mm_model = None):
        
        if mm_model is None:
            # default model
            def mm_model(x, v_max, K_m):
                return v_max*x/(K_m + x)
        elif show_average_fit:
            raise ValueError("Cannot show average fit with custom model.")
            
        #first, fill it with NaNs as a placeholder:
        mm_to_plot = {chamber_idx: np.nan for chamber_idx in self._run_data[run_name]["chamber_idxs"]}

        chamber_name_to_id_dict = self._db_conn.get_chamber_name_to_id_dict()

        enzyme_concentrations = deepcopy(self._run_data[run_name]["enzyme_concentration"])

        initial_rates_dict = self._db_conn.get_analysis(run_name, 'linear_regression', 'slopes')
        initial_rate_arr = np.array(list(initial_rates_dict.values()))
        chamber_idxs = list(initial_rates_dict.keys())
      
        filters = self._db_conn.get_filters(run_name, 'filtered_initial_rates')
        filtered_initial_slopes = deepcopy(initial_rate_arr)
        for filter in filters: filtered_initial_slopes *= filter

        sample_mm_dict = self._db_conn.get_sample_analysis_dict(run_name, 'mm_filtered')

        #then, fill in the values we have:
        for name, values in sample_mm_dict.items():
            for chamber_idx in chamber_name_to_id_dict[name]:
                cham_kcat = deepcopy(self._db_conn.get_analysis(run_name, 'mm_raw', 'kcat')[chamber_idx])

                mm_to_plot[chamber_idx] = cham_kcat
        
        chamber_names_dict = self._db_conn.get_chamber_name_dict()

        #plotting function: We'll generate an MM subplot for each chamber.
        def plot_chamber_mm(chamber_id, ax):

            #get the substrate concentrations that match with each initial rate:
            substrate_concs = self._run_data[run_name]["conc_data"]

            ## TODO: upper / lower bound for mean MM plotting
            ### PLOT MEAN KI FIT###
            #find the name of the chamber:
            chamber_name = chamber_names_dict[chamber_id]

            
            #first, find all chambers with this name:
            #if there's no data, just skip!
            if chamber_name not in sample_mm_dict.keys():
                return ax
            chamber_id_list = sample_mm_dict[chamber_name]['mm_replicates']
            #convert to array indices:
            chamber_id_list = [list(chamber_idxs).index(x) for x in chamber_id_list]

            #get the initial rates for each chamber:
            initial_slopes = deepcopy(filtered_initial_slopes[chamber_id_list,:])
        
            normed_initial_slopes = initial_slopes / (enzyme_concentrations[chamber_id_list][: , np.newaxis] * enzyme_conc_conversion)

           
            #get average
            initial_slopes_avg = np.nanmean(normed_initial_slopes, axis=0)
            #get error bars
            initial_slopes_std = np.nanstd(normed_initial_slopes, axis=0)

            x_data = substrate_concs
            y_data = initial_slopes_avg

            #plot with error bars:
            ax.errorbar(x_data, y_data, yerr=initial_slopes_std,  ls="None", capsize=3, color='orange')
            ax.scatter(x_data, y_data, color='orange', s=100)
          
            if show_average_fit:
                x_linspace = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
                ax.plot(x_linspace, mm_model(x_linspace, sample_mm_dict[chamber_name]['kcat'], sample_mm_dict[chamber_name]["K_m"]), color='orange', label='Average Fit')
                ax.fill_between(x_linspace, mm_model(x_linspace, sample_mm_dict[chamber_name]['kcat'] - sample_mm_dict[chamber_name]['kcat_std'], sample_mm_dict[chamber_name]['K_m']),
                                            mm_model(x_linspace, sample_mm_dict[chamber_name]['kcat'] + sample_mm_dict[chamber_name]['kcat_std'], sample_mm_dict[chamber_name]['K_m']),
                                            color='orange', alpha=0.3)

            ### PLOT INDIVIDUAL K_i VALUES ###
            chamber_initial_slopes = deepcopy(filtered_initial_slopes[list(chamber_idxs).index(chamber_id), :])
           
            chamber_normed_initial_slopes = chamber_initial_slopes/ (enzyme_concentrations[list(chamber_idxs).index(chamber_id)] * enzyme_conc_conversion)
            
            x_data = substrate_concs
            y_data = chamber_normed_initial_slopes

            fit_params = deepcopy(self._db_conn.get_analysis(run_name, 'mm_raw', 'fit_params')[chamber_id])
                    
            fit_params['v_max'] = fit_params['v_max']/(enzyme_concentrations[list(chamber_idxs).index(chamber_id)]* enzyme_conc_conversion)
            #plot with error bars:
            ax.scatter(x_data, y_data, color='blue', label='Chamber', zorder=3)
            x_linspace = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
            
            ax.plot(x_linspace, mm_model(x_linspace, **fit_params), color='blue', label='Chamber Fit', zorder=3)

            ax.legend(loc='upper left')
            ax.text(0.7, 0.25, '$k_{cat}$' + ': {:.2f} '.format(fit_params['v_max']) +'$s^{-1}$', transform=ax.transAxes, fontsize=15,)
            ax.text(0.7, 0.18, '$K_M$: {:.2f} uM'.format(fit_params['K_m']), transform=ax.transAxes, fontsize=15)
            ax.text(0.55,0.10, ' $\overline{k_{cat}}$: ' + '{:.2f}'.format(sample_mm_dict[chamber_name]["kcat"]) 
                    + '$\pm$ {:.2f}'.format(sample_mm_dict[chamber_name]["kcat_std"]) + ' $s^{-1}$', transform=ax.transAxes, fontsize=15,) 
            ax.text(0.55,0.03, ' $\overline{K_M}$: ' + '{:.2f}'.format(sample_mm_dict[chamber_name]["K_m"]) 
                    + '$\pm$ {:.2f}'.format(sample_mm_dict[chamber_name]["K_m_std"]) + ' uM', transform=ax.transAxes, fontsize=15)           
            return ax


        ### PLOT THE CHIP: now, we plot
        plot_chip(mm_to_plot, chamber_names_dict, graphing_function=plot_chamber_mm, title='Filtered MM')


    def export_ic50_result_csv(self, path_to_save:str, run_name:str):

        chamber_names_dict = self._db_conn.get_chamber_name_dict()
        sample_ic50_dict = self._db_conn.get_sample_analysis_dict(run_name, 'ic50_filtered')

        #Full CSV, showing data for each CHAMBER:
        output_csv_name = 'inhibition'

        with open(os.path.join(path_to_save, output_csv_name+'.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            #write header:
            writer.writerow(['id', 
                            'x,y',
                            'substrate_name', 
                            'assay_type', 
                            'replicates', 
                            'Ki', 
                            'Ki_mean_filtered', 
                            'Ki_stdev_filtered', 
                            'enzyme',])
            #write data for each chamber:
            for i, chamber_idx in enumerate(self._run_data[run_name]["chamber_idxs"]):
                sample_name = chamber_names_dict[chamber_idx]
                #get index in sample_names:
                if sample_name in sample_ic50_dict.keys():
                    sample_dict = sample_ic50_dict[sample_name]
                    row = [chamber_names_dict[chamber_idx], #id
                            chamber_idx, #x,y
                            sample_name, #substrate_name
                            "ic50_filtered", #assay_type
                            len(sample_dict["ic50_replicates"]), #replicates
                            self._db_conn.get_analysis(run_name, 'ic50_raw','ic50')[chamber_idx], #ic50
                            sample_dict["ic50"], #kcat_mean_filtered
                            sample_dict["ic50_error"], #kcat_stdev_filtered
                            self._run_data[run_name]["enzyme_concentration"][i], #enzyme
                            ]
                else:
                    row = [chamber_names_dict[chamber_idx], #id
                            chamber_idx, #x,y
                            sample_name, #substrate_name
                            "ic50_filtered", #assay_type
                            'NaN', #replicates
                            self._db_conn.get_analysis(run_name, 'ic50_raw', 'ic50')[chamber_idx], #ic50
                            'NaN', #K_i_mean_filtered
                            'NaN', #K_i_stdev_filtered
                            self._run_data[run_name]["enzyme_concentration"][i], #enzyme
                    ]
                
                writer.writerow(row)

            #Summary CSV, showing data for each SAMPLE:
        output_csv_name = 'inhibition_summary'

        with open(os.path.join(path_to_save, output_csv_name)+'_short.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            #write header:
            writer.writerow(['id', 
                            'substrate_name', 
                            'assay_type', 
                            'replicates', 
                            'Ki_mean_filtered', 
                            'Ki_stdev_filtered', 
                            'enzyme'])
            #write data:
            for name, sample_dict in sample_ic50_dict.items():
                row = [name,
                    name,
                    "ic50_filtered", 
                    len(sample_dict["ic50_replicates"]), 
                    sample_dict["ic50"], 
                    sample_dict["ic50_error"], 
                    "What should this be?",
                    ]
                writer.writerow(row)
    def export_mm_result_csv(self, path_to_save:str, run_name:str):

        chamber_names_dict = self._db_conn.get_chamber_name_dict()
        sample_mm_dict = self._db_conn.get_sample_analysis_dict(run_name, 'mm_filtered')

        #Full CSV, showing data for each CHAMBER:
        output_csv_name = 'mm'

        with open(os.path.join(path_to_save, output_csv_name+'.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            #write header:
            writer.writerow(['id', 
                            'x,y',
                            'sample_name', 
                            'assay_type', 
                            'replicates', 
                            'kcat', 
                            'Km',
                            'kcat_mean_filtered', 
                            'kcat_stdev_filtered',
                            'Km_mean_filtered',
                            'Km_stdev_filtered', 
                            'enzyme',])
            #write data for each chamber:
            for i, chamber_idx in enumerate(self._run_data[run_name]["chamber_idxs"]):
                sample_name = chamber_names_dict[chamber_idx]
                #get index in sample_names:
                if sample_name in sample_mm_dict.keys():
                    sample_dict = sample_mm_dict[sample_name]
                    row = [chamber_names_dict[chamber_idx], #id
                            chamber_idx, #x,y
                            sample_name, #substrate_name
                            "mm_filtered", #assay_type
                            len(sample_dict["mm_replicates"]), #replicates
                            self._db_conn.get_analysis(run_name, 'mm_raw','kcat')[chamber_idx], #kcat
                            self._db_conn.get_analysis(run_name, 'mm_raw','K_m')[chamber_idx], #Km
                            sample_dict["kcat"], #kcat_mean_filtered
                            sample_dict["kcat_std"], #kcat_stdev_filtered
                            sample_dict["K_m"], #Km_mean_filtered
                            sample_dict["K_m_std"], #Km_stdev_filtered
                            self._run_data[run_name]["enzyme_concentration"][i], #enzyme
                            ]
                else:
                    row = [chamber_names_dict[chamber_idx], #id
                            chamber_idx, #x,y
                            sample_name, #substrate_name
                            "mm_filtered", #assay_type
                            'NaN', #replicates
                            self._db_conn.get_analysis(run_name, 'mm_raw', 'kcat')[chamber_idx], #ic50
                            self._db_conn.get_analysis(run_name, 'mm_raw', 'K_m')[chamber_idx], #ic50
                            'NaN', #kcat_mean_filtered
                            'NaN', #kcat_stdev_filtered
                            'NaN', #Km_mean_filtered
                            'NaN', #Km_stdev_filtered
                            self._run_data[run_name]["enzyme_concentration"][i], #enzyme
                    ]
                
                writer.writerow(row)

            #Summary CSV, showing data for each SAMPLE:
        output_csv_name = 'mm_summary'

        with open(os.path.join(path_to_save, output_csv_name)+'_short.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            #write header:
            writer.writerow(['id', 
                            'sample_name', 
                            'assay_type', 
                            'replicates', 
                            'kcat_mean_filtered',
                            'kcat_stdev_filtered',
                            'Km_mean_filtered',
                            'Km_stdev_filtered',
                            'enzyme'])
            #write data:
            for name, sample_dict in sample_mm_dict.items():
                row = [name,
                    name,
                    "mm_filtered", 
                    len(sample_dict["mm_replicates"]), 
                    sample_dict["kcat"], 
                    sample_dict["kcat_std"],
                    sample_dict["K_m"],
                    sample_dict["K_m_std"], 
                    "What should this be?",
                    ]
                writer.writerow(row)
    ##############################
    #### Freiatas W.I.P. Code ####
    ##############################

    # def __init__(self, file:str, new:bool=False, units_registry=None):
    #     '''
    #     Initializes an HTBAM_Experiment object.
    #     Input:
    #         file (str): the path to the .HTBAM file to be read.
    #         new (bool): if True, a new .HTBAM file will be created.
    #         units_registry (pint.UnitRegistry): a pint unit registry to be used for all units.
    #     Output:
    #         None
    #     '''

    #     super().__init__()
    #     self._experiment_file = Path(file)

    #     if units_registry is None:
    #         units_registry = pint.UnitRegistry()
    #     self.ureg = units_registry
    #     #make sure these are set up
    #     self.ureg.setup_matplotlib(True)
    #     self.ureg.define('RFU = [luminosity]')

    #     #does it have the correct extension?
    #     if self._experiment_file.suffix != ".HTBAM":
    #         raise ValueError(f"File {self._experiment_file} does not have the correct extension. Must be .HTBAM")

    #     if not new:
    #         data = self._get_dict_from_file()
    #     else:
    #         data = self._init_dict()
    #         Path(self._experiment_file).touch()
    #         self._write_file(data)

    #     #is it the correct version?
    #     if data["file_version"] != CURRENT_VERSION:
    #         print(f"Warning: File {self._experiment_file} was created with a different version. You're currently using {CURRENT_VERSION}.")

    # def __repr__(self) -> str:
    #     def recursive_string(d: dict, indent: int, width=5) -> str:
    #         s = "\t"*indent + '{\n'
    #         for i, (key, value) in enumerate(d.items()):
    #             if i == width:
    #                 s += "\t"*indent +"...\n"
    #                 break
    #             s += "\t"*indent + f"{key}: "
    #             if isinstance(value, dict):
    #                 s += "\n" + recursive_string(value, indent+1)
    #             else:
    #                 s += f"{value}\n"
    #         s += "\t"*indent + '}\n'
    #         return s
        
    #     data = self.get('') #get the whole thing
    #     return recursive_string(data, 0)
    
    # def get(self, path):
    #     '''
    #     Returns the data from a given "path" in the database.
    #     Input: 
    #         path (str): the "path" to the data to be returned.
    #             ex: "runs/standard_0/assays/0/chambers/1,1/sum_chamber"
    #     Output:
    #         data: the data at the given path.
    #     '''
    #     #is path a str or Path object?
    #     if type(path) == str:
    #         path = Path(path)

    #     #if root, return the whole thing
    #     if path == "":
    #         return self._get_dict_from_file()
        
    #     #split the path object:
    #     path = path.parts
    #     data = self._get_dict_from_file()
    #     path_traversed = ""
    #     for p in path:
    #         # TODO: get wildcard working, like db.get('runs/standard_0/assays/*/chambers/1,1/sum_chamber')
    #         # if p == "*":
    #         #     #if we're at a wildcard, find the matching values in ALL children:
    #         #     if type(data) == dict:
    #         #         children = list(data.keys())
    #         #     else:
    #         #         raise ValueError(f"Wildcard found at {path_traversed}, which is a {type(data)}. Wildcards can only be used on dicts.")
    #         #     data_list = [self.get(path_traversed + child + "/") for child in children]
    #         #     for d in data_list:
    #         #         print(d)
    #         #     data = np.concatenate(data_list)
    #         #     return data
            
    #         #handle errors
    #         if p not in data:
    #             error_string = f"Path {path_traversed+p} not found in database. \n"
    #             if type(data) == dict:
    #                 error_string += f"Made it to {path_traversed}, which has keys {data.keys()}"
    #             else:
    #                 error_string += f"Made it to {path_traversed}, which is a {type(data)}"
    #             raise ValueError(error_string)
    #         data = data[p]
    #         path_traversed += p + "/"

    #     #if this is a quantity, convert to a pint.Quantity
    #     if type(data) == dict:
    #         if 'is_quantity' in data.keys() and data['is_quantity']:
    #             data = self._dict_to_quantity(data)

    #     # #if this is a dict, convert all quantities to pint.Quantities
    #     # if type(data) == dict:
    #     #     data = self._make_quantities_from_serialized_dict(data)
        
    #     return data
    
    # ##########################################################################################
    # ########################### READING / WRITING to/from FILE  ##############################
    # ##########################################################################################

    # def _init_dict(self) -> dict:
    #     '''
    #     Populates an initial dictionary with chamber specific metadata.
    #             Parameters:
    #                     None

    #             Returns:
    #                     None
    #     '''        
    #     return {
    #         "file_version": CURRENT_VERSION,
    #         "chamber_metadata": {},
    #         "button_quant": {},
    #         "runs": {},
    #     }

    # def _get_dict_from_file(self) -> dict:
    #     '''
    #     Returns the dictionary stored in the .HTBAM file.
    #         Parameters: None
    #         Returns: json_dict (dict): Dictionary stored in the .HTBAM file.
    #     '''
    #     with open(self._experiment_file, 'r') as fp:
    #         json_dict = json.load(fp)
    #     return json_dict

    # def _write_file(self, data):
    #     '''This writes the database to file, as a dict -> json.
    #     This will overwrite the existing file.'''
    #     with open(self._experiment_file, 'w') as fp:
    #         json.dump(data, fp, indent=4)

    # def _update_file(self, path, new_data):
    #     '''This appends data to a given path in the database'''
    #     #is path a str or Path object?
    #     if type(path) == str:
    #         path = Path(path)

    #     #is new_data a quantity? If so, convert to dict
    #     if type(new_data) == pint.Quantity:
    #         new_data = self._quantity_to_dict(new_data)

    #     path = path.parts
    #     full_data = self._get_dict_from_file()
    #     current_data = full_data #this will be updated as we traverse the path
    #     path_traversed = ""

    #     #N.B.: If anything is passed by value instead of by reference, this will break.
    #     for p in path[:-1]:
    #         #handle errors
    #         if p not in current_data:
    #             error_string = f"Path {path_traversed+p} not found in database. \n"
    #             if type(current_data) == dict:
    #                 error_string += f"Made it to {path_traversed}, which has keys {current_data.keys()}"
    #             else:
    #                 error_string += f"Made it to {path_traversed}, which is a {type(current_data)}"
    #             raise ValueError(error_string)
            
    #         #continue down the path
    #         current_data = current_data[p]
    #         path_traversed += p + "/"

    #     #update the data
    #     if path[-1] in current_data: 
    #         if current_data[path[-1]] != {}: #We can't overwrite data, but we'll allow overwriting blank placeholder dicts.
    #             raise ValueError(f"Path {path_traversed+path[-1]} already exists in database. Overwriting data is forbidden.")
    #     current_data[path[-1]] = new_data
        
    #     #write to file
    #     #now, we've iterated down our full dict and changed some part of it. We'll pass back the full dict.
    #     self._write_file(full_data)

    # ##########################################################################################
    # ######################### SERIALIZE / DESERIALIZE QUANTITY DATA ##########################
    # ##########################################################################################
    # def _dict_to_quantity(self, data: dict) -> pint.Quantity:
    #     '''
    #     Converts a dictionary with keys 'values', 'unit', and 'is_quantity' to a pint.Quantity.
    #     This allows us to receive our data from a json file.
    #     '''
    #     if not data['is_quantity']:
    #         raise ValueError("Data is not a quantity.")
        
    #     try:
    #         return np.array(data['values']) * self.ureg(data['unit'])
    #     except:
    #         raise ValueError(f"Could not convert data to quantity. Data: {data}")
        
    # def _quantity_to_dict(self, quantity: pint.Quantity) -> dict:
    #     '''
    #     Converts a pint.Quantity to a dictionary with keys 'values', 'unit', and 'is_quantity'.
    #     The reason we need this is so we can store our data in a JSON. json.dump() can't handle np.arrays or pint.Quantities.
    #     '''
    #     return {
    #         'values': quantity.magnitude.tolist(),
    #         'unit': str(quantity.units),
    #         'is_quantity': True
    #     }
    
    # def _make_serializable_dict(self, data: dict) -> dict:
    #     '''
    #     Converts all pint.Quantities in a dictionary to dictionaries with keys 'values', 'unit', and 'is_quantity'.
    #     This allows us to store our data in a JSON. json.dump() can't handle np.arrays or pint.Quantities.
    #     '''
    #     for key, value in data.items():
    #         if isinstance(value, pint.Quantity):
    #             data[key] = self._quantity_to_dict(value)
    #         elif isinstance(value, dict):
    #             data[key] = self._make_serializable_dict(value)
    #     return data
    
    # def _make_quantities_from_serialized_dict(self, data: dict) -> dict:
    #     '''
    #     Converts all dictionaries with keys 'values', 'unit', and 'is_quantity' to pint.Quantities.
    #     This allows us to receive our data from a json file.
    #     '''
    #     for key, value in data.items():
    #         if isinstance(value, dict):
    #             if 'is_quantity' in value.keys() and value['is_quantity']:
    #                 data[key] = self._dict_to_quantity(value)
    #             else:
    #                 data[key] = self._make_quantities_from_serialized_dict(value)
    #     return data

    # ##########################################################################################
    # ######################### LOADING EXPERIMENT DATA FROM CSVs  #############################
    # ##########################################################################################
    # #(make _load_chamber_metadata !)
    # def _load_chamber_metadata(self,standard_data_df) -> None:
    #     '''
    #     Populates an json_dict with kinetic data with the following schema:
    #     {button_quant: {
         
    #         1,1: {
    #             sum_chamber: {'values': [...],
    #                         'unit': 'RFU',
    #                         'is_quantity': True},
    #             std_chamber: {'values': [...],
    #                         'unit': 'RFU',
    #                         'is_quantity': True},
    #         },
    #         ...
    #         }}

    #     Parameters:
    #         standard_data_df (pd.DataFrame): Dataframe from standard curve

    #     Returns:
    #         None
    #     '''    
    #     unique_chambers = standard_data_df[['id','x_center_chamber', 'y_center_chamber', 'radius_chamber', 
    #         'xslice', 'yslice', 'indices']].drop_duplicates(subset=["indices"]).set_index("indices")
    #     self._update_file(Path("chamber_metadata"), unique_chambers.to_dict("index") )

    # def load_standard_data_from_file(self, standard_curve_data_path: str, standard_name: str, standard_type: str, standard_units: str) -> None:
    #     '''
    #     Populates an dict with standard curve data with the following schema, and saves to the .HTBAM file:
    #     {standard_run_#: {
    #         name: str,
    #         type: str,
    #         assays: {
    #             1: {
    #                 conc: float,
    #                 time:
    #                     {'values': [...],
    #                     'unit': 's',
    #                     'is_quantity': True},
    #                 chambers: {
    #                     1,1: {
    #                         sum_chamber: {'values': [...],
    #                                       'units': 'RFU',
    #                                       'is_quantity': True},
    #                         std_chamber: {'values': [...],
    #                                       'unit': 'RFU'}
    #                                       'is_quantity': True},
    #                     },
    #                     ...
    #                     }}}}

    #     Parameters:
    #         standard_curve_data_path (str): Path to standard curve data
    #         standard_name (str): Name of standard curve
    #         standard_type (str): Type of standard curve
    #         standard_units (str): Units of standard curve

    #     Returns:
    #             None
    #     '''
    #     #First, check if our standard_units is a valid unit of concentration:
    #     if self.ureg(standard_units).dimensionality != self.ureg.molar.dimensionality:
    #         raise ValueError(f"Units {standard_units} are not a valid unit of concentration.\nIs your capitalization correct?")
        
    #     standard_data_df = pd.read_csv(standard_curve_data_path)
    #     standard_data_df['indices'] = standard_data_df.x.astype('str') + ',' + standard_data_df.y.astype('str')
    #     i = 0    
    #     std_assay_dict = {}
    #     #TODO: this bad convention of column names will be phased out soon.
    #     for prod_conc, subset in standard_data_df.groupby("concentration_uM"):
    #         squeezed = _squeeze_df(subset, grouping_index="indices", squeeze_targets=['sum_chamber', 'std_chamber'])
    #         squeezed["time_s"] = pd.Series([[0]]*len(squeezed), index=squeezed.index) #this tomfoolery is used to create a list with a single value, 0, for the standard curve assays.
            
    #         #turn prod_conc into a pint.Quantity:
    #         prod_conc = np.array(prod_conc) * self.ureg(standard_units)
            
    #         #turn it into a pint.Quantity:
    #         time_quantity = squeezed.iloc[0]["time_s"] * self.ureg("s")

    #         #now, we need to properly format the data for each chamber:
    #         chambers_dict_unformatted = squeezed.drop(columns=["time_s", "indices"]).to_dict("index")
    #         chambers_dict = {}
    #         for chamber_coord, chamber_data in chambers_dict_unformatted.items():
    #             chambers_dict[chamber_coord]  = {}
    #             for key, value in chamber_data.items():
    #                 #convert to pint.Quantity:
    #                 chambers_dict[chamber_coord][key] = value * self.ureg("RFU")

    #         #make dict for each assay
    #         std_assay_dict[i] = {
    #             "conc": prod_conc, #serialize using our custom _quantity_to_dict function
    #             "time": time_quantity, 
    #             "chambers": chambers_dict}
            
    #         i += 1

    #     std_run_num = len([key for key in self.get(Path('runs')) if "standard_" in key])
    #     standard_data_dict = {
    #         "name": standard_name,
    #         "type": standard_type,
    #         "assays": std_assay_dict
    #         }
        
    #     #append to file
    #     standard_data_dict = self._make_serializable_dict(standard_data_dict) #convert all quantities to dicts so we can save to json
    #     self._update_file(Path("runs") / f"standard_{std_run_num}", standard_data_dict)
        
    #     #update chamber metadata
    #     self._load_chamber_metadata(standard_data_df)
       
    # def load_kinetics_data_from_file(self, kinetic_data_path: str, kinetic_name: str, kinetic_type: str, kinetic_units: str) -> None:
    #     '''
    #     Populates an dict with kinetic data with the following schema, and saves to the .HTBAM file:
    #     {kinetics_run_#: {
    #         name: str,
    #         type: str,
    #         assays: {
    #             1: {
    #                 conc: float,
    #                 time: {'values': [...],
    #                         'unit': 's',
    #                         'is_quantity': True},,
    #                 chambers: {
    #                     1,1: {
    #                         sum_chamber: {'values': [...],
    #                                       'unit': 'RFU',
    #                                       'is_quantity': True},
    #                         std_chamber: {'values': [...],
    #                                       'unit': 'RFU'}
    #                                       'is_quantity': True},
    #                     },
    #                     ...
    #                     }}}}

    #     Parameters:
    #         kinetic_data_path (str): Path to kinetic data
    #         kinetic_name (str): Name of kinetic data
    #         kinetic_type (str): Type of kinetic data
    #         kinetic_units (str): Units of kinetic data

    #     Returns:
    #             None
    #     '''    

    #     kinetic_data_df = pd.read_csv(kinetic_data_path)
    #     kinetic_data_df['indices'] = kinetic_data_df.x.astype('str') + ',' + kinetic_data_df.y.astype('str')

    #     def parse_concentration(conc_str: str):
    #         '''
    #         Currently, we're storing substrate concentration as a string in the kinetics data.
    #         This will be changed in the future to store as a float + unit as a string. For now,
    #         we will parse jankily.
    #         '''
    #         print('Warning: parsing concentration from string')
    #         #first, remove the unit and everything following
    #         conc = conc_str.split(kinetic_units)[0]
    #         #concentration number uses underscore as decimal point. Here, we replace and convert to a float:
    #         conc = float(conc.replace("_", "."))
    #         return conc
        
    #     i = 0    
    #     kin_dict = {}
    #     for sub_conc, subset in kinetic_data_df.groupby("series_index"):
    #         squeezed = _squeeze_df(subset, grouping_index="indices", squeeze_targets=["time_s",'sum_chamber', 'std_chamber'])
            
    #         #turn sub_conc into a pint.Quantity:
    #         sub_conc = np.array(parse_concentration(sub_conc)) * self.ureg(kinetic_units)

    #         #turn time into a pint.Quantity:
    #         time_quantity = np.array(squeezed.iloc[0]["time_s"]) * self.ureg("s")

    #         #now, we need to properly format the data for each chamber:
    #         chambers_dict_unformatted = squeezed.drop(columns=["time_s", "indices"]).to_dict("index")
    #         chambers_dict = {}
    #         for chamber_coord, chamber_data in chambers_dict_unformatted.items():
    #             chambers_dict[chamber_coord]  = {}
    #             for key, value in chamber_data.items():
    #                 #convert to pint.Quantity:
    #                 chambers_dict[chamber_coord][key] = value * self.ureg("RFU")

    #         #make dict for each assay
    #         kin_dict[i] = {
    #             "conc": sub_conc, 
    #             "time": time_quantity,
    #             "chambers": chambers_dict}
    #         i += 1

    #     kinetics_run_num = len([key for key in self.get(Path('runs')) if "kinetics_" in key])
    #     kinetics_data_dict = {
    #         "name": kinetic_name,
    #         "type": kinetic_type,
    #         "assays": kin_dict
    #     }
        
    #     #append to file
    #     kinetics_data_dict = self._make_serializable_dict(kinetics_data_dict) #convert all quantities to dicts so we can save to json
    #     self._update_file(Path("runs") / f"kinetics_{kinetics_run_num}", kinetics_data_dict)

    # def load_button_quant_data_from_file(self, kinetic_data_path: str) -> None:
    #     '''
    #     Populates an json_dict with kinetic data with the following schema:
    #     {button_quant: {
         
    #         1,1: {
    #             sum_chamber: [...],
    #             std_chamber: [...]
    #         },
    #         ...
    #         }}

    #             Parameters:
    #                     None

    #             Returns:
    #                     None
    #     '''    
    #     kinetic_data_df = pd.read_csv(kinetic_data_path)
    #     kinetic_data_df['indices'] = kinetic_data_df.x.astype('str') + ',' + kinetic_data_df.y.astype('str')
    #     try:
    #         unique_buttons = kinetic_data_df[["summed_button_Button_Quant","summed_button_BGsub_Button_Quant",
    #         "std_button_Button_Quant", "indices"]].drop_duplicates(subset=["indices"]).set_index("indices")
    #     except KeyError:
    #         raise HtbamDBException("ButtonQuant columns not found in kinetic data.")
            
    #     button_quant_dict = unique_buttons.to_dict("index")
    #     #convert to pint.Quantity:
    #     for chamber_coord, chamber_dict in button_quant_dict.items():
    #         for key, value in chamber_dict.items():
    #             button_quant_dict[chamber_coord][key] = np.array([value]) * self.ureg("RFU")


    #     button_quant_dict = self._make_serializable_dict(button_quant_dict) #convert all quantities to dicts so we can save to json
    #     self._update_file("button_quant", button_quant_dict)


    # ##########################################################################################
    # ############################## UTILITIES: ANALYSIS  ######################################
    # ##########################################################################################
    # def get_chamber_coords(self):
    #     '''
    #     Returns the chamber ids for a given run.
    #     Input:
    #         None
    #     Output:
    #         chamber_ids: an array of the chamber ids (in the format '1,1' ... '32,56')
    #             shape: (n_chambers,)
    #     '''
    #     chamber_coords = np.array(list(self.get('chamber_metadata').keys()))
    #     return chamber_coords

    # def get_chamber_names(self):
    #     '''
    #     Returns the chamber names for a given run.
    #     Input:
    #         None
    #     Output:
    #         chamber_names: an array of the chamber names (in the format 'ecADK'...)
    #             shape: (n_chambers,)
    #     '''
    #     metadata_dict = self.get('chamber_metadata')
    #     chamber_coords = self.get_chamber_coords() #this way chamber_coords and chamber_names are always the same order.
    #     chamber_names = np.array([metadata_dict[chamber_coord]['id'] for chamber_coord in chamber_coords])
    #     return chamber_names
    
    # def get_run_data(self, run_name):
    #     '''
    #     Returns the data from a given run as numpy arrays.
    #     Input: 
    #         run_name (str): the name of the run to be converted to numpy arrays.
    #     Output:
    #         chamber_ids: an array of the chamber ids (in the format '1,1' ... '32,56')
    #             shape: (n_chambers,)
    #         luminance_data: an array of the luminance data for each chamber
    #             shape: (n_time_points, n_chambers, n_assays)
    #         conc_data: an array of the concentration data for each chamber.
    #             shape: (n_assays,)
    #         time_data: an array of the time data for each time point.
    #             shape: (n_time_points, n_assays)
    #     '''
    #     #get data from this run as a dict:
    #     run_data = self.get(Path("runs") / run_name)
    #     #convert all serialized quantities to pint.Quantities:
    #     run_data = self._make_quantities_from_serialized_dict(run_data)

    #     #get chamber_coords from file:
    #     chamber_coords = np.array(list(self.get('chamber_metadata').keys()))
    #     luminance_data = None
    #     time_data = None
    #     conc_data = np.array([])

    #     #Each assay may have recorded a different # of time points.
    #     #First, we'll just check what the max # of time points is:
    #     max_time_points = 0
    #     for assay in run_data['assays'].keys():
    #         current_assay_time_points = len(run_data['assays'][assay]['time'])
    #         if current_assay_time_points > max_time_points:
    #             max_time_points = current_assay_time_points

    #     for assay in run_data['assays'].keys():
    #         #to make things easier later, we'll be sorting the datapoints by time value.
    #         #Get time data:
    #         current_time_array = run_data['assays'][assay]['time']
    #         current_time_array = current_time_array.astype(float) #so we can pad with NaNs
    #         #pad the array with NaNs if there are fewer time points than the max
    #         current_time_array = np.pad(current_time_array, (0, max_time_points - len(current_time_array)), 'constant', constant_values=np.nan)
    #         #sort, and capture sorting idxs:
    #         sorting_idxs = np.argsort(current_time_array)
    #         current_time_array = current_time_array[sorting_idxs]
    #         current_time_array = np.expand_dims(current_time_array, axis=1)
    #         #add to our dataset
    #         if time_data is None:
    #             time_data = current_time_array
    #         else:
    #             time_data = np.concatenate([time_data, current_time_array], axis=1)

    #         #Get luminance data:
    #         current_luminance_array = None
    #         for chamber_idx in chamber_coords:
    #             #collect from DB
    #             current_chamber_array = run_data['assays'][assay]['chambers'][chamber_idx]['sum_chamber']
    #             #set type to float:
    #             current_chamber_array = current_chamber_array.astype(float)
    #             #pad the array with NaNs if there are fewer time points than the max
    #             current_chamber_array = np.pad(current_chamber_array, (0, max_time_points - len(current_chamber_array)), 'constant', constant_values=np.nan)
    #             #sort by time:
    #             current_chamber_array = current_chamber_array[sorting_idxs]
    #             #add a dimension at the end:
    #             current_chamber_array = np.expand_dims(current_chamber_array, axis=1)

    #             if current_luminance_array is None:
    #                 current_luminance_array = current_chamber_array
    #             else:
    #                 current_luminance_array = np.concatenate([current_luminance_array, current_chamber_array], axis=1)
    #         #add a dimension at the end:
    #         current_luminance_array = np.expand_dims(current_luminance_array, axis=2)
    #         #add to our dataset
    #         if luminance_data is None:
    #             luminance_data = current_luminance_array
    #         else:
    #             luminance_data = np.concatenate([luminance_data, current_luminance_array], axis=2)
            
    #         #Get concentration data:
    #         #collect from DB
    #         current_conc = run_data['assays'][assay]['conc']
    #         conc_data = np.append(conc_data, current_conc)

    #     #sort once more, by conc_data:
    #     sorting_idxs = np.argsort(conc_data)
    #     conc_data = conc_data[sorting_idxs]

    #     #sort luminance data by conc_data:
    #     luminance_data = luminance_data[:,:,sorting_idxs]
        
    #     return chamber_coords, luminance_data, conc_data, time_data
    
    # def save_new_analysis(self, run_name, analysis_name, chamber_dict):
    #     '''
    #     Creates a new analysis in the database. 
    #     Input:
    #         analysis_name (str): the name of the analysis to be created.
    #         run_name (str): the name of the run to be analyzed.
    #         analysis_type (str): the type of analysis to be performed.
    #         analysis_params (dict): a dictionary of parameters for the analysis.
    #     Output:
    #         None
    #     '''

    #     analysis_path = Path("runs") / run_name / 'analyses' / analysis_name

    #     #use get(path) to check if the analysis already exists:
    #     if 'analyses' not in self.get(Path("runs") / run_name).keys():
    #         self._update_file(Path("runs") / run_name / "analyses", {})

    #     #our analysis must have one entry for each chamber. Let's verify this:
    #     chamber_coords = self.get_chamber_coords()
    #     for chamber_coord in chamber_coords:
    #         if chamber_coord not in chamber_dict.keys():
    #             raise ValueError(f"Chamber {chamber_coord} is missing from the analysis data. \n \
    #                                 Analysis must have one entry for each chamber.")

    #     analysis_dict = {
    #         "chambers": chamber_dict,
    #     }

    #     #write to file
    #     analysis_dict = self._make_serializable_dict(analysis_dict)
    #     self._update_file(analysis_path, analysis_dict)
    

    

    # def export_json(self):
    #     '''This writes the database to file, as a dict -> json'''
    #     with open('db.json', 'w') as fp:
    #         json.dump(self._json_dict, fp, indent=4)