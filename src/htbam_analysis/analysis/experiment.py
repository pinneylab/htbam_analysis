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
from htbam_analysis.analysis.fit import mm_model
#from htbam_analysis.analysis.fit import fit_luminance_vs_time, fit_luminance_vs_concentration

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt


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
            dep_var_units=run_data.dep_var_units,
            meta=meta
        )
        self.set_run(save_as, masked)

    ### DATA EXPORT ###
    def export_sample_vmax_data(self, run_name: str, output_dir: str = None, file_name: str = None):
        '''
        Export per-sample vmax data to a CSV file, for Data2D objects.
        This is an intermediate output step, before we divdie by [E] and output kcats.
        Contains per-sample vmax, KM, replicate number (after filtering), and lists chambers IDs.

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

        # Get sample names
        sample_names = run_data.indep_vars.sample_IDs
        unique_samples = np.unique(sample_names)

        # Get units
        vmax_units = f"{run_data.dep_var_units[0]:~}"
        kM_units = f"{run_data.dep_var_units[1]:~}"

        # New empty dataframe:
        df = pd.DataFrame(columns=['sample', f'avg_v_max ({vmax_units})', f'avg_K_M ({kM_units})', 'avg_fit_R2', 'replicates', 'chamber_IDs'])

        # Iterate over samples
        for sample in unique_samples:
            # Get indices of replicates for this sample
            sample_indices = np.where(sample_names == sample)[0]

            # Get rates for this sample
            sample_vmax = run_data.dep_var[..., 0][sample_indices]
            sample_kM = run_data.dep_var[..., 1][sample_indices]
            sample_r2 = run_data.dep_var[..., 2][sample_indices]

            # Get chamber IDs for this sample
            sample_chamber_IDs = run_data.indep_vars.chamber_IDs[sample_indices]

            # Get replicate number by counting non-nan values
            replicate_number = np.count_nonzero(~np.isnan(sample_vmax))

            nonnan_sample_chamber_IDs = sample_chamber_IDs[~np.isnan(sample_vmax)]

            # Get average rate
            if replicate_number > 0:
                avg_vmax = np.nanmean(sample_vmax)
                avg_kM = np.nanmean(sample_kM)
                avg_r2 = np.nanmean(sample_r2)
            else:
                avg_vmax = np.nan
                avg_kM = np.nan
                avg_r2 = np.nan

            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([{
                'sample': sample,
                f'avg_v_max ({vmax_units})': avg_vmax,
                f'avg_K_M ({kM_units})': avg_kM,
                'avg_fit_R2': avg_r2,
                'replicates': replicate_number,
                'chamber_IDs': nonnan_sample_chamber_IDs,
            }])], ignore_index=True)

        # Save to CSV
        if file_name is None:
            file_name = f'{run_name}_sample_rate_data.csv'
        df.to_csv(f'{output_dir}/{file_name}', index=False)

        print(f"Sample rate data exported to {output_dir}/{file_name}")


    def export_chamber_vmax_data(self, run_name: str, enzyme_concentration_run_name: str, output_dir: str = None, file_name: str = None):
        '''
        Export per-chamber vmax data to a CSV file, for Data2D objects.
        Input:
            run_name (str): the name of the run to be exported.
            enzyme_concentration_run_name (str): the name of the run with enzyme concentration data.
            output_dir (str): the directory to save the CSV file. If None, uses current directory.
            file_name (str): the name of the file to save the CSV file. If None, uses run_name.
        Output:
            None
        '''
        run_data = self.get_run(run_name)

        enzyme_concentration_data = self.get_run(enzyme_concentration_run_name)

        assert isinstance(run_data, Data2D), "run_data must be of type Data2D."

        if output_dir is None:
            output_dir = os.getcwd()

        # Get sample names
        chamber_names = run_data.indep_vars.chamber_IDs
        sample_names = run_data.indep_vars.sample_IDs

        # Get units
        vmax_units = f"{run_data.dep_var_units[0]:~}"
        kM_units = f"{run_data.dep_var_units[1]:~}"
        E_units = f"{enzyme_concentration_data.dep_var_units[0]:~}"

        # New empty dataframe:
        df = pd.DataFrame(columns=['chamber', 'sample', f'v_max ({vmax_units})', f'K_M ({kM_units})', f'E_conc ({E_units})', 'fit_R2'])

        # Iterate over samples
        for chamber in chamber_names:
            # Get indices of chambers for this sample
            chamber_index = np.where(chamber_names == chamber)[0]

            # Get rates for this sample
            chamber_vmax = run_data.dep_var[..., 0][chamber_index][0]
            chamber_kM = run_data.dep_var[..., 1][chamber_index][0]
            chamber_r2 = run_data.dep_var[..., 2][chamber_index][0]

            chamber_E_conc = enzyme_concentration_data.dep_var[..., 0][chamber_index][0]

            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([{
                'chamber': chamber,
                'sample': sample_names[chamber_index],
                f'v_max ({vmax_units})': chamber_vmax,
                f'K_M ({kM_units})': chamber_kM,
                f'E_conc ({E_units})': chamber_E_conc,
                'fit_R2': chamber_r2,
            }])], ignore_index=True)

        # Save to CSV
        if file_name is None:
            file_name = f'{run_name}_chamber_rate_data.csv'
        df.to_csv(f'{output_dir}/{file_name}', index=False)

        print(f"Chamber rate data exported to {output_dir}/{file_name}")


    

    def export_MM_sample_data(self, run_name: str, enzyme_concentration_run_name: str, output_dir: str = None, file_name: str = None):
        '''
        Export per-sample kcat data to a CSV file.
        Calculates kcat = vmax / [E] for each replicate, then averages.

        Input:
            run_name (str): the name of the run with vmax/Km data.
            enzyme_concentration_run_name (str): the name of the run with enzyme concentration data.
            output_dir (str): the directory to save the CSV file. If None, uses current directory.
            file_name (str): the name of the file to save the CSV file. If None, uses run_name.
        Output:
            None
        '''
        run_data = self.get_run(run_name)
        enzyme_concentration_data = self.get_run(enzyme_concentration_run_name)

        assert isinstance(run_data, Data2D), "run_data must be of type Data2D."

        if output_dir is None:
            output_dir = os.getcwd()

        # Get sample names
        sample_names = run_data.indep_vars.sample_IDs
        unique_samples = np.unique(sample_names)

        # Get units
        vmax_unit_obj = run_data.dep_var_units[0]
        E_unit_obj = enzyme_concentration_data.dep_var_units[0]
        kcat_unit_obj = run_data.dep_var_units[3]

        vmax_units = f"{vmax_unit_obj:~}"
        kM_units = f"{run_data.dep_var_units[1]:~}"
        E_units = f"{E_unit_obj:~}"
        kcat_units = f"{kcat_unit_obj:~}"

        # New empty dataframe:
        df = pd.DataFrame(columns=['sample', f'avg_k_cat ({kcat_units})', f'avg_K_M ({kM_units})', f'avg_E_conc ({E_units})', 'avg_fit_R2', 'replicates', 'chamber_IDs'])

        # Iterate over samples
        for sample in unique_samples:
            # Get indices of replicates for this sample
            sample_indices = np.where(sample_names == sample)[0]

            # Get rates for this sample
            sample_vmax = run_data.dep_var[..., 0][sample_indices]
            sample_kM = run_data.dep_var[..., 1][sample_indices]
            sample_r2 = run_data.dep_var[..., 2][sample_indices]
            sample_kcat = run_data.dep_var[..., 3][sample_indices]
            
            # Get enzyme concentrations (assuming alignment)
            sample_E_conc = enzyme_concentration_data.dep_var[..., 0][sample_indices]

            # Get chamber IDs for this sample
            sample_chamber_IDs = run_data.indep_vars.chamber_IDs[sample_indices]

            # Get replicate number by counting non-nan values in kcat
            replicate_number = np.count_nonzero(~np.isnan(sample_kcat))

            nonnan_sample_chamber_IDs = sample_chamber_IDs[~np.isnan(sample_kcat)]

            # Get average rate
            if replicate_number > 0:
                avg_kcat = np.nanmean(sample_kcat)
                avg_kM = np.nanmean(sample_kM)
                avg_E_conc = np.nanmean(sample_E_conc)
                avg_r2 = np.nanmean(sample_r2)
            else:
                avg_kcat = np.nan
                avg_kM = np.nan
                avg_E_conc = np.nan
                avg_r2 = np.nan

            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([{
                'sample': sample,
                f'avg_k_cat ({kcat_units})': avg_kcat,
                f'avg_K_M ({kM_units})': avg_kM,
                f'avg_E_conc ({E_units})': avg_E_conc,
                'avg_fit_R2': avg_r2,
                'replicates': replicate_number,
                'chamber_IDs': nonnan_sample_chamber_IDs,
            }])], ignore_index=True)

        # Save to CSV
        if file_name is None:
            file_name = f'{run_name}_sample_kcat_data.csv'
        df.to_csv(f'{output_dir}/{file_name}', index=False)

        print(f"Sample kcat data exported to {output_dir}/{file_name}")


    def export_MM_chamber_data(self, run_name: str, enzyme_concentration_run_name: str, output_dir: str = None, file_name: str = None):
        '''
        Export per-chamber kcat data to a CSV file.
        Calculates kcat = vmax / [E].

        Input:
            run_name (str): the name of the run to be exported.
            enzyme_concentration_run_name (str): the name of the run with enzyme concentration data.
            output_dir (str): the directory to save the CSV file. If None, uses current directory.
            file_name (str): the name of the file to save the CSV file. If None, uses run_name.
        Output:
            None
        '''
        run_data = self.get_run(run_name)
        enzyme_concentration_data = self.get_run(enzyme_concentration_run_name)

        assert isinstance(run_data, Data2D), "run_data must be of type Data2D."

        if output_dir is None:
            output_dir = os.getcwd()

        # Get sample names
        chamber_names = run_data.indep_vars.chamber_IDs
        sample_names = run_data.indep_vars.sample_IDs

        # Get units
        vmax_unit_obj = run_data.dep_var_units[0]
        E_unit_obj = enzyme_concentration_data.dep_var_units[0]
        kcat_unit_obj = run_data.dep_var_units[3]

        kcat_units = f"{kcat_unit_obj:~}"
        kM_units = f"{run_data.dep_var_units[1]:~}"
        E_units = f"{E_unit_obj:~}"

        # New empty dataframe:
        df = pd.DataFrame(columns=['chamber', 'sample', f'k_cat ({kcat_units})', f'K_M ({kM_units})', f'E_conc ({E_units})', 'fit_R2'])

        # Iterate over chambers
        for chamber in chamber_names:
            # Get indices of chambers for this sample
            chamber_index = np.where(chamber_names == chamber)[0]

            # Get rates for this sample
            chamber_vmax = run_data.dep_var[..., 0][chamber_index][0]
            chamber_kM = run_data.dep_var[..., 1][chamber_index][0]
            chamber_r2 = run_data.dep_var[..., 2][chamber_index][0]
            chamber_kcat = run_data.dep_var[..., 3][chamber_index][0]

            chamber_E_conc = enzyme_concentration_data.dep_var[..., 0][chamber_index][0]

            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([{
                'chamber': chamber,
                'sample': sample_names[chamber_index][0],
                f'k_cat ({kcat_units})': chamber_kcat,
                f'K_M ({kM_units})': chamber_kM,
                f'E_conc ({E_units})': chamber_E_conc,
                'fit_R2': chamber_r2,
            }])], ignore_index=True)

        # Save to CSV
        if file_name is None:
            file_name = f'{run_name}_chamber_kcat_data.csv'
        df.to_csv(f'{output_dir}/{file_name}', index=False)

        print(f"Chamber kcat data exported to {output_dir}/{file_name}")


    ### These are the old output functions - I will remove them in a future update
    # def export_run_data_raw(self, run_name: str, output_dir: str = None):
    #     '''
    #     Export per-chamber data to a CSV file, for Data2D objects.
    #     Input:
    #         run_name (str): the name of the run to be exported.
    #         output_dir (str): the directory to save the CSV file. If None, uses current directory.
    #     Output:
    #         None
    #     '''
    #     run_data = self.get_run(run_name)

    #     assert isinstance(run_data, Data2D), "run_data must be of type Data2D."

    #     if output_dir is None:
    #         output_dir = os.getcwd()
        
    #     # Create output directory if it doesn't exist
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    #     # Define the output file path
    #     output_file = Path(output_dir) / f"{run_name}_data.csv"

    #     chamber_IDs = run_data.indep_vars.chamber_IDs
    #     sample_IDs = run_data.indep_vars.sample_IDs
        
    #     # Save the data to a CSV file
    #     with open(output_file, 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         header = ['chamber_ID', 'sample_ID'] + run_data.dep_var_type
    #         writer.writerow(header)
    #         for i in range(run_data.dep_var.shape[0]):
    #             row = [chamber_IDs[i], sample_IDs[i]] + run_data.dep_var[i].tolist()
    #             writer.writerow(row)
        
    #     print(f"Run data exported to {output_file}")

    # def export_run_data_processed(self, run_name: str, output_dir: str = None):
    #     '''
    #     Export per-sample data to a CSV file, for Data2D objects.
    #     Input:
    #         run_name (str): the name of the run to be exported.
    #         output_dir (str): the directory to save the CSV file. If None, uses current directory.
    #     Output:
    #         None
    #     '''
    #     run_data = self.get_run(run_name)
    #     assert isinstance(run_data, Data2D), "run_data must be of type Data2D."
        
    #     sample_IDs = run_data.indep_vars.sample_IDs
    #     chamber_IDs = run_data.indep_vars.chamber_IDs

    #     # get the mean, std, and count for each value across each sample
    #     sample_list = np.unique(sample_IDs)
    #     sample_data = {sample: [] for sample in sample_list}
    #     for i, sample in enumerate(sample_list):
    #         sample_mask = (sample_IDs == sample)
    #         sample_data[sample] = {
    #             'chamber_IDs': chamber_IDs[sample_mask],
    #             'mean': np.nanmean(run_data.dep_var[sample_mask], axis=0),
    #             'std': np.nanstd(run_data.dep_var[sample_mask], axis=0),
    #             'count': np.sum(~np.isnan(run_data.dep_var[sample_mask]), axis=0)
    #         }

    #     if output_dir is None:
    #         output_dir = os.getcwd()
    #     # Create output directory if it doesn't exist
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
    #     # Define the output file path
    #     output_file = Path(output_dir) / f"{run_name}_processed_data.csv"
    #     # Save the data to a CSV file
    #     with open(output_file, 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         header = ['sample_ID', 'chamber_IDs'] + run_data.dep_var_type
    #         writer.writerow(header)
    #         for sample, data in sample_data.items():
    #             row = [sample, ','.join(data['chamber_IDs'].tolist())] + data['mean'].tolist()
    #             writer.writerow(row)
    #     print(f"Processed run data exported to {output_file}")

    def export_mm_subplots(self,
                           analysis_name: str,
                           model_fit_name: str,
                           export_path: str,
                           model_pred_data_name: str = None,
                           x_log: bool = False,
                           y_log: bool = False):
        '''
        Export a PDF of Michaelis-Menten subplots for each chamber in a 32x56 grid.
        
        Input:
            analysis_name (str): Name of the analysis run (Data3D).
            model_fit_name (str): Name of the fit run (Data2D).
            export_path (str): Path to save the generated PDF.
            model_pred_data_name (str): Optional name of prediction run (Data3D).
            x_log (bool): If True, use log scale for Concentration.
            y_log (bool): If True, use log scale for Initial Rate.
        '''
        # 1. Fetch Data
        analysis: Data3D = self.get_run(analysis_name)
        si = analysis.dep_var_type.index("slope")
        slope_unit = analysis.dep_var_units[si]
        slopes = analysis.dep_var[..., si] * slope_unit  # (n_conc, n_chambers)
        conc   = analysis.indep_vars.concentration       # (n_conc,)

        mf_fit: Data2D = self.get_run(model_fit_name)
        fit_types = mf_fit.dep_var_type
        fit_vals = mf_fit.dep_var
        fit_units = mf_fit.dep_var_units
        
        preds = None
        if model_pred_data_name:
            mf_pred: Data3D = self.get_run(model_pred_data_name)
            yi = mf_pred.dep_var_type.index("y_pred")
            y_pred_unit = mf_pred.dep_var_units[yi]
            preds = mf_pred.dep_var[..., yi] * y_pred_unit # (n_conc, n_chambers)

        chambers = analysis.indep_vars.chamber_IDs
        samples  = analysis.indep_vars.sample_IDs
        sample_names = {cid: samples[i] for i, cid in enumerate(chambers)}
        
        # 2. Setup Figure
        # Grid is 32 wide (x=1..32) and 56 high (y=1..56)
        nrows, ncols = 56, 32
        fig, axes = plt.subplots(nrows, ncols, figsize=(64, 112))
        
        # 3. Iterate and Plot
        # axes is (nrows, ncols)
        tqdm.write("Plotting chambers...")
        for y in tqdm(range(nrows)):
            for x in range(ncols):
                ax = axes[y, x]
                # Chamber ID is "x,y" (1-based)
                # In plot_chip, x is col index+1, y is row index+1
                cid = f"{x+1},{y+1}"
                
                if cid not in sample_names:
                    ax.axis('off')
                    continue
                
                # Check if we found the chamber in the data
                idx = (chambers == cid)
                if not np.any(idx):
                     ax.axis('off')
                     continue

                # Plot content
                x_data = conc
                y_data = slopes[:, idx].flatten()
                ax.scatter(x_data, y_data, alpha=0.7, s=15, label='data') 
                
                if preds is not None:
                     # CI Logic
                    sample = sample_names[cid]
                    same_idxs = [i for i, s in enumerate(samples) if s == sample]
                    y_all = preds[:, same_idxs]
                    
                    y_min = np.nanpercentile(y_all, 2.5, axis=1)
                    y_max = np.nanpercentile(y_all, 97.5, axis=1)
                    
                    y_p = preds[:, idx].flatten()
                    ax.plot(x_data, y_p, color='red', linewidth=1, label='fit')
                    ax.fill_between(x_data, y_min, y_max, color='gray', alpha=0.3, label='95% CI')
                
                # Fit params text
                # We need to find the index of this chamber in the fit data
                chamb_idx_arr = np.where(chambers == cid)[0]
                if len(chamb_idx_arr) > 0:
                    chamb_idx = chamb_idx_arr[0]
                    vals = fit_vals[chamb_idx]
                    txt = "\n".join(f"{nm}={v:.2f} {u:~}" for nm,v,u in zip(fit_types, vals, fit_units))
                    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                            va='top', fontsize=6, bbox=dict(boxstyle="round", fc="white", alpha=0.6))

                ax.set_title(f"{cid}: {sample_names[cid]}", fontsize=8)
                ax.set_xlabel(f"Conc ({conc.units:~})", fontsize=6)
                ax.set_ylabel(f"Rate ({slopes.units:~})", fontsize=6)
                
                if x_log: ax.set_xscale("log")
                if y_log: ax.set_yscale("log")
                
                ax.tick_params(axis='both', which='major', labelsize=6)
                
        plt.tight_layout()
        # Ensure directory exists
        tqdm.write(f"Exporting MM subplots to {export_path}")
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(export_path, dpi=100)
        plt.close(fig)
        print(f"Exported MM subplots to {export_path}")


    def export_mm_subplots_by_sample(self,
                           analysis_name: str,
                           model_fit_name: str,
                           export_path: str,
                           dep_var_label: str = 'slope',
                           model_pred_data_name: str = None,
                           x_log: bool = False,
                           y_log: bool = False):
        '''
        Export a PDF of Michaelis-Menten subplots for each SAMPLE with replicates.
        
        We'll use the following process:
        1. For each sample, fetch the replicates.
        2. Plot the average rates / [E] vs concentration with 1 std dev error bars for replicates.
        3. Get the stdev of the kcats. Plot an envelope of the fit lines with +-1 stdev kcat (with mean KM).
        
        Input:
            analysis_name (str): Name of the analysis run (Data3D).
            model_fit_name (str): Name of the fit run (Data2D).
            export_path (str): Path to save the generated PDF.
            model_pred_data_name (str): Optional name of prediction run (Data3D).
            x_log (bool): If True, use log scale for Concentration.
            y_log (bool): If True, use log scale for Initial Rate.
        '''
        # 1. Fetch Data
        analysis: Data3D = self.get_run(analysis_name)
        si = analysis.dep_var_type.index(dep_var_label)
        slope_unit = analysis.dep_var_units[si]
        slopes = analysis.dep_var[..., si] * slope_unit  # (n_conc, n_chambers)
        conc   = analysis.indep_vars.concentration       # (n_conc,)

        mf_fit: Data2D = self.get_run(model_fit_name)
        fit_types = mf_fit.dep_var_type
        fit_vals = mf_fit.dep_var
        fit_units = mf_fit.dep_var_units
        
        # Get kcat and km indices and data
        kcat_idx = mf_fit.dep_var_type.index("kcat")
        all_kcats = mf_fit.dep_var[..., kcat_idx] * fit_units[kcat_idx]   # (n_chambers,)
        kM_idx = mf_fit.dep_var_type.index("K_m")
        all_kMs = mf_fit.dep_var[..., kM_idx] * fit_units[kM_idx]   # (n_chambers,)

        chambers = analysis.indep_vars.chamber_IDs
        samples  = analysis.indep_vars.sample_IDs
        
        # Group chambers by sample
        unique_samples = np.unique(samples)
        sample_to_chambers = {s: [] for s in unique_samples}
        for i, s in enumerate(samples):
            sample_to_chambers[s].append(i) # Store indices
            
        # 2. Setup Figure
        # Determine grid size based on number of samples
        n_samples = len(unique_samples)
        
        # Let's try to make a somewhat square grid, or use the standard 8x12 plate layout if it matches
        ncols = 8
        nrows = int(np.ceil(n_samples / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
        
        tqdm.write("Plotting samples...")
        
        # 3. Iterate and Plot
        for i, sample in enumerate(tqdm(unique_samples)):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]
            
            indices = sample_to_chambers[sample] # indices of replicates in 'slopes' and 'chambers' arrays
            
            # --- Aggregating Rates ---
            # slopes is (n_conc, n_chambers)
            # we want slopes[:, indices] -> (n_conc, n_replicates)
            replicate_rates = slopes[:, indices] 
            
            mean_rates = np.nanmean(replicate_rates, axis=1) # (n_conc,)
            std_rates = np.nanstd(replicate_rates, axis=1)   # (n_conc,)
            
            # Plot error bars
            # errorbar doesn't handle quantities with units well in older mpl versions, ensuring magnitude
            if hasattr(mean_rates, 'magnitude'):
                 y = mean_rates.magnitude
                 yerr = std_rates.magnitude
                 x = conc.magnitude
            else:
                 y = mean_rates
                 yerr = std_rates
                 x = conc
            
            ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, label='mean rate', color='blue')

            # --- Aggregating Fit Parameters ---
            sample_kcats = all_kcats[indices]
            sample_kMs = all_kMs[indices]
            
            mean_kcat = np.nanmean(sample_kcats)
            mean_km = np.nanmean(sample_kMs)
            
            kcat_stdev = np.nanstd(sample_kcats)
            km_stdev = np.nanstd(sample_kMs)
            
            kcat_up = mean_kcat + kcat_stdev
            kcat_down = mean_kcat - kcat_stdev
            
            # --- Generating Prediction Lines ---
            # mm_model(S, Vmax, Km) -> here we treat Vmax as kcat because likely normalized
            # But wait, mm_model usually expects Vmax. 
            # If the user plotted "kcat" in plot_MM_div_E_chip, it implies 'slopes' are normalized.
            # So mm_model(conc, kcat, Km) should return normalized rates.
            
            pred_conc_range = np.linspace(conc.min(), conc.max(), 100)
            pred_y_mean = mm_model(pred_conc_range, mean_kcat, mean_km)
            pred_y_up = mm_model(pred_conc_range, kcat_up, mean_km)
            pred_y_down = mm_model(pred_conc_range, kcat_down, mean_km)
            
            # Ensure magnitude for plotting
            if hasattr(pred_y_mean, 'magnitude'):
                 py_mean = pred_y_mean.magnitude
                 py_up = pred_y_up.magnitude
                 py_down = pred_y_down.magnitude
            else:
                 py_mean = pred_y_mean
                 py_up = pred_y_up
                 py_down = pred_y_down
            
            ax.plot(pred_conc_range, py_mean, color="blue", label="mean fit")
            ax.fill_between(pred_conc_range, py_down, py_up, color="blue", alpha=0.2, label=r'$k_{cat} \pm 1 \sigma$')
            
            # Annotate
            if hasattr(mean_kcat, 'units'):
                kcat_str = f"{mean_kcat.magnitude:.2f} ± {kcat_stdev.magnitude:.2f} {mean_kcat.units:~}"
            else:
                kcat_str = f"{mean_kcat:.2f} ± {kcat_stdev:.2f}"
                
            if hasattr(mean_km, 'units'):
                km_str = f"{mean_km.magnitude:.2f} ± {km_stdev.magnitude:.2f} {mean_km.units:~}"
            else:
                km_str = f"{mean_km:.2f} ± {km_stdev:.2f}"
            
            ax.text(0.95, 0.05, 
                    f"$\\overline{{k_{{cat}}}}$ = {mean_kcat.magnitude:.2f} ± {kcat_stdev:.2f~}\n"
                    f"$\\overline{{K_{{M}}}}$ = {mean_km.magnitude:.2f} ± {km_stdev:.2f~}", 
                    transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8)
                    #bbox=dict(boxstyle="round", fc="white", alpha=0.7))

            ax.set_title(f"{sample}", fontsize=10)
            
            # Labels
            conc_unit = f"({conc.units:~})" if hasattr(conc, 'units') else ""
            rate_unit = f"({slopes.units:~})" if hasattr(slopes, 'units') else ""
            
            ax.set_xlabel(f"$[S]$ {conc_unit}", fontsize=8)
            ax.set_ylabel(f"$V_0/[E]$ {rate_unit}", fontsize=8)
            
            if x_log: ax.set_xscale("log")
            if y_log: ax.set_yscale("log")
            
            ax.tick_params(axis='both', which='major', labelsize=8)

        # Hide empty axes
        for j in range(i + 1, nrows * ncols):
            row = j // ncols
            col = j % ncols
            axes[row, col].axis('off')

        plt.tight_layout()
        tqdm.write(f"Exporting MM subplots by sample to {export_path}")
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(export_path, dpi=100)
        plt.close(fig)
        print(f"Exported MM subplots by sample to {export_path}")