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
from htbam_analysis.db_api.htbam_db_api import AbstractHtbamDBAPI, HtbamDBException
from htbam_analysis.db_api.data import Data4D, Data3D, Data2D, Meta
from htbam_analysis.analysis.plot import plot_chip
from htbam_analysis.analysis.fit import mm_model, binding_isotherm_model
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
        df = pd.DataFrame(columns=['sample', 
                                   f'avg_k_cat ({kcat_units})', f'std_k_cat ({kcat_units})',
                                   f'avg_K_M ({kM_units})', f'std_K_M ({kM_units})',
                                   f'avg_E_conc ({E_units})', f'std_E_conc ({E_units})',
                                   'avg_fit_R2', 'std_fit_R2',
                                   'replicates', 'chamber_IDs'])

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
                std_kcat = np.nanstd(sample_kcat)
                avg_kM = np.nanmean(sample_kM)
                std_kM = np.nanstd(sample_kM)
                avg_E_conc = np.nanmean(sample_E_conc)
                std_E_conc = np.nanstd(sample_E_conc)
                avg_r2 = np.nanmean(sample_r2)
                std_r2 = np.nanstd(sample_r2)
            else:
                avg_kcat = np.nan
                std_kcat = np.nan
                avg_kM = np.nan
                std_kM = np.nan
                avg_E_conc = np.nan
                std_E_conc = np.nan
                avg_r2 = np.nan
                std_r2 = np.nan

            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([{
                'sample': sample,
                f'avg_k_cat ({kcat_units})': avg_kcat,
                f'std_k_cat ({kcat_units})': std_kcat,
                f'avg_K_M ({kM_units})': avg_kM,
                f'std_K_M ({kM_units})': std_kM,
                f'avg_E_conc ({E_units})': avg_E_conc,
                f'std_E_conc ({E_units})': std_E_conc,
                'avg_fit_R2': avg_r2,
                'std_fit_R2': std_r2,
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
        vmax_units = f"{vmax_unit_obj:~}"

        # New empty dataframe:
        df = pd.DataFrame(columns=['chamber', 'sample', f'v_max ({vmax_units})', f'k_cat ({kcat_units})', f'K_M ({kM_units})', f'E_conc ({E_units})', 'fit_R2'])

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
                f'v_max ({vmax_units})': chamber_vmax,
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

    def export_mm_subplots_by_chamber(self,
                           analysis_name: str,
                           model_fit_name: str,
                           export_path: str,
                           dep_var_label: str = 'slope',
                           model_pred_data_name: str = None,
                           dpi: int = 100,
                           x_log: bool = False,
                           y_log: bool = False):
        '''
        Export a PDF of Michaelis-Menten subplots for each chamber in a 32x56 grid.
        Plots a 95% confidence interval over the replicates for that chamber in gray.
        
        Input:
            analysis_name (str): Name of the analysis run (Data3D).
            model_fit_name (str): Name of the fit run (Data2D).
            export_path (str): Path to save the generated PDF. Also can be a list of paths (if you want to save a PNG and PDF, for example)
            dep_var_label (str): Label of the dependent variable to plot (default "slope").
            model_pred_data_name (str): Optional name of prediction run (Data3D).
            dpi (int): DPI for the exported image.
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
        header_height = 2.5  # inches
        total_height = 112 + header_height
        fig, axes = plt.subplots(nrows, ncols, figsize=(64, total_height))
        
        # Add description text at the top
        description_text = (
            "Michaelis-Menten Fit Subplots by Chamber\n\n"
            "This PDF displays the Michaelis-Menten fit subplots for each chamber across the 32x56 grid.\n"
            "• Blue Circle Markers: Represent the experimental data points (slopes) at each substrate concentration.\n"
            "• Solid Red Line: Represents the best-fit Michaelis-Menten curve calculated for that specific chamber.\n"
            "• Gray Shaded Envelope: Represents the 95% confidence interval of the fit curves across all replicates of the same sample.\n"
            "• Inset Text Box: Lists the calculated fit parameters (e.g., $k_{cat}$, $K_M$) with their corresponding values and units."
        )
        
        fig.text(0.5, 1.0 - (0.5 / total_height), description_text,
                 ha='center', va='top', fontsize=10, multialignment='left',
                 bbox=dict(boxstyle="round,pad=0.8", fc="white", ec="silver", lw=1))
        
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

                # Robust unit handling for plotting
                if hasattr(x_data, 'magnitude'): x_data = x_data.magnitude
                if hasattr(y_data, 'magnitude'): y_data = y_data.magnitude
                
                chamb_idx_arr = np.where(chambers == cid)[0]
                chamb_idx = chamb_idx_arr[0]
                vals = fit_vals[chamb_idx]

                ax.set_title(f"{cid}: {sample_names[cid]}", fontsize=8)
                ax.set_xlabel(f"Conc ({conc.units:~})", fontsize=6)
                ax.set_ylabel(f"Rate ({slopes.units:~})", fontsize=6)

                if preds is None or np.isnan(vals).all():
                    continue

                ax.scatter(x_data, y_data, alpha=0.7, s=15, label='data') 

                # CI Logic
                sample = sample_names[cid]
                same_idxs = [i for i, s in enumerate(samples) if s == sample]
                y_all = preds[:, same_idxs]
                
                y_min = np.nanpercentile(y_all, 2.5, axis=1)
                y_max = np.nanpercentile(y_all, 97.5, axis=1)
                y_p = preds[:, idx].flatten()

                # Robust unit handling for predictions
                if hasattr(y_p, 'magnitude'): y_p = y_p.magnitude
                if hasattr(y_min, 'magnitude'): y_min = y_min.magnitude
                if hasattr(y_max, 'magnitude'): y_max = y_max.magnitude
                
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
                
        top_fraction = 1.0 - (header_height / total_height)
        plt.tight_layout(rect=[0, 0, 1, top_fraction])
        # Ensure directory exists
        tqdm.write(f"Exporting MM subplots to {export_path}")

        if type(export_path) == str:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(export_path, dpi=dpi)
        elif type(export_path) == list:
            for path in export_path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"Exported MM subplots to {export_path}")


    def export_mm_subplots_by_sample(self,
                           analysis_name: str,
                           model_fit_name: str,
                           export_path: str,
                           dep_var_label: str = 'slope',
                           model_pred_data_name: str = None,
                           hide_excluded_samples: bool = False,
                           dpi: int = 100,
                           aspect_ratio: float = 1.5,
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
            export_path (str): Path to save the generated PDF. Can also be a list of paths (if you want to save a PNG and PDF, for example)
            model_pred_data_name (str): Optional name of prediction run (Data3D).
            hide_excluded_samples (bool): If True, exclude samples with missing replicates.
            dpi (int): DPI for the exported image.
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
        ncols = 3
        nrows = int(np.ceil(n_samples / ncols))
        
        # Reserve height at the top for description text
        header_height = 2.5  # inches
        plot_height = (4/aspect_ratio)*nrows
        total_height = plot_height + header_height
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, total_height), squeeze=False)
        # Set aspect ratio to 1.25:1, which allows for 3x5 plots on a 8.5/11 page:
        for ax in axes.flatten():
            ax.set_box_aspect(1/aspect_ratio)   
            
        # Add description text at the top
        description_text = (
            "Michaelis-Menten Fit Subplots by Sample\n\n"
            "This PDF displays the Michaelis-Menten fit subplots for each sample across its replicates.\n"
            "• Blue Circle Markers & Error Bars: Represent the mean initial rate normalized by enzyme concentration ($V_0 / [E]$) across replicates\n"
            "  at each substrate concentration ($[S]$), with error bars indicating $\\pm 1$ standard deviation.\n"
            "• Solid Blue Line: Represents the best-fit Michaelis-Menten curve calculated using the mean $k_{cat}$ and mean $K_M$ values\n"
            "  across replicates.\n"
            "• Light Blue Shaded Envelope: Represents the $\\pm 1$ standard deviation interval of the fit curves, generated by varying $k_{cat}$\n"
            "  by $\\pm 1$ standard deviation ($k_{cat} \\pm 1 \\sigma$) while keeping $K_M$ at its mean value.\n"
            "• Text Annotation: Shows the calculated average $k_{cat}$ and $K_M$ values along with their standard deviations\n"
            "  across all valid replicates."
        )
        
        fig.text(0.5, 1.0 - (0.5 / total_height), description_text,
                 ha='center', va='top', fontsize=10, multialignment='left',
                 bbox=dict(boxstyle="round,pad=0.8", fc="white", ec="silver", lw=1))
        
        tqdm.write("Plotting samples...")
        
        # 3. Iterate and Plot
        skipped_samples = []
        for i, sample in enumerate(tqdm(unique_samples)):
            i = i - len(skipped_samples)
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

            # --- Aggregating Fit Parameters ---
            sample_kcats = all_kcats[indices]
            sample_kMs = all_kMs[indices]
            
            mean_kcat = np.nanmean(sample_kcats)
            mean_km = np.nanmean(sample_kMs)
            
            kcat_stdev = np.nanstd(sample_kcats)
            km_stdev = np.nanstd(sample_kMs)
            
            kcat_up = mean_kcat + kcat_stdev
            kcat_down = mean_kcat - kcat_stdev

            if hide_excluded_samples:
                if np.isnan(mean_kcat.magnitude):
                    print(f"Sample {sample} has no (non-NaN) replicates. Skipping.")
                    skipped_samples.append(sample)
                    continue

            # Plotting rates
            ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, label='mean rate', color='blue')
            
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

        top_fraction = 1.0 - (header_height / total_height)
        plt.tight_layout(rect=[0, 0, 1, top_fraction])
        tqdm.write(f"Exporting MM subplots by sample to {export_path}")
        
        if type(export_path) == str:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(export_path, dpi=dpi)
        elif type(export_path) == list:
            for path in export_path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(path, dpi=dpi)
        
        plt.close(fig)
        print(f"Exported MM subplots by sample to {export_path}")

    
    def export_end_to_end_summary_by_sample(self,
                           export_dir: str = 'end_to_end_summaries',
                           sample_ids: list = None,
                           button_quant_csv: str = 'button_quant.csv',
                           standard_curve_csv: str = 'standard_data.csv.bz2',
                           kinetics_csv: str = 'kinetics_data.csv.bz2',
                           enzyme_conc_run: str = 'enzyme_concentrations',
                           kinetics_fits_run: str = 'kinetics_ADP_conc_fits_bgsub',
                           kinetics_raw_run: str = 'kinetics_ADP_conc',
                           mm_fits_run: str = 'masked_fits_with_kcat_filtered',
                           mm_pred_run: str = 'pred_data_div_E',
                           mask_runs: dict = None,
                           dpi: int = 150):
        '''
        Export a PDF of end-to-end processing for each SAMPLE with replicates.
        '''
        import tifffile
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import ast
        import pandas as pd
        from tqdm import tqdm
        
        # Load CSVs to DataFrames
        tqdm.write("Loading CSVs for stamp images...")
        bq_df = pd.read_csv(button_quant_csv)
        sc_df = pd.read_csv(standard_curve_csv)
        kin_df = pd.read_csv(kinetics_csv)

        # 1. Fetch Data
        enzyme_conc = self.get_run(enzyme_conc_run)
        kinetics_fits = self.get_run(kinetics_fits_run)
        kinetics_raw = self.get_run(kinetics_raw_run)
        mf_fit = self.get_run(mm_fits_run)
        
        # Data objects
        time_points = kinetics_raw.indep_vars.time
        sc_conc = self.get_run('NADPH_standard').indep_vars.concentration
        raw_rates = kinetics_raw.dep_var[..., 0] 
        conc = kinetics_fits.indep_vars.concentration
        slopes = kinetics_fits.dep_var[..., kinetics_fits.dep_var_type.index('slope')]

        vmax_idx = mf_fit.dep_var_type.index('v_max')
        kcat_idx = mf_fit.dep_var_type.index('kcat')
        km_idx = mf_fit.dep_var_type.index('K_m')

        all_kcats = mf_fit.dep_var[..., kcat_idx]
        all_kms = mf_fit.dep_var[..., km_idx]

        if mm_pred_run:
            mf_pred = self.get_run(mm_pred_run)
            yi = 0
            preds = mf_pred.dep_var[..., yi]
        else:
            preds = None

        # Masks
        mask_data = {}
        if mask_runs:
            for m_name in mask_runs.keys():
                mask_run = self.get_run(m_name)
                # Assumes mask is DataND with dep_var_type ['mask']
                mask_data[m_name] = mask_run.dep_var[..., 0].astype(bool)

        chambers = enzyme_conc.indep_vars.chamber_IDs
        samples = enzyme_conc.indep_vars.sample_IDs

        unique_samples = np.unique(samples)
        if sample_ids:
            unique_samples = [s for s in unique_samples if s in sample_ids]

        # Caching for TIFFs
        tiff_cache = {}
        def get_stamp(df, x_idx, y_idx, time_index=None, sc_conc_val=None):
            # Find the row
            mask = (df['x'] == x_idx) & (df['y'] == y_idx)
            if time_index is not None:
                if 'time_index' in df.columns:
                    mask = mask & (df['time_index'] == time_index)
                else:
                    mask = mask & (df['index'] == time_index)
            if sc_conc_val is not None:
                # need to find closest conc if floating point issues
                conc_col = 'concentration' if 'concentration' in df.columns else 'LMET_conc'
                # Assuming concentration matches directly, or string match
                mask = mask & (df['time_index'] == sc_conc_val) # wait! Standard curve CSV time_index holds the concentration index!
                
            sub = df[mask]
            if len(sub) == 0:
                return None
            row = sub.iloc[0]
            img_path = row['image_path']
            
            # replace bgsub_images with summary_images/bgsub_images
            img_path = img_path.replace('bgsub_images', 'summary_images/bgsub_images')
            
            # resolve relative to experiment root
            parts = Path(img_path).parts
            if 'summary_images' in parts:
                idx = parts.index('summary_images')
                rel_img_path = Path(*parts[idx:])
                abs_path = (Path(button_quant_csv).parent / rel_img_path).resolve()
            else:
                abs_path = (Path(button_quant_csv).parent / img_path).resolve()
            
            if abs_path not in tiff_cache:
                try:
                    tiff_cache[abs_path] = tifffile.memmap(str(abs_path))
                except Exception as e:
                    return None
                    
            img = tiff_cache[abs_path]
            
            y_start = (y_idx - 1) * 102 + 1
            y_end = y_start + 100
            x_start = (x_idx - 1) * 102 + 1
            x_end = x_start + 100
            
            return img[y_start:y_end, x_start:x_end]

        tqdm.write("Plotting end-to-end summaries...")
        for sample in tqdm(unique_samples):
            # Find chambers for this sample
            sample_mask = (samples == sample)
            chamber_indices = np.where(sample_mask)[0]
            if len(chamber_indices) == 0:
                continue

            n_reps = len(chamber_indices)
            
            # Sample level kcat stdev
            sample_kcats = all_kcats[chamber_indices]
            mean_kcat = np.nanmean(sample_kcats)
            std_kcat = np.nanstd(sample_kcats)
            mean_km = np.nanmean(all_kms[chamber_indices])

            max_conc = len(conc)
            sc_num = len(sc_conc)
            try:
                max_timepoints = max(len(t) for t in time_points)
            except Exception:
                max_timepoints = len(time_points)
            
            ncols = 5
            
            bq_aspect = 1.0
            sc_aspect = 1.0 / sc_num if sc_num > 0 else 1.0
            kin_aspect = max_timepoints / max_conc if max_conc > 0 else 1.0
            rates_aspect = 1.5
            mm_aspect = float(n_reps)
            
            width_ratios = [bq_aspect, sc_aspect, kin_aspect, rates_aspect, mm_aspect]
            total_aspect = sum(width_ratios)
            
            fig = plt.figure(figsize=(total_aspect * 2.5 * 1.15, n_reps * 2.5))
            gs = gridspec.GridSpec(n_reps, ncols, figure=fig, width_ratios=width_ratios, wspace=0.3, hspace=0.4)
            
            # Extract units for plotting
            p_unit = ""
            if hasattr(kinetics_raw, 'dep_var_units') and len(kinetics_raw.dep_var_units) > 0:
                pu = kinetics_raw.dep_var_units[0]
                p_unit = f" ({pu:~})" if hasattr(pu, '__format__') else f" ({pu})"
            
            t_unit = ""
            if hasattr(time_points, 'units'):
                t_unit = f" ({time_points.units:~})"
            elif hasattr(time_points, '__len__') and len(time_points) > 0 and hasattr(time_points[0], 'units'):
                t_unit = f" ({time_points[0].units:~})"

            # --- Setup Shared MM Plot ---
            ax_mm = fig.add_subplot(gs[:, 4])
            x_mm = conc
            if hasattr(x_mm, "magnitude"): x_mm = x_mm.magnitude
            x_range = np.linspace(0, max(x_mm), 100) if len(x_mm) > 0 else []
            from htbam_analysis.analysis.fit import mm_model
            if not np.isnan(mean_kcat) and not np.isnan(mean_km) and len(x_range) > 0:
                mean_kcat_m = mean_kcat.magnitude if hasattr(mean_kcat, "magnitude") else mean_kcat
                mean_km_m = mean_km.magnitude if hasattr(mean_km, "magnitude") else mean_km
                std_kcat_m = std_kcat.magnitude if hasattr(std_kcat, "magnitude") else std_kcat
                y_mean = mm_model(x_range, mean_kcat_m, mean_km_m)
                y_up = mm_model(x_range, mean_kcat_m + std_kcat_m, mean_km_m)
                y_down = mm_model(x_range, mean_kcat_m - std_kcat_m, mean_km_m)
                ax_mm.fill_between(x_range, y_down, y_up, color="gray", alpha=0.3, label="Sample $\pm 1\sigma$ kcat")
                ax_mm.plot(x_range, y_mean, "k--", label="Sample Fit")
            c_unit = f" ({conc.units:~})" if hasattr(conc, 'units') else ""
            v0_run = self.get_run("masked_V0_div_E_vs_conc")
            v0_unit = ""
            if hasattr(v0_run, 'dep_var_units') and len(v0_run.dep_var_units) > 0:
                vu = v0_run.dep_var_units[0]
                v0_unit = f" ({vu:~})" if hasattr(vu, '__format__') else f" ({vu})"
                
            ax_mm.set_xlabel(f"[S]{c_unit}", fontsize=10)
            ax_mm.set_ylabel(f"$V_0/[E]${v0_unit}", fontsize=10)
            ax_mm.set_title("Michaelis-Menten", fontsize=12)
            ax_mm.tick_params(labelsize=8)
            
            if not np.isnan(mean_kcat) and not np.isnan(mean_km):
                kcat_u = mf_fit.dep_var_units[kcat_idx] if hasattr(mf_fit, 'dep_var_units') and len(mf_fit.dep_var_units) > kcat_idx else ""
                kcat_u = f" {kcat_u:~}" if hasattr(kcat_u, '__format__') else (f" {kcat_u}" if kcat_u else "")
                
                km_u = mf_fit.dep_var_units[km_idx] if hasattr(mf_fit, 'dep_var_units') and len(mf_fit.dep_var_units) > km_idx else ""
                km_u = f" {km_u:~}" if hasattr(km_u, '__format__') else (f" {km_u}" if km_u else "")
                
                text_str = f"$k_{{cat}}$: {mean_kcat_m:.2f} $\\pm$ {std_kcat_m:.2f}{kcat_u}\n$K_M$: {mean_km_m:.2f}{km_u}"
                ax_mm.text(0.05, 0.95, text_str, transform=ax_mm.transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))

            # --- First pass: gather stamps to compute global normalizations ---
            sample_data = []
            bq_all_pixels = []
            sc_all_pixels = []
            kin_all_pixels = []

            for row_i, c_idx in enumerate(chamber_indices):
                cid = chambers[c_idx]
                x_idx, y_idx = map(int, cid.split(','))
                
                # Fetch BQ
                stamp_bq = get_stamp(bq_df, x_idx, y_idx)
                
                # Fetch SC
                sc_stamps = []
                for ti in range(len(sc_conc)):
                    st = get_stamp(sc_df, x_idx, y_idx, time_index=ti)
                    if st is not None and st.shape == (100, 100):
                        sc_stamps.append(st)
                stacked_sc = np.vstack(sc_stamps) if sc_stamps else None

                # Fetch KIN
                kin_grid_rows = []
                for conc_i in range(len(conc)):
                    kin_row_stamps = []
                    for ti in range(max_timepoints):
                        mask = (kin_df['x'] == x_idx) & (kin_df['y'] == y_idx) & (kin_df['time_index'] == ti)
                        sub = kin_df[mask & (kin_df['index'] == conc_i)]
                        st = None
                        if len(sub) > 0:
                            row = sub.iloc[0]
                            img_path = row['image_path'].replace('bgsub_images', 'summary_images/bgsub_images')
                            parts = Path(img_path).parts
                            if 'summary_images' in parts:
                                idx_part = parts.index('summary_images')
                                rel_img_path = Path(*parts[idx_part:])
                                abs_path = (Path(button_quant_csv).parent / rel_img_path).resolve()
                            else:
                                abs_path = (Path(button_quant_csv).parent / img_path).resolve()
                            if abs_path not in tiff_cache:
                                try:
                                    tiff_cache[abs_path] = tifffile.memmap(str(abs_path))
                                except Exception:
                                    pass
                            if abs_path in tiff_cache:
                                img = tiff_cache[abs_path]
                                y_start = (y_idx - 1) * 102 + 1
                                y_end = y_start + 100
                                x_start = (x_idx - 1) * 102 + 1
                                x_end = x_start + 100
                                st = img[y_start:y_end, x_start:x_end]
                        if st is not None and st.shape == (100, 100):
                            kin_row_stamps.append(st)
                        else:
                            kin_row_stamps.append(np.zeros((100, 100))) # placeholder
                            
                    if kin_row_stamps:
                        kin_grid_rows.append(np.hstack(kin_row_stamps))
                
                kin_grid = np.vstack(kin_grid_rows) if kin_grid_rows else None
                
                sample_data.append({
                    'c_idx': c_idx, 'cid': cid, 'x_idx': x_idx, 'y_idx': y_idx,
                    'bq': stamp_bq, 'sc': stacked_sc, 'kin': kin_grid
                })

                # Accumulate valid pixels for robust normalization (ignoring zero-padding and annotations > 55000)
                def add_valid(img, pixel_list):
                    if img is not None:
                        mask = (img > 0) & (img < 55000)
                        if np.any(mask):
                            p = img[mask]
                            if len(p) > 10000:
                                p = np.random.choice(p, 10000, replace=False)
                            pixel_list.append(p)

                add_valid(stamp_bq, bq_all_pixels)
                add_valid(stacked_sc, sc_all_pixels)
                add_valid(kin_grid, kin_all_pixels)

            def get_bounds(pixel_list):
                if not pixel_list: return 0, 65535
                arr = np.concatenate(pixel_list)
                if len(arr) == 0: return 0, 65535
                return np.percentile(arr, 1), np.percentile(arr, 99)

            bq_vmin, bq_vmax = get_bounds(bq_all_pixels)
            sc_vmin, sc_vmax = get_bounds(sc_all_pixels)
            kin_vmin, kin_vmax = get_bounds(kin_all_pixels)

            # --- Second pass: Plotting ---
            for row_i, data in enumerate(sample_data):
                c_idx, cid = data['c_idx'], data['cid']
                x_idx, y_idx = data['x_idx'], data['y_idx']
                
                # Check global masks for this chamber
                chamber_filtered_msg = None
                for m_name, m_msg in mask_runs.items():
                    m_arr = mask_data[m_name]
                    if m_arr.shape == (len(chambers),):
                        if not m_arr[c_idx]:
                            chamber_filtered_msg = m_msg
                            break
                    elif m_arr.ndim == 2:
                        if not np.any(m_arr[:, c_idx]):
                            chamber_filtered_msg = f"Excluded: All Conc Filtered ({m_msg})"
                            break

                # Col 0: Button Quant
                ax_bq = fig.add_subplot(gs[row_i, 0])
                if data['bq'] is not None and data['bq'].shape == (100, 100):
                    ax_bq.imshow(data['bq'], cmap='gray', vmin=bq_vmin, vmax=bq_vmax)
                ax_bq.axis('off')
                
                e_c = enzyme_conc.dep_var[c_idx, 0]
                e_unit = enzyme_conc.dep_var_units[0]
                ax_bq.set_title(f"Rep {row_i+1}\n{cid}\n[E]: {e_c:.2f} {e_unit:~}", fontsize=8)

                # Col 1: Standard Curve
                ax_sc = fig.add_subplot(gs[row_i, 1])
                if data['sc'] is not None:
                    ax_sc.imshow(data['sc'], cmap='gray', vmin=sc_vmin, vmax=sc_vmax)
                ax_sc.axis('off')
                ax_sc.set_title("Std Curve", fontsize=8)

                # Col 2: Kinetics
                ax_kin = fig.add_subplot(gs[row_i, 2])
                if data['kin'] is not None:
                    ax_kin.imshow(data['kin'], cmap='gray', vmin=kin_vmin, vmax=kin_vmax)
                ax_kin.axis('off')
                ax_kin.set_title("Kinetics (t $\\rightarrow$, [S] $\\downarrow$)", fontsize=8)

                # Col 3: Initial Rates
                ax_rates = fig.add_subplot(gs[row_i, 3])
                
                colors = plt.cm.viridis(np.linspace(0, 1, max_conc))
                
                for conc_i in range(max_conc):
                    # plot raw rates [P] vs time
                    x_t = time_points[conc_i]
                    if hasattr(x_t, 'magnitude'): x_t = x_t.magnitude
                    
                    y_p = raw_rates[conc_i, :, c_idx]
                    if hasattr(y_p, 'magnitude'): y_p = y_p.magnitude
                    
                    # Determine if this conc was filtered
                    conc_filtered = False
                    conc_filter_msg = ""
                    for m_name, m_msg in mask_runs.items():
                        m_arr = mask_data[m_name]
                        if m_arr.ndim == 2: # (conc, chamber)
                            if not m_arr[conc_i, c_idx]:
                                conc_filtered = True
                                conc_filter_msg = m_msg
                                break
                    
                    c = 'red' if conc_filtered else colors[conc_i]
                    ls = '--' if conc_filtered else '-'
                    
                    ax_rates.scatter(x_t, y_p, color=c, s=5, alpha=0.5)
                    
                    # fit line: slope * x + intercept
                    # we don't have intercept easily? Let's just plot from 0 to max(x_t)
                    sl = slopes[conc_i, c_idx]
                    if hasattr(sl, 'magnitude'): sl = sl.magnitude
                    
                    if not np.isnan(sl):
                        # intercept = y_p[0] roughly, actually we have fit_concentration_vs_time 
                        # Let's just draw a line from origin with `slope`?
                        # Initial rate fit might have intercept, but we don't have it loaded.
                        # Let's use y_p[1] - slope*x_t[1] as intercept approx
                        if len(y_p) > 1 and not np.isnan(y_p[1]):
                            interc = y_p[1] - sl * x_t[1]
                            ax_rates.plot(x_t, sl * x_t + interc, color=c, ls=ls, lw=1)
                            
                    if conc_filtered and not chamber_filtered_msg:
                        # annotate next to the last point
                        ax_rates.text(x_t[-1], y_p[-1], conc_filter_msg, color='red', fontsize=6)
                
                if chamber_filtered_msg:
                    # Draw red box and text
                    for spine in ax_rates.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)
                    ax_rates.text(0.5, 0.5, f"CHAMBER EXCLUDED:\n{chamber_filtered_msg}", 
                                  color='red', fontsize=10, fontweight='bold',
                                  ha='center', va='center', transform=ax_rates.transAxes,
                                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

                ax_rates.set_xlabel(f"Time{t_unit}", fontsize=6)
                ax_rates.set_ylabel(f"[P]{p_unit}", fontsize=6)
                ax_rates.set_title("Initial Rates", fontsize=8)
                ax_rates.tick_params(labelsize=6)

                # MM Plot additions
                if not chamber_filtered_msg:
                    v0_div_e = self.get_run("masked_V0_div_E_vs_conc").dep_var[..., 0][:, c_idx]
                    if hasattr(v0_div_e, "magnitude"): v0_div_e = v0_div_e.magnitude
                    ax_mm.scatter(x_mm, v0_div_e, color="blue", s=15, zorder=5, alpha=0.5)
                    if preds is not None:
                        chamber_pred = preds[:, c_idx]
                        if hasattr(chamber_pred, "magnitude"): chamber_pred = chamber_pred.magnitude
                        ax_mm.plot(x_mm, chamber_pred, color="blue", linewidth=0.5, alpha=0.5)
            if ax_mm.get_legend_handles_labels()[1]:
                ax_mm.legend(fontsize=8)

            Path(export_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(export_dir) / f"{sample}.pdf"
            plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            tqdm.write(f"Saved {out_path}")

    def export_binding_sample_data(
        self,
        run_name: str,
        enzyme_concentration_run_name: str,
        output_dir: str = None,
        file_name: str = None,
    ):
        '''
        Export per-sample binding isotherm fit data to a CSV file.
        '''
        run_data = self.get_run(run_name)
        enzyme_concentration_data = self.get_run(enzyme_concentration_run_name)
        assert isinstance(run_data, Data2D), "run_data must be of type Data2D."

        if output_dir is None:
            output_dir = os.getcwd()

        sample_names = run_data.indep_vars.sample_IDs
        unique_samples = np.unique(sample_names)

        r_max_units = f"{run_data.dep_var_units[0]:~}"
        kd_units = f"{run_data.dep_var_units[1]:~}"
        E_units = f"{enzyme_concentration_data.dep_var_units[0]:~}"

        df = pd.DataFrame(columns=[
            'sample',
            f'avg_r_max ({r_max_units})', f'std_r_max ({r_max_units})',
            f'avg_Kd ({kd_units})', f'std_Kd ({kd_units})',
            f'avg_E_conc ({E_units})', f'std_E_conc ({E_units})',
            'avg_fit_R2', 'std_fit_R2',
            'replicates', 'chamber_IDs',
        ])

        for sample in unique_samples:
            sample_indices = np.where(sample_names == sample)[0]
            sample_rmax = run_data.dep_var[..., 0][sample_indices]
            sample_kd = run_data.dep_var[..., 1][sample_indices]
            sample_r2 = run_data.dep_var[..., 2][sample_indices]
            sample_E_conc = enzyme_concentration_data.dep_var[..., 0][sample_indices]
            sample_chamber_IDs = run_data.indep_vars.chamber_IDs[sample_indices]

            replicate_number = np.count_nonzero(~np.isnan(sample_kd))
            nonnan_sample_chamber_IDs = sample_chamber_IDs[~np.isnan(sample_kd)]

            if replicate_number > 0:
                avg_rmax = np.nanmean(sample_rmax)
                std_rmax = np.nanstd(sample_rmax)
                avg_kd = np.nanmean(sample_kd)
                std_kd = np.nanstd(sample_kd)
                avg_E_conc = np.nanmean(sample_E_conc)
                std_E_conc = np.nanstd(sample_E_conc)
                avg_r2 = np.nanmean(sample_r2)
                std_r2 = np.nanstd(sample_r2)
            else:
                avg_rmax = std_rmax = avg_kd = std_kd = avg_E_conc = std_E_conc = avg_r2 = std_r2 = np.nan

            df = pd.concat([df, pd.DataFrame([{
                'sample': sample,
                f'avg_r_max ({r_max_units})': avg_rmax,
                f'std_r_max ({r_max_units})': std_rmax,
                f'avg_Kd ({kd_units})': avg_kd,
                f'std_Kd ({kd_units})': std_kd,
                f'avg_E_conc ({E_units})': avg_E_conc,
                f'std_E_conc ({E_units})': std_E_conc,
                'avg_fit_R2': avg_r2,
                'std_fit_R2': std_r2,
                'replicates': replicate_number,
                'chamber_IDs': nonnan_sample_chamber_IDs,
            }])], ignore_index=True)

        if file_name is None:
            file_name = f'{run_name}_sample_binding_data.csv'
        df.to_csv(f'{output_dir}/{file_name}', index=False)
        print(f"Sample binding data exported to {output_dir}/{file_name}")

    def export_binding_chamber_data(
        self,
        run_name: str,
        enzyme_concentration_run_name: str,
        output_dir: str = None,
        file_name: str = None,
    ):
        '''
        Export per-chamber binding isotherm fit data to a CSV file.
        '''
        run_data = self.get_run(run_name)
        enzyme_concentration_data = self.get_run(enzyme_concentration_run_name)
        assert isinstance(run_data, Data2D), "run_data must be of type Data2D."

        if output_dir is None:
            output_dir = os.getcwd()

        chamber_names = run_data.indep_vars.chamber_IDs
        sample_names = run_data.indep_vars.sample_IDs

        r_max_units = f"{run_data.dep_var_units[0]:~}"
        kd_units = f"{run_data.dep_var_units[1]:~}"
        E_units = f"{enzyme_concentration_data.dep_var_units[0]:~}"

        df = pd.DataFrame(columns=[
            'chamber', 'sample',
            f'r_max ({r_max_units})', f'Kd ({kd_units})',
            f'E_conc ({E_units})', 'fit_R2',
        ])

        for chamber in chamber_names:
            chamber_index = np.where(chamber_names == chamber)[0]
            chamber_rmax = run_data.dep_var[..., 0][chamber_index][0]
            chamber_kd = run_data.dep_var[..., 1][chamber_index][0]
            chamber_r2 = run_data.dep_var[..., 2][chamber_index][0]
            chamber_E_conc = enzyme_concentration_data.dep_var[..., 0][chamber_index][0]

            df = pd.concat([df, pd.DataFrame([{
                'chamber': chamber,
                'sample': sample_names[chamber_index][0],
                f'r_max ({r_max_units})': chamber_rmax,
                f'Kd ({kd_units})': chamber_kd,
                f'E_conc ({E_units})': chamber_E_conc,
                'fit_R2': chamber_r2,
            }])], ignore_index=True)

        if file_name is None:
            file_name = f'{run_name}_chamber_binding_data.csv'
        df.to_csv(f'{output_dir}/{file_name}', index=False)
        print(f"Chamber binding data exported to {output_dir}/{file_name}")

    def export_binding_subplots_by_chamber(
        self,
        analysis_name: str,
        model_fit_name: str,
        export_path: str,
        dep_var_label: str = 'fluorescence_ratio',
        model_pred_data_name: str = None,
        dpi: int = 100,
        x_log: bool = False,
        y_log: bool = False,
    ):
        '''
        Export a PDF of binding isotherm subplots for each chamber in a 32x56 grid.
        '''
        analysis: Data3D = self.get_run(analysis_name)
        ri = analysis.dep_var_type.index(dep_var_label)
        ratio_unit = analysis.dep_var_units[ri]
        ratios = analysis.dep_var[..., ri] * ratio_unit  # (n_conc, n_chambers)
        conc = analysis.indep_vars.concentration

        mf_fit: Data2D = self.get_run(model_fit_name)
        fit_types = mf_fit.dep_var_type
        fit_vals = mf_fit.dep_var
        fit_units = mf_fit.dep_var_units

        preds = None
        if model_pred_data_name:
            mf_pred: Data3D = self.get_run(model_pred_data_name)
            yi = mf_pred.dep_var_type.index("y_pred")
            y_pred_unit = mf_pred.dep_var_units[yi]
            preds = mf_pred.dep_var[..., yi] * y_pred_unit

        chambers = analysis.indep_vars.chamber_IDs
        samples = analysis.indep_vars.sample_IDs
        sample_names = {cid: samples[i] for i, cid in enumerate(chambers)}

        nrows, ncols = 56, 32
        header_height = 2.5
        total_height = 112 + header_height
        fig, axes = plt.subplots(nrows, ncols, figsize=(64, total_height))

        description_text = (
            "Binding Isotherm Subplots by Chamber\n\n"
            "This PDF displays binding isotherm subplots for each chamber across the 32x56 grid.\n"
            "• Blue Circle Markers: Experimental fluorescence ratio at each ligand concentration.\n"
            "• Solid Red Line: Best-fit binding isotherm for that chamber.\n"
            "• Gray Shaded Envelope: 95% confidence interval across replicates of the same sample.\n"
            "• Inset Text Box: Fitted r_max, Kd, and R² values."
        )
        fig.text(
            0.5, 1.0 - (0.5 / total_height), description_text,
            ha='center', va='top', fontsize=10, multialignment='left',
            bbox=dict(boxstyle="round,pad=0.8", fc="white", ec="silver", lw=1),
        )

        tqdm.write("Plotting binding chambers...")
        for y in tqdm(range(nrows)):
            for x in range(ncols):
                ax = axes[y, x]
                cid = f"{x+1},{y+1}"

                if cid not in sample_names:
                    ax.axis('off')
                    continue

                idx = (chambers == cid)
                if not np.any(idx):
                    ax.axis('off')
                    continue

                x_data = conc
                y_data = ratios[:, idx].flatten()
                if hasattr(x_data, 'magnitude'):
                    x_data = x_data.magnitude
                if hasattr(y_data, 'magnitude'):
                    y_data = y_data.magnitude

                chamb_idx_arr = np.where(chambers == cid)[0]
                chamb_idx = chamb_idx_arr[0]
                vals = fit_vals[chamb_idx]

                ax.set_title(f"{cid}: {sample_names[cid]}", fontsize=8)
                ax.set_xlabel(f"Conc ({conc.units:~})", fontsize=6)
                ax.set_ylabel(f"Ratio ({ratios.units:~})", fontsize=6)

                if preds is None or np.isnan(vals).all():
                    continue

                ax.scatter(x_data, y_data, alpha=0.7, s=15, label='data')

                sample = sample_names[cid]
                same_idxs = [i for i, s in enumerate(samples) if s == sample]
                y_all = preds[:, same_idxs]
                y_min = np.nanpercentile(y_all, 2.5, axis=1)
                y_max = np.nanpercentile(y_all, 97.5, axis=1)
                y_p = preds[:, idx].flatten()
                if hasattr(y_p, 'magnitude'):
                    y_p = y_p.magnitude
                if hasattr(y_min, 'magnitude'):
                    y_min = y_min.magnitude
                if hasattr(y_max, 'magnitude'):
                    y_max = y_max.magnitude

                ax.plot(x_data, y_p, color='red', linewidth=1, label='fit')
                ax.fill_between(x_data, y_min, y_max, color='gray', alpha=0.3, label='95% CI')

                txt = "\n".join(f"{nm}={v:.2f} {u:~}" for nm, v, u in zip(fit_types, vals, fit_units))
                ax.text(
                    0.05, 0.95, txt, transform=ax.transAxes,
                    va='top', fontsize=6, bbox=dict(boxstyle="round", fc="white", alpha=0.6),
                )

                if x_log:
                    ax.set_xscale("log")
                if y_log:
                    ax.set_yscale("log")
                ax.tick_params(axis='both', which='major', labelsize=6)

        top_fraction = 1.0 - (header_height / total_height)
        plt.tight_layout(rect=[0, 0, 1, top_fraction])
        tqdm.write(f"Exporting binding subplots to {export_path}")

        if type(export_path) == str:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(export_path, dpi=dpi)
        elif type(export_path) == list:
            for path in export_path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"Exported binding subplots to {export_path}")

    def export_binding_subplots_by_sample(
        self,
        analysis_name: str,
        model_fit_name: str,
        export_path: str,
        dep_var_label: str = 'fluorescence_ratio',
        model_pred_data_name: str = None,
        hide_excluded_samples: bool = False,
        dpi: int = 100,
        aspect_ratio: float = 1.5,
        x_log: bool = False,
        y_log: bool = False,
    ):
        '''
        Export a PDF of binding isotherm subplots for each sample with replicates.
        '''
        analysis: Data3D = self.get_run(analysis_name)
        ri = analysis.dep_var_type.index(dep_var_label)
        ratio_unit = analysis.dep_var_units[ri]
        ratios = analysis.dep_var[..., ri] * ratio_unit
        conc = analysis.indep_vars.concentration

        mf_fit: Data2D = self.get_run(model_fit_name)
        rmax_idx = mf_fit.dep_var_type.index("r_max")
        kd_idx = mf_fit.dep_var_type.index("Kd")
        all_rmax = mf_fit.dep_var[..., rmax_idx] * mf_fit.dep_var_units[rmax_idx]
        all_kds = mf_fit.dep_var[..., kd_idx] * mf_fit.dep_var_units[kd_idx]

        chambers = analysis.indep_vars.chamber_IDs
        samples = analysis.indep_vars.sample_IDs

        unique_samples = np.unique(samples)
        sample_to_chambers = {s: [] for s in unique_samples}
        for i, s in enumerate(samples):
            sample_to_chambers[s].append(i)

        n_samples = len(unique_samples)
        ncols = 3
        nrows = int(np.ceil(n_samples / ncols))
        header_height = 2.5
        plot_height = (4 / aspect_ratio) * nrows
        total_height = plot_height + header_height

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, total_height), squeeze=False)
        for ax in axes.flatten():
            ax.set_box_aspect(1 / aspect_ratio)

        description_text = (
            "Binding Isotherm Subplots by Sample\n\n"
            "This PDF displays binding isotherm subplots for each sample across its replicates.\n"
            "• Blue Circle Markers & Error Bars: Mean fluorescence ratio across replicates at each\n"
            "  ligand concentration, with error bars indicating ±1 standard deviation.\n"
            "• Solid Blue Line: Best-fit isotherm using mean r_max and mean Kd across replicates.\n"
            "• Light Blue Shaded Envelope: ±1 standard deviation interval from varying r_max while\n"
            "  keeping Kd at its mean value.\n"
            "• Text Annotation: Average r_max and Kd values with standard deviations."
        )
        fig.text(
            0.5, 1.0 - (0.5 / total_height), description_text,
            ha='center', va='top', fontsize=10, multialignment='left',
            bbox=dict(boxstyle="round,pad=0.8", fc="white", ec="silver", lw=1),
        )

        tqdm.write("Plotting binding samples...")
        skipped_samples = []
        for i, sample in enumerate(tqdm(unique_samples)):
            i = i - len(skipped_samples)
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]

            indices = sample_to_chambers[sample]
            replicate_ratios = ratios[:, indices]
            mean_ratios = np.nanmean(replicate_ratios, axis=1)
            std_ratios = np.nanstd(replicate_ratios, axis=1)

            if hasattr(mean_ratios, 'magnitude'):
                y = mean_ratios.magnitude
                yerr = std_ratios.magnitude
                x = conc.magnitude
            else:
                y = mean_ratios
                yerr = std_ratios
                x = conc

            sample_rmax = all_rmax[indices]
            sample_kds = all_kds[indices]
            mean_rmax = np.nanmean(sample_rmax)
            mean_kd = np.nanmean(sample_kds)
            rmax_stdev = np.nanstd(sample_rmax)
            kd_stdev = np.nanstd(sample_kds)
            rmax_up = mean_rmax + rmax_stdev
            rmax_down = mean_rmax - rmax_stdev

            if hide_excluded_samples:
                if hasattr(mean_rmax, 'magnitude'):
                    is_nan = np.isnan(mean_rmax.magnitude)
                else:
                    is_nan = np.isnan(mean_rmax)
                if is_nan:
                    skipped_samples.append(sample)
                    continue

            ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, label='mean ratio', color='blue')

            pred_conc_range = np.linspace(conc.min(), conc.max(), 100)
            pred_y_mean = binding_isotherm_model(pred_conc_range, mean_rmax, mean_kd)
            pred_y_up = binding_isotherm_model(pred_conc_range, rmax_up, mean_kd)
            pred_y_down = binding_isotherm_model(pred_conc_range, rmax_down, mean_kd)

            if hasattr(pred_y_mean, 'magnitude'):
                py_mean = pred_y_mean.magnitude
                py_up = pred_y_up.magnitude
                py_down = pred_y_down.magnitude
            else:
                py_mean = pred_y_mean
                py_up = pred_y_up
                py_down = pred_y_down

            ax.plot(pred_conc_range, py_mean, color="blue", label="mean fit")
            ax.fill_between(pred_conc_range, py_down, py_up, color="blue", alpha=0.2, label=r'$r_{max} \pm 1 \sigma$')

            ax.text(
                0.95, 0.05,
                f"$\\overline{{r_{{max}}}}$ = {mean_rmax.magnitude:.2f} ± {rmax_stdev.magnitude:.2f}\n"
                f"$\\overline{{K_d}}$ = {mean_kd.magnitude:.2f} ± {kd_stdev.magnitude:.2f}",
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8,
            )

            ax.set_title(f"{sample}", fontsize=10)
            conc_unit = f"({conc.units:~})" if hasattr(conc, 'units') else ""
            ratio_unit_str = f"({ratios.units:~})" if hasattr(ratios, 'units') else ""
            ax.set_xlabel(f"[Ligand] {conc_unit}", fontsize=8)
            ax.set_ylabel(f"Fluorescence Ratio {ratio_unit_str}", fontsize=8)

            if x_log:
                ax.set_xscale("log")
            if y_log:
                ax.set_yscale("log")
            ax.tick_params(axis='both', which='major', labelsize=8)

        for j in range(i + 1, nrows * ncols):
            row = j // ncols
            col = j % ncols
            axes[row, col].axis('off')

        top_fraction = 1.0 - (header_height / total_height)
        plt.tight_layout(rect=[0, 0, 1, top_fraction])
        tqdm.write(f"Exporting binding subplots by sample to {export_path}")

        if type(export_path) == str:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(export_path, dpi=dpi)
        elif type(export_path) == list:
            for path in export_path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"Exported binding subplots by sample to {export_path}")

    def export_binding_end_to_end_summary_by_sample(
        self,
        export_dir: str = 'binding_end_to_end_summaries',
        sample_ids: list = None,
        binding_raw_run: str = 'binding',
        binding_fits_run: str = 'binding_fits',
        binding_pred_run: str = None,
        dpi: int = 150,
    ):
        '''
        Export a PDF binding isotherm summary for each sample with replicates.
        '''
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from tqdm import tqdm
        from htbam_analysis.analysis.fit import binding_isotherm_model

        binding_raw = self.get_run(binding_raw_run)
        mf_fit = self.get_run(binding_fits_run)

        assert isinstance(binding_raw, Data3D), "binding_raw_run must be Data3D."
        assert isinstance(mf_fit, Data2D), "binding_fits_run must be Data2D."

        ri = binding_raw.dep_var_type.index('fluorescence_ratio')
        ratios = binding_raw.dep_var[:, :, ri]
        conc = binding_raw.indep_vars.concentration

        rmax_idx = mf_fit.dep_var_type.index('r_max')
        kd_idx = mf_fit.dep_var_type.index('Kd')
        all_rmax = mf_fit.dep_var[..., rmax_idx]
        all_kds = mf_fit.dep_var[..., kd_idx]

        preds = None
        if binding_pred_run:
            mf_pred = self.get_run(binding_pred_run)
            yi = mf_pred.dep_var_type.index('y_pred')
            preds = mf_pred.dep_var[..., yi]

        chambers = binding_raw.indep_vars.chamber_IDs
        samples = binding_raw.indep_vars.sample_IDs
        unique_samples = np.unique(samples)
        if sample_ids:
            unique_samples = [s for s in unique_samples if s in sample_ids]

        if hasattr(conc, 'magnitude'):
            x_mm = conc.magnitude
        else:
            x_mm = np.asarray(conc, dtype=float)
        x_range = np.linspace(0, max(x_mm), 100) if len(x_mm) > 0 else []

        Path(export_dir).mkdir(parents=True, exist_ok=True)

        for sample in tqdm(unique_samples, desc='Binding summaries'):
            chamber_indices = np.where(samples == sample)[0]
            if len(chamber_indices) == 0:
                continue

            n_reps = len(chamber_indices)
            mean_rmax = np.nanmean(all_rmax[chamber_indices])
            std_rmax = np.nanstd(all_rmax[chamber_indices])
            mean_kd = np.nanmean(all_kds[chamber_indices])

            fig, axes = plt.subplots(n_reps, 2, figsize=(10, 3 * n_reps), squeeze=False)

            for row, c_idx in enumerate(chamber_indices):
                ax_data = axes[row, 0]
                y_data = ratios[:, c_idx]
                if hasattr(y_data, 'magnitude'):
                    y_data = y_data.magnitude
                ax_data.scatter(x_mm, y_data, color='blue', s=20)
                ax_data.set_xlabel('[Ligand]')
                ax_data.set_ylabel('Fluorescence Ratio')
                ax_data.set_title(f"{chambers[c_idx]}: {samples[c_idx]}")

                ax_fit = axes[row, 1]
                if preds is not None:
                    y_pred = preds[:, c_idx]
                    if hasattr(y_pred, 'magnitude'):
                        y_pred = y_pred.magnitude
                    ax_fit.plot(x_mm, y_pred, color='red', label='fit')
                elif len(x_range) > 0 and not np.isnan(mean_rmax) and not np.isnan(mean_kd):
                    rmax_m = mean_rmax.magnitude if hasattr(mean_rmax, 'magnitude') else mean_rmax
                    kd_m = mean_kd.magnitude if hasattr(mean_kd, 'magnitude') else mean_kd
                    std_m = std_rmax.magnitude if hasattr(std_rmax, 'magnitude') else std_rmax
                    y_mean = binding_isotherm_model(x_range, rmax_m, kd_m)
                    y_up = binding_isotherm_model(x_range, rmax_m + std_m, kd_m)
                    y_down = binding_isotherm_model(x_range, rmax_m - std_m, kd_m)
                    ax_fit.fill_between(x_range, y_down, y_up, color='gray', alpha=0.3)
                    ax_fit.plot(x_range, y_mean, 'k--')
                ax_fit.set_xlabel('[Ligand]')
                ax_fit.set_ylabel('Fluorescence Ratio')
                ax_fit.set_title('Binding Isotherm')

            fig.suptitle(f"Sample {sample}", fontsize=14)
            fig.tight_layout()
            out_path = Path(export_dir) / f"{sample}.pdf"
            plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            tqdm.write(f"Saved {out_path}")

