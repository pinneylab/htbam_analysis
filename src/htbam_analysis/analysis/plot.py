from dash import Dash, dcc, html, Input, Output, no_update, State
import plotly.graph_objs as go
import base64
import tempfile
import matplotlib.pyplot as plt
import numpy as np

# Utilities
from typing import TYPE_CHECKING, List, Dict, Optional, Any
if TYPE_CHECKING:
    from htbam_analysis.analysis.experiment import HTBAMExperiment

# HTBAM Data
from htbam_db_api.data import Data4D, Data3D, Data2D

# Plotting
import seaborn as sns

def plot_chip(plotting_var, chamber_names, graphing_function=None, title=None):
    ''' This function creates a Dash visualization of a chip, based on a certain Run (run_name)
        Inputs:
            plotting_var: a dictionary mapping chamber_id to the variable to be plotted for that chamber
            chamber_names: a dictionary mapping chamber_id to the name of the sample in the chamber (e.g. '1,1': ecADK_XYZ')
            graphing_function: a function that takes in a single chamber_id (e.g. '1,1') and matplotlib axis and returns the axis object after plotting.
            title: a string to be used as the title of the plot
        TODO: make all the variables stored in Dash properly...
    '''

    # Make the image array
    #NB: eventually, store width/height in DB and reference!
    img_array = np.zeros([56,32])

    # Here we're plotting to value for each chamber (e.g. coloring by std curve slope)
    units = ''
    for chamber_id, value in plotting_var.items():
        x = int(chamber_id.split(',')[0])
        y = int(chamber_id.split(',')[1])
        
        # Check if value has units (Pint quantity)
        if hasattr(value, 'magnitude') and hasattr(value, 'units'):
            img_array[y-1,x-1] = float(value.magnitude)
            if not units:
                units = str(f'{value.units:~}')
        else:
            img_array[y-1,x-1] = float(value) 
    
    #generate title
    if title is None:
        title = ''
    
    #Create the figure
    layout = go.Layout()

    # create 1‐indexed axes
    x_vals = list(range(1, img_array.shape[1] + 1))
    y_vals = list(range(1, img_array.shape[0] + 1))

    # To discard outliers that mess with the data, we're using the 5th and 95th percentiles.
    zmin, zmax = np.nanpercentile(img_array, [5, 95])

    heatmap = go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=img_array,
        zmin=zmin,
        zmax=zmax,
        colorscale='Viridis',
        colorbar=dict(title=units),
        hovertemplate='x=%{x}<br>y=%{y}<br>z=%{z} ' + units +'<extra></extra>'
    )
    
    fig = go.Figure(layout=layout, data=heatmap)

    #center title in fig
    fig.update_layout(title=title,
                        title_x=0.5, 
                        yaxis=dict(scaleanchor="x", scaleratio=1, autorange='reversed'), 
                        xaxis=dict(scaleratio=1),
                        plot_bgcolor='rgba(0,0,0,0)',
                        width=600, height=600,
                        hovermode='x')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    #create dash app:
    app = Dash(__name__)
    app.layout = html.Div(
        style={'display': 'flex', 'backgroundColor': 'white', 'minHeight': '100vh'},
        children=[
            html.Div(
                style={'flex': '1'},
                children=[
                    dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
                    dcc.Input(id="search-input", type="text", placeholder="Search for sample...", style={'width': '100%', 'marginTop': '10px'}),
                    dcc.Tooltip(id="graph-tooltip"),
                ]
            ),
            html.Div(id="side-panel", style={'flex': '1', 'paddingLeft': '20px', 'backgroundColor': 'white'})
        ]
    )

    ### GRAPHING FUNCTION ON HOVER:
    if graphing_function is not None:
        @app.callback(
            Output("graph-tooltip", "show"),
            Output("graph-tooltip", "bbox"),
            Output("graph-tooltip", "children"),
            Input("graph", "hoverData"),
        )
        def display_hover(hoverData):
            if hoverData is None:
                return False, no_update, no_update
            # demo only shows the first point, but other points may also be available
            pt = hoverData["points"][0]
            chamber_id = str(pt['x']) + ',' + str(pt['y'])
            bbox = pt["bbox"]
            chamber_name = chamber_names[chamber_id]
            #get the data for the point:
            fig, ax = plt.subplots()
            ax = graphing_function(chamber_id, ax)
            #reduce whitespace on margins of graph:
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
            #save the figure as a temp file:
            tempfile_name = tempfile.NamedTemporaryFile().name+'.png'
            plt.savefig(tempfile_name)
            plt.close()
            # #read in temp file as base64 encoded string:
            with open(tempfile_name, "rb") as image_file:
                img_src = "data:image/png;base64," + str(base64.b64encode(image_file.read()).decode("utf-8"))
            children = [
                html.Div(children=[
                    #no space after header:
                    html.H3('{},{}:  {}'.format(pt['x'], pt['y'], chamber_name), style={"color": 'black', "fontFamily":"Arial", "textAlign": "center", "marginBottom": "0px"}), #1-index
                    #add the image with reduced whitespace:
                    html.Img(src=img_src, style={"width": "100%"}),
                ],
                style={'width': '400px', 'white-space': 'none'})
            ]

            return True, bbox, children

    ### HIGHLIGHT ON HOVER / CLICK / SEARCH ###
    @app.callback(
        Output("graph", "figure"),
        Input("graph", "hoverData"),
        Input("graph", "clickData"),
        Input("search-input", "value"),
        State("graph", "figure"),
    )
    def update_highlights(hoverData, clickData, search_value, fig):
        # clear old shapes
        fig["layout"]["shapes"] = []

        # search => cyan outlines
        if search_value:
            search_str = search_value.lower()
            for cid, name in chamber_names.items():
                if name and search_str in name.lower():
                    i, j = map(int, cid.split(","))
                    fig["layout"]["shapes"].append({
                        "type": "rect",
                        "x0": i-0.4, "x1": i+0.4,
                        "y0": j-0.4, "y1": j+0.4,
                        "line": {"color": "cyan", "width": 3},
                        "fillcolor": "rgba(0,0,0,0)",
                        "name": "search"
                    })

        # hover => red outlines
        if hoverData:
            pt = hoverData["points"][0]
            sample = chamber_names.get(f"{pt['x']},{pt['y']}")
            if sample:
                for cid, name in chamber_names.items():
                    if name == sample:
                        i, j = map(int, cid.split(","))
                        fig["layout"]["shapes"].append({
                            "type": "rect",
                            "x0": i-0.4, "x1": i+0.4,
                            "y0": j-0.4, "y1": j+0.4,
                            "line": {"color": "red", "width": 2},
                            "fillcolor": "rgba(0,0,0,0)",
                            "name": "hover"
                        })

        # click => magenta outlines
        if clickData:
            pt = clickData["points"][0]
            sample = chamber_names.get(f"{pt['x']},{pt['y']}")
            if sample:
                for cid, name in chamber_names.items():
                    if name == sample:
                        i, j = map(int, cid.split(","))
                        fig["layout"]["shapes"].append({
                            "type": "rect",
                            "x0": i-0.4, "x1": i+0.4,
                            "y0": j-0.4, "y1": j+0.4,
                            "line": {"color": "magenta", "width": 3},
                            "fillcolor": "rgba(0,0,0,0)",
                            "name": "selected"
                        })

        return fig

    ### GRAPHING FUNCTION ON CLICK (side‐panel) ###
    if graphing_function is not None:
        @app.callback(
            Output("side-panel", "children"),
            Input("graph", "clickData"),
        )
        def display_click_side(clickData):
            if clickData is None:
                return no_update
            # identify clicked chamber
            pt = clickData["points"][0]
            cid = f"{pt['x']},{pt['y']}"
            sample = chamber_names.get(cid, "")

            # generate inset plot PNG
            fig2, ax2 = plt.subplots()
            ax2 = graphing_function(cid, ax2)
            fig2.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
            tmp = tempfile.NamedTemporaryFile().name + ".png"
            plt.savefig(tmp)
            plt.close(fig2)
            with open(tmp, "rb") as f:
                img_src = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

            # return side‐panel contents
            return html.Div([
                html.H3(f"{cid}: {sample}", style={"textAlign": "center"}),
                html.Img(src=img_src, style={"width": "100%"})
            ])

    app.run_server()

def plot_chip_by_variable(experiment: 'HTBAMExperiment', analysis_name: str, variable: str):
    '''
    Plot a full chip with raw data and a specified variable.

    Parameters:
        experiment ('HTBAMExperiment'): the experiment object.
        analysis_name (str): the name of the analysis to be plotted.
        variable (str): the name of the variable to be plotted.

    Returns:
        None
    '''
    #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
    
    analysis_data = experiment.get_run(analysis_name)     # Analysis data (to show slopes/intercepts)
    
    plotting_var = variable
    plotting_var_unit = analysis_data.dep_var_units[analysis_data.dep_var_type.index(plotting_var)]

    # Verify we have the variable:
    if plotting_var not in analysis_data.dep_var_type:
        raise ValueError(f"'{plotting_var}' not found in analysis data. Available variables: {analysis_data.dep_vars_types}")
    else:
        plotting_var_index = analysis_data.dep_var_type.index(plotting_var)

    concentration = analysis_data.dep_var[..., plotting_var_index] * plotting_var_unit # (n_chambers, n_conc, 1)
    
    #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
    #chamber_names_dict = experiment._db_conn.get_chamber_name_dict()
    chamber_names = analysis_data.indep_vars.chamber_IDs # (n_chambers,)
    sample_names =  analysis_data.indep_vars.sample_IDs # (n_chambers,)

    # Create dictionary mapping chamber_id -> sample_name:
    sample_names_dict = {}
    for i, chamber_id in enumerate(chamber_names):
        sample_names_dict[chamber_id] = sample_names[i]

    # Create dictionary mapping chamber_id -> concentration:
    concentration_dict = {}
    for i, chamber_id in enumerate(chamber_names):
        concentration_dict[chamber_id] = concentration[i]
        

    #plotting function: We'll generate a subplot for each chamber, showing the histogram across replicates with our chosen chamber in red.
    def plot_chamber_variable(chamber_id, ax):
        #parameters:
        # get the sample_ID for this chamber:
        sample_id = sample_names_dict[chamber_id]
        # Get all replicates with this sample_id:
        repl_indices = [i for i, s in enumerate(sample_names) if s == sample_id]
        # Get the concentration values for these replicates:
        repl_concentrations = concentration[repl_indices]

        #x_data = concentration_dict[chamber_id]
        ax.hist(repl_concentrations, bins=10)
        # add the current datapoint as a red bar:
        ax.axvline(concentration_dict[chamber_id], color='red', linestyle='dashed', linewidth=2)
        ax.set_title(f'{chamber_id}: {sample_names_dict[chamber_id]}')
        ax.set_xlabel(f'{plotting_var}')
        ax.set_ylabel('Count')
        return ax

    plot_chip(concentration_dict, sample_names_dict, title=f'Analysis: {plotting_var}', graphing_function=plot_chamber_variable)

def plot_standard_curve_chip(experiment: 'HTBAMExperiment', analysis_name: str, experiment_name: str):
    '''
    Plot a full chip with raw data and std curve slopes.

    Parameters:
        experiment ('HTBAMExperiment'): the experiment object.
        analysis_name (str): the name of the analysis to be plotted.
        experiment_name (str): the name of the raw experiment data to be plotted.

    Returns:
        None
    '''
    #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
    
    experiment_data = experiment.get_run(experiment_name) # Raw data from experiment (to show datapoints)
    analysis_data = experiment.get_run(analysis_name)     # Analysis data (to show slopes/intercepts)
    
    slope_idx = analysis_data.dep_var_type.index('slope')          # index of slope in dep_vars
    intercept_idx = analysis_data.dep_var_type.index('intercept')  # index of intercept in dep_vars
    r_squared_idx = analysis_data.dep_var_type.index('r_squared')  # index of r_squared in dep_vars
    
    slope_unit = analysis_data.dep_var_units[slope_idx]
    intercept_unit = analysis_data.dep_var_units[intercept_idx]
    r_squared_unit = analysis_data.dep_var_units[r_squared_idx]
    
    # Extract slopes and intercepts from analysis data
    slopes_to_plot = analysis_data.dep_var[..., slope_idx] * slope_unit          # (n_chambers,)
    intercepts_to_plot = analysis_data.dep_var[..., intercept_idx] * intercept_unit  # (n_chambers,)
    r_squared = analysis_data.dep_var[..., r_squared_idx] * r_squared_unit  # (n_chambers,)

    luminance_unit = experiment_data.dep_var_units[experiment_data.dep_var_type.index('luminance')]
    
    # Extract luminance and concentration from experiment data
    luminance_idx = experiment_data.dep_var_type.index('luminance')  # index of luminance in dep_vars
    luminance = experiment_data.dep_var[..., luminance_idx] * luminance_unit  # (n_chambers, n_timepoints, n_conc)
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
            ax.plot(x_data, m*x_data + b)
            ax.set_title(f'{chamber_id}: {sample_names_dict[chamber_id]}')
            ax.set_xlabel(f'Concentration ({concentration.units:~})')
            ax.set_ylabel(f'Luminance ({luminance.units:~})')
            ax.legend([f'Current Chamber Slope: {slopes_dict[chamber_id].magnitude:.2f} {slope_unit:~}'])
        return ax
    
    plot_chip(slopes_dict, sample_names_dict, graphing_function=plot_chamber_slopes, title='Standard Curve: Slope')

def plot_initial_rates_chip(experiment: 'HTBAMExperiment', analysis_name: str, experiment_name: str, skip_start_timepoint: bool = True,
                            plot_xmax: float = None, plot_ymax: float = None,
                            plot_xmin: float = None, plot_ymin: float = None):
    '''
    Plot a full chip with raw data and fit initial rates.

    Parameters:
        experiment ('HTBAMExperiment'): the experiment object.
        analysis_name (str): the name of the analysis to be plotted.
        experiment_name (str): the name of the experiment to be plotted.
        skip_start_timepoint (bool): whether to skip the first timepoint in the analysis (Sometimes are unusually low). Default is True.

    Returns:
        None
    '''
    #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
    
    experiment_data = experiment.get_run(experiment_name) # Raw data from experiment (to show datapoints)
    analysis_data = experiment.get_run(analysis_name)     # Analysis data (to show slopes/intercepts)
    
    # Extract slopes and intercepts from analysis data
    slopes_idx = analysis_data.dep_var_type.index('slope')          # index of slope in dep_vars
    intercepts_idx = analysis_data.dep_var_type.index('intercept')  # index of intercept in dep_vars
    r_squared_idx = analysis_data.dep_var_type.index('r_squared')  # index of r_squared in dep_vars
    
    slope_unit = analysis_data.dep_var_units[slopes_idx]
    intercept_unit = analysis_data.dep_var_units[intercepts_idx]
    r_squared_unit = analysis_data.dep_var_units[r_squared_idx]

    # Extract slopes and intercepts from analysis data
    slopes_to_plot = analysis_data.dep_var[..., slopes_idx] * slope_unit          # (n_chambers, n_conc)
    intercepts_to_plot = analysis_data.dep_var[..., intercepts_idx] * intercept_unit  # (n_chambers, n_conc)
    r_squared = analysis_data.dep_var[..., r_squared_idx] * r_squared_unit          # (n_chambers, n_conc)

    # Extract product_concentration (Y) from experiment data
    product_conc_idx = experiment_data.dep_var_type.index('concentration')  # index of luminance in dep_vars
    product_conc_unit = experiment_data.dep_var_units[product_conc_idx]
    product_conc = experiment_data.dep_var[..., product_conc_idx] * product_conc_unit  # (n_chambers, n_timepoints, n_conc)
    substrate_conc = experiment_data.indep_vars.concentration # (n_conc,)
    time_data = experiment_data.indep_vars.time # (n_conc, n_timepoints)

    # If skip_start_timepoint is True, we'll skip the first timepoint in the analysis
    if skip_start_timepoint:
        product_conc = product_conc[:, 1:, :] # (n_chambers, n_timepoints-1, n_conc)
        time_data = time_data[:, 1:] # (n_conc, n_timepoints-1)
        #slopes_to_plot = slopes_to_plot[:, 1:]
    
    #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
    #chamber_names_dict = experiment._db_conn.get_chamber_name_dict()
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
        #data_index = list(experiment._run_data[run_name]["chamber_idxs"]).index(chamber_id)
        x_data = time_data # same for all chambers              (n_timepoints, n_conc)
        y_data = product_conc[:, :, chamber_names == chamber_id]  #(n_timepoints, n_conc)
    
        m = slopes_to_plot[:, chamber_names == chamber_id]
        b = intercepts_to_plot[:, chamber_names == chamber_id]
        
        colors = sns.color_palette('husl', n_colors=y_data.shape[0])

        for i in range(y_data.shape[0]): #over each substrate concentration:

            ax.scatter(x_data[i], y_data[i,:].flatten(), color=colors[i], alpha=0.3) # raw data
            ax.plot(x_data[i], m[i]*x_data[i] + b[i], color=colors[i], alpha=1, linewidth=2, label=f'{substrate_conc[i]:~}')  # fitted line

        # Set axis limits if provided
        if plot_xmax is not None:
            ax.set_xlim(right=plot_xmax)
        if plot_ymax is not None:
            ax.set_ylim(top=plot_ymax)
        if plot_xmin is not None:
            ax.set_xlim(left=plot_xmin)
        if plot_ymin is not None:
            ax.set_ylim(bottom=plot_ymin)
        
        ax.set_xlabel(f'Time ({time_data.units:~})')
        ax.set_ylabel(f'Product Concentration ({product_conc.units:~})')

        ax.legend()

        return ax
    
    plot_chip(slopes_dict, sample_names_dict, graphing_function=plot_chamber_initial_rates, title='Kinetics: Initial Rates')

def plot_initial_rates_vs_concentration_chip(experiment: 'HTBAMExperiment',
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
    analysis: Data3D = experiment.get_run(analysis_name)
    si = analysis.dep_var_type.index("slope")
    slope_unit = analysis.dep_var_units[si]
    slopes = analysis.dep_var[..., si] * slope_unit  # (n_conc, n_chambers)
    conc   = analysis.indep_vars.concentration       # (n_conc,)

    if model_pred_data_name:
        mf_pred: Data3D = experiment.get_run(model_pred_data_name)
        yi = mf_pred.dep_var_type.index("y_pred")
        y_pred_unit = mf_pred.dep_var_units[yi]
        preds = mf_pred.dep_var[..., yi] * y_pred_unit          # (n_conc, n_chambers)

    if model_fit_name:
        mf_fit: Data2D = experiment.get_run(model_fit_name)
        fit_types = mf_fit.dep_var_type           # e.g. ["v_max","K_m","r_squared"]
        fit_vals = mf_fit.dep_var                # shape (n_chamb, len(fit_types))
        fit_units = mf_fit.dep_var_units

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
            txt = "".join(f"{nm}={v:.2f} {u:~}\n" for nm,v,u in zip(fit_types, vals, fit_units))
            ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                    va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        if model_pred_data_name or model_fit_name:
            ax.legend()
        ax.set_title(f"{cid}: {sample_names[cid]}")
        ax.set_xlabel(f"Concentration ({conc.units:~})")
        ax.set_ylabel(f"Initial Rate ({slopes.units:~})")
        if x_log: ax.set_xscale("log")
        if y_log: ax.set_yscale("log")
        return ax

    plot_chip(mean_rates, sample_names,
              graphing_function=plot_rates_vs_conc,
              title="Initial Rates vs Concentration")

def plot_MM_chip(experiment: 'HTBAMExperiment',
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
    analysis: Data3D = experiment.get_run(analysis_name)
    si = analysis.dep_var_type.index("slope")
    slope_unit = analysis.dep_var_units[si]
    slopes = analysis.dep_var[..., si] * slope_unit  # (n_conc, n_chambers)
    conc   = analysis.indep_vars.concentration       # (n_conc,)

    mf_fit: Data2D = experiment.get_run(model_fit_name)
    fit_types = mf_fit.dep_var_type           # e.g. ["v_max","K_m","r_squared"]
    fit_vals = mf_fit.dep_var                # shape (n_chamb, len(fit_types))
    fit_units = mf_fit.dep_var_units

    mm_idx = mf_fit.dep_var_type.index("v_max")
    mms = mf_fit.dep_var[..., mm_idx] * fit_units[mm_idx]   # (n_chambers,)

    if model_pred_data_name:
        mf_pred: Data3D = experiment.get_run(model_pred_data_name)
        yi = mf_pred.dep_var_type.index("y_pred")
        y_pred_unit = mf_pred.dep_var_units[yi]
        preds = mf_pred.dep_var[..., yi] * y_pred_unit          # (n_conc, n_chambers)

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
            
            # Show the 95% confidence interval:
            y_min = np.nanpercentile(y_all, 2.5, axis=1)
            y_max = np.nanpercentile(y_all, 97.5, axis=1)
            
            # then overplot this chamber’s model fit
            y_p = preds[:, idx].flatten()
            ax.plot(x, y_p, color="red", label="current well fit")
            ax.fill_between(x, y_min, y_max, color="gray", alpha=0.3, label='95% CI')

        if model_fit_name:
            # extract this chamber's fit row
            chamb_idx = np.where(chambers == cid)[0][0]
            vals = fit_vals[chamb_idx]
            txt = "".join(f"{nm}={v:.2f} {u:~}\n" for nm,v,u in zip(fit_types, vals, fit_units))
            ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                    va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        
        if model_pred_data_name or model_fit_name:
            ax.legend()
        ax.set_title(f"{cid}: {sample_names[cid]}")
        ax.set_xlabel(f"Concentration ({conc.units:~})")
        ax.set_ylabel(f"Initial Rate ({slopes.units:~})")
        if x_log: ax.set_xscale("log")
        if y_log: ax.set_yscale("log")
        return ax

    plot_chip(mms_to_plot, sample_names,
              graphing_function=plot_rates_vs_conc,
              title="Initial Rates vs Concentration")   

def plot_MM_div_E_chip(experiment: 'HTBAMExperiment',
                analysis_name: str,
                model_fit_name: str,
                dep_var_name='slope',
                x_log: bool = False,
                y_log: bool = False):
    """
    Plot MM values, with inset initial rates vs substrate concentration for each chamber.
    Optionally overlay fitted curve from `model_pred_data_name` and
    annotate fit parameters from `model_fit_name`.
    """
    analysis: Data3D = experiment.get_run(analysis_name)
    si = analysis.dep_var_type.index(dep_var_name)
    slope_unit = analysis.dep_var_units[si]
    slopes = analysis.dep_var[..., si] * slope_unit  # (n_conc, n_chambers)
    conc   = analysis.indep_vars.concentration       # (n_conc,)

    mf_fit: Data2D = experiment.get_run(model_fit_name)
    fit_types = mf_fit.dep_var_type           # e.g. ["v_max","K_m","r_squared", "kcat"]
    fit_vals = mf_fit.dep_var                # shape (n_chamb, len(fit_types))
    fit_units = mf_fit.dep_var_units

    kcat_idx = mf_fit.dep_var_type.index("kcat")
    all_kcats = mf_fit.dep_var[..., kcat_idx] * fit_units[kcat_idx]   # (n_chambers,)
    kM_idx = mf_fit.dep_var_type.index("K_m")
    all_kMs = mf_fit.dep_var[..., kM_idx] * fit_units[kM_idx]   # (n_chambers,)

    # We'll be generating pred_data on-the-fly by pushing our kcat up and down one stdev.

    chambers = analysis.indep_vars.chamber_IDs        # (n_chambers,)
    samples  = analysis.indep_vars.sample_IDs         # (n_chambers,)
    sample_names = {cid: samples[i] for i, cid in enumerate(chambers)}
    #mean_rates   = {cid: np.nanmean(slopes[:, i]) for i, cid in enumerate(chambers)}
    kcats_to_plot = {cid: all_kcats[i] for i, cid in enumerate(chambers)}
    kMs_to_plot = {cid: all_kMs[i] for i, cid in enumerate(chambers)}

    def plot_rates_vs_conc(cid, ax):
        idx = (chambers == cid)
        x   = conc
        y   = slopes[:, idx].flatten()
        ax.scatter(x, y, alpha=0.7, label="current well")


        sample = sample_names[cid]
        same_idxs = [i for i, s in enumerate(samples) if s == sample]
        
        sample_kcats = all_kcats[same_idxs]
        sample_kMs = all_kMs[same_idxs]

        current_kcat = kcats_to_plot[cid]
        current_km = kMs_to_plot[cid]

        mean_kcat = np.nanmean(sample_kcats)
        mean_km = np.nanmean(sample_kMs)

        # stdev of kcat?
        kcat_stdev = np.nanstd(sample_kcats)
        km_stdev = np.nanstd(sample_kMs)
        
        # push kcat up and down one stdev
        kcat_up = mean_kcat + kcat_stdev
        kcat_down = mean_kcat - kcat_stdev

        # calculate new slopes
        from htbam_analysis.analysis.fit import mm_model

        pred_y_mean = mm_model(conc, mean_kcat, mean_km)
        pred_y_up = mm_model(conc, kcat_up, mean_km)
        pred_y_down = mm_model(conc, kcat_down, mean_km)
        pred_y_current = mm_model(conc, current_kcat, current_km)

        ax.plot(x, pred_y_current, color="blue", label="current well fit")

        ax.plot(x, pred_y_mean, color="gray", label="mean well fit")
        ax.fill_between(x, pred_y_down, pred_y_up, color="gray", alpha=0.3, label='$k_{cat}$ ± 1 stdev')

        ax.text(0.05, 0.95, 
                f"$\\overline{{k_{{cat}}}}$ = {mean_kcat.magnitude:.2f} ± {kcat_stdev:.2f~}\n"
                f"$\\overline{{K_{{M}}}}$ = {mean_km.magnitude:.2f} ± {km_stdev:.2f~}", 
                transform=ax.transAxes,
                va="top", fontsize=8, zorder=10,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        # if model_fit_name:
        #     # extract this chamber's fit row
        #     chamb_idx = np.where(chambers == cid)[0][0]
        #     vals = fit_vals[chamb_idx]
        #     txt = "".join(f"{nm}={v:.2f} {u:~}\n" for nm,v,u in zip(fit_types, vals, fit_units))
        #     ax.text(0.05, 0.95, txt, transform=ax.transAxes,
        #             va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        
        #if model_pred_data_name or model_fit_name:
        ax.legend()
        ax.set_title(f"{cid}: {sample_names[cid]}")
        ax.set_xlabel(f"$[S]$ ({conc.units:~})")
        ax.set_ylabel(f"$V_0/[E]$ ({slopes.units:~})")
        if x_log: ax.set_xscale("log")
        if y_log: ax.set_yscale("log")
        return ax

    plot_chip(kcats_to_plot, sample_names,
              graphing_function=plot_rates_vs_conc,
              title="V_0/[E] vs [S]")       

def plot_ic50_chip(experiment: 'HTBAMExperiment',
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
    analysis: Data3D = experiment.get_run(analysis_name)
    si = analysis.dep_var_type.index("slope")
    slope_unit = analysis.dep_var_units[si]
    slopes = analysis.dep_var[..., si] * slope_unit  # (n_conc, n_chambers)
    conc   = analysis.indep_vars.concentration       # (n_conc,)

    mf_fit: Data2D = experiment.get_run(model_fit_name)
    fit_types = mf_fit.dep_var_type           # e.g. ["v_max","K_m","r_squared"]
    fit_vals = mf_fit.dep_var                # shape (n_chamb, len(fit_types))
    fit_units = mf_fit.dep_var_units

    ic50s_idx = mf_fit.dep_var_type.index("ic50")
    ic50s = mf_fit.dep_var[..., ic50s_idx] * fit_units[ic50s_idx]    # (n_chambers,)

    if model_pred_data_name:
        mf_pred: Data3D = experiment.get_run(model_pred_data_name)
        yi = mf_pred.dep_var_type.index("y_pred")
        y_pred_unit = mf_pred.dep_var_units[yi]
        preds = mf_pred.dep_var[..., yi] * y_pred_unit          # (n_conc, n_chambers)

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

            # Show the 95% confidence interval:
            y_min = np.nanpercentile(y_all, 2.5, axis=1)
            y_max = np.nanpercentile(y_all, 97.5, axis=1)
            
            # then overplot this chamber’s model fit
            y_p = preds[:, idx].flatten()
            ax.plot(x, y_p, color="red", label="current well fit")
            ax.fill_between(x, y_min, y_max, color="gray", alpha=0.3, label='95% CI')

        if model_fit_name:
            # extract this chamber's fit row
            chamb_idx = np.where(chambers == cid)[0][0]
            vals = fit_vals[chamb_idx]
            txt = "".join(f"{nm}={v:.2f} {u:~}\n" for nm,v,u in zip(fit_types, vals, fit_units))
            ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                    va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        
        if model_pred_data_name or model_fit_name:
            ax.legend()
        ax.set_title(f"{cid}: {sample_names[cid]}")
        ax.set_xlabel(f"Concentration ({conc.units:~})")
        ax.set_ylabel(f"Initial Rate ({slopes.units:~})")
        if x_log: ax.set_xscale("log")
        if y_log: ax.set_yscale("log")
        return ax

    plot_chip(ic50s_to_plot, sample_names,
              graphing_function=plot_rates_vs_conc,
              title="Initial Rates vs Concentration")

def plot_enzyme_concentration_chip(experiment: 'HTBAMExperiment', analysis_name: str, skip_start_timepoint: bool = True):
    '''
    Plot a full chip with raw data and fit initial rates.

    Parameters:
        experiment ('HTBAMExperiment'): the experiment object.
        analysis_name (str): the name of the analysis to be plotted.
        skip_start_timepoint (bool): whether to skip the first timepoint in the analysis (Sometimes are unusually low). Default is True.

    Returns:
        None
    '''
    #plotting variable: We'll plot by enzyme concentration. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
    analysis_data = experiment.get_run(analysis_name)     # Analysis data (to show slopes/intercepts)
    
    plotting_var = 'concentration'

    # Verify we have the variable:
    if plotting_var not in analysis_data.dep_var_type:
        raise ValueError(f"'{plotting_var}' not found in analysis data. Available variables: {analysis_data.dep_vars_types}")
    else:
        plotting_var_index = analysis_data.dep_var_type.index(plotting_var)

    conc_units = analysis_data.dep_var_units[plotting_var_index]
    concentration = analysis_data.dep_var[..., plotting_var_index] * conc_units # (n_chambers, n_conc, 1)
    
    #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
    #chamber_names_dict = experiment._db_conn.get_chamber_name_dict()
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

    def graphing_function(chamber_id, ax):
        # plot histogram of the concentration from all chambers with the same sample name as the chamber_id
        sample_name = sample_names_dict[chamber_id]
        conc = concentration[sample_names == sample_name]
        ax.hist(conc, bins=10)
        # Make the bin for the current chamber red:
        # What bin has the max count?
        bin_max_count = np.max(np.histogram(conc.magnitude, bins=10)[0])
        ax.vlines(conc_dict[chamber_id].magnitude, 0, bin_max_count, colors='red', linestyles='--')
        ax.set_title(f'Concentration of {sample_name}')
        ax.set_xlabel(f'Concentration {conc_units:~}')
        ax.set_ylabel('Count')
        # print legend current chamber with concentration:
        ax.legend([f'Current Chamber: {conc_dict[chamber_id].magnitude:.2f} {conc_units:~}'])
        return ax
    
    plot_chip(conc_dict, sample_names_dict, 
            graphing_function=graphing_function,
            title=f'Enzyme Concentration')

def plot_mask_chip(experiment: 'HTBAMExperiment', mask_name: str):
    '''
    Plot a full chip with raw data and fit initial rates.

    Parameters:
        experiment ('HTBAMExperiment'): the experiment object.
        mask_name (str): the name of the mask to be plotted. (Data3D or Data2D)

    Returns:
        None
    '''
    #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
    mask_data = experiment.get_run(mask_name)     # Analysis data (to show slopes/intercepts)
    
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

def plot_chip_by_variable(experiment: 'HTBAMExperiment', analysis_name: str, variable: str):
    '''
    Plot a chip using arbitrary DataND object, by specifying the variable to plot.

    Parameters:
        experiment ('HTBAMExperiment'): the experiment object.
        analysis_name (str): the name of the analysis to be plotted. (Data3D or Data2D)
        variable (str): the variable to plot.

    Returns:
        None
    '''

    #plotting variable: We'll plot by luminance. We need a dictionary mapping chamber id (e.g. '1,1') to the value to be plotted (e.g. slope)
    data = experiment.get_run(analysis_name)     # Analysis data (to show slopes/intercepts)
    
    dtype = type(data)
    assert dtype in [Data3D, Data2D], "data must be of type Data3D or Data2D."

    mask_idx = data.dep_var_type.index(variable)

    # If we're using a data2D
    mask = data.dep_var[..., mask_idx] # (n_conc, n_chambers,)

    # We want to plot the number of concentrations that pass the mask in each well.
    # so, we'll sum across the concentration dimension, leaving an (n_chambers,) array

    #passed_conc = np.sum(mask, axis=0)
    #print(mask.shape)

    #chamber_names: We'll provide the name of the sample in each chamber as well, in the same way:
    chamber_names = data.indep_vars.chamber_IDs # (n_chambers,)
    sample_names =  data.indep_vars.sample_IDs # (n_chambers,)

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
    
    plot_chip(mask_sum, sample_names_dict, title=f'{variable}')
