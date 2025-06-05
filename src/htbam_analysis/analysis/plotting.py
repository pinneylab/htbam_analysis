from dash import Dash, dcc, html, Input, Output, no_update, State
import plotly.graph_objs as go
import base64
import tempfile
import matplotlib.pyplot as plt
import numpy as np

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
    for chamber_id, value in plotting_var.items():
        x = int(chamber_id.split(',')[0])
        y = int(chamber_id.split(',')[1])
        img_array[y-1,x-1] = value 
    
    #generate title
    if title is None:
        title = ''
    
    #Create the figure
    layout = go.Layout()

    # create 1‐indexed axes
    x_vals = list(range(1, img_array.shape[1] + 1))
    y_vals = list(range(1, img_array.shape[0] + 1))

    heatmap = go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=img_array,
        colorscale='Viridis',
        hovertemplate='x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>'
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
        style={'display': 'flex'},
        children=[
            html.Div(
                style={'flex': '1'},
                children=[
                    dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
                    dcc.Tooltip(id="graph-tooltip"),
                ]
            ),
            html.Div(id="side-panel", style={'flex': '1', 'paddingLeft': '20px', 'backgroundColor': 'white'})  # Set side-panel background to white
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

    ### HIGHLIGHT ON HOVER / CLICK ###
    @app.callback(
        Output("graph", "figure"),
        Input("graph", "hoverData"),
        Input("graph", "clickData"),
        State("graph", "figure"),
    )
    def update_highlights(hoverData, clickData, fig):
        # clear old shapes
        fig["layout"]["shapes"] = []

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