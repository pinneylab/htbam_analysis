import argparse
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column
import numpy as np
from PIL import Image

from bokeh.io import curdoc
from bokeh.events import Tap
from bokeh.models import Slider
from bokeh.models import Div

"""

"""

def compute_corner_slices(stitched_image):
    
    fraction = 1/8
    height, width = stitched_image.shape
    h_crop = int(height * fraction)
    w_crop = int(width * fraction)

    slices = [
        (slice(0, h_crop), slice(0, w_crop)),                            # Top-left
        (slice(0, h_crop), slice(width - w_crop, width)),               # Top-right
        (slice(height - h_crop, height), slice(0, w_crop)),            # Bottom-left
        (slice(height - h_crop, height), slice(width - w_crop, width)) # Bottom-right
    ]

    return slices

# init parser
parser = argparse.ArgumentParser(description='Launches corner-picking GUI')
parser.add_argument("stitched_image", type=str, help='Path to a stitched image.')
args = parser.parse_args()
stitched_image = np.array(Image.open(args.stitched_image)).astype(float)

# grab corners
corners = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
corner_slices = compute_corner_slices(stitched_image)

plots = []
click_sources = []
corner_coords = [None] * 4
coord_display = Div(text="COPY/PASTE ME INTO PROCESSING NOTEBOOK: (None, None), (None, None), (None, None), (None, None)", styles={"font-size": "20px"})

image_sources = []  # To store the original cropped images
image_renderers = []  # To update image brightness later

for idx, s in enumerate(corner_slices):

    # Load and convert to NumPy
    corner_image = stitched_image[s]
    height, width = corner_image.shape
    x0, y0 = s[0].start, s[1].start
    dw, dh = s[1].stop - s[1].start, s[0].stop - s[0].start

    # Click source for recording clicks
    source = ColumnDataSource(data=dict(x=[], y=[], index=[]))
    click_sources.append(source)

    # Set up plot
    # TODO: make width and height more flexible
    p = figure(
        title=corners[idx],
        width=800, height=800,
        tools="tap",
        x_range=(s[1].start, s[1].stop),
        y_range=(s[0].stop, s[0].start)
    )

    # Store for brightness adjustments
    img_cropped_original = corner_image.copy()
    image_sources.append(img_cropped_original)

    renderer = p.image(image=[corner_image], x=y0, y=x0, dw=dw, dh=dh, palette="Greys256")
    image_renderers.append(renderer)

    p.scatter('x', 'y', size=5, color='red', alpha=0.6, source=source)
    
    def make_callback(i, src):
        def on_tap(event):
            # Save coordinate
            src.data = dict(x=[event.x], y=[event.y])

            y_offset = 0
            x_offset = 0
            corner_coords[i] = (event.x + x_offset, event.y + y_offset)

            formatted = "COPY/PASTE ME INTO PROCESSING NOTEBOOK: " + ", ".join(
                f"(({f'{pt[0]:.0f}' if pt is not None else 'None'}, {f'{pt[1]:.0f}' if pt is not None else 'None'}))"
                for pt in corner_coords
            )

            coord_display.text = formatted
        return on_tap

    p.on_event(Tap, make_callback(idx, source))

    plots.append(p)

def update_brightness(attr, old, new):
    scale = brightness_slider.value
    for i in range(4):
        tmp = image_sources[i].copy()
        adjusted = np.clip(tmp * scale, 0, 255)
        image_renderers[i].data_source.data = dict(
            image=[adjusted],
            x=[image_renderers[i].glyph.x],
            y=[image_renderers[i].glyph.y],
            dw=[image_renderers[i].glyph.dw],
            dh=[image_renderers[i].glyph.dh]
        )

brightness_slider = Slider(start=0.1, end=5.0, value=1.0, step=0.1, title="Brightness", width=800, styles={"font-size": "20px"})
brightness_slider.on_change("value", update_brightness)

# Arrange in a 2x2 grid
grid = gridplot([[plots[0], plots[1]], [plots[2], plots[3]]])
curdoc().add_root(column(grid, brightness_slider, coord_display))
