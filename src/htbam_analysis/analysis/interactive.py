import base64
import io
import numpy as np
import plotly.graph_objs as go
import ipywidgets as widgets
from PIL import Image
from typing import Dict, Union, Tuple
from IPython.display import display

class ImageMaskPicker:
    def __init__(self, 
                 images: Dict[str, Union[str, np.ndarray]], 
                 n_cols: int = 32, 
                 n_rows: int = 56,
                 figsize: Tuple[int, int] = (800, 800)):
        """
        Interactive Mask Picker for Jupyter Notebooks.
        
        Args:
            images: A dictionary mapping image names to either a file path or a numpy array.
            n_cols: Number of columns in the device grid.
            n_rows: Number of rows in the device grid.
            figsize: Dimensions of the Plotly FigureWidget (width, height).
        """
        self.images_raw = {}
        for k, v in images.items():
            if isinstance(v, str):
                self.images_raw[k] = np.array(Image.open(v)).astype(np.float32)
            else:
                self.images_raw[k] = v.astype(np.float32)
                
        self.image_names = list(self.images_raw.keys())
        self.current_img_name = self.image_names[0]
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.figsize = figsize
        
        # Determine image dimensions from the first image
        self.img_h, self.img_w = self.images_raw[self.current_img_name].shape[:2]
        self.cell_w = self.img_w / self.n_cols
        self.cell_h = self.img_h / self.n_rows
        
        self.bad_chambers = set() # Store as (row, col) 0-indexed
        self.history = []
        self._updating_widgets = False
        
        self._build_widgets()
        self._load_current_image()
        self._update_image_trace()
        self._draw_shapes()

    def _build_widgets(self):
        self.dropdown = widgets.Dropdown(
            options=self.image_names,
            value=self.current_img_name,
            description='Image:',
            layout=widgets.Layout(width='auto')
        )
        self.dropdown.observe(self._on_image_change, names='value')
        
        self.slider_low = widgets.FloatSlider(description='Low', layout=widgets.Layout(width='400px'))
        self.slider_high = widgets.FloatSlider(description='High', layout=widgets.Layout(width='400px'))
        
        self.slider_low.observe(self._on_contrast_change, names='value')
        self.slider_high.observe(self._on_contrast_change, names='value')
        
        self.btn_reset_contrast = widgets.Button(description='Reset Contrast')
        self.btn_reset_contrast.on_click(self._on_reset_contrast)
        
        self.btn_clear = widgets.Button(description='Clear All', button_style='danger')
        self.btn_clear.on_click(self._on_clear)
        
        self.btn_undo = widgets.Button(description='Undo', button_style='warning')
        self.btn_undo.on_click(self._on_undo)
        
        # Layout
        self.controls = widgets.VBox([
            self.dropdown,
            widgets.HBox([self.slider_low, self.slider_high, self.btn_reset_contrast]),
            widgets.HBox([self.btn_undo, self.btn_clear])
        ])
        
        self.fig = go.FigureWidget()
        self.fig.layout.width = self.figsize[0]
        self.fig.layout.height = self.figsize[1]
        self.fig.layout.margin = dict(l=0, r=0, t=30, b=0)
        
        # Configure axes to match image dimensions
        self.fig.layout.xaxis.visible = False
        self.fig.layout.yaxis.visible = False
        self.fig.layout.xaxis.range = [-self.cell_w, self.img_w]
        self.fig.layout.yaxis.range = [self.img_h, -self.cell_h] # Reversed for images
        
        self.fig.layout.dragmode = 'pan'
        self.fig.layout.title = 'Click chambers to toggle FAIL | Drag to Pan/Zoom'
        
        # Add empty image trace - will be updated with source
        self.fig.add_trace(go.Image())
        self.fig.data[0].on_click(self._on_click)

    def _load_current_image(self):
        img = self.images_raw[self.current_img_name]
        
        # Determine image min/max
        self.img_min = float(np.nanmin(img))
        self.img_max = float(np.nanmax(img))
        if self.img_min == self.img_max:
            self.img_max = self.img_min + 1.0
            
        self.default_vmin = float(np.nanpercentile(img, 1.0))
        self.default_vmax = float(np.nanpercentile(img, 99.0))
        
        self._updating_widgets = True
        try:
            self.slider_low.min = self.img_min
            self.slider_low.max = self.img_max
            self.slider_low.value = self.default_vmin
            
            self.slider_high.min = self.img_min
            self.slider_high.max = self.img_max
            self.slider_high.value = self.default_vmax
        finally:
            self._updating_widgets = False

    def _on_image_change(self, change):
        if self._updating_widgets: return
        self.current_img_name = change.new
        
        # Check if dimensions changed
        img = self.images_raw[self.current_img_name]
        new_h, new_w = img.shape[:2]
        if new_h != self.img_h or new_w != self.img_w:
            self.img_h, self.img_w = new_h, new_w
            self.cell_w = self.img_w / self.n_cols
            self.cell_h = self.img_h / self.n_rows
            
            self.fig.layout.xaxis.range = [-self.cell_w, self.img_w]
            self.fig.layout.yaxis.range = [self.img_h, -self.cell_h]
            
            # Must redraw shapes to match new dimensions
            self._draw_shapes()
            
        self._load_current_image()
        self._update_image_trace()

    def _on_contrast_change(self, change):
        if self._updating_widgets: return
        self._update_image_trace()
        
    def _on_reset_contrast(self, b):
        self._updating_widgets = True
        try:
            self.slider_low.value = self.default_vmin
            self.slider_high.value = self.default_vmax
        finally:
            self._updating_widgets = False
        self._update_image_trace()
        
    def _on_clear(self, b):
        self.bad_chambers.clear()
        self.history.clear()
        self._draw_shapes()
        
    def _toggle_chambers(self, chambers_to_toggle):
        added = set()
        removed = set()
        for key in chambers_to_toggle:
            if key in self.bad_chambers:
                self.bad_chambers.remove(key)
                removed.add(key)
            else:
                self.bad_chambers.add(key)
                added.add(key)
        if added or removed:
            self.history.append({'added': added, 'removed': removed})
        self._draw_shapes()

    def _on_undo(self, b):
        if not self.history:
            return
        last_action = self.history.pop()
        self.bad_chambers -= last_action['added']
        self.bad_chambers |= last_action['removed']
        self._draw_shapes()

    def _update_image_trace(self):
        img = self.images_raw[self.current_img_name]
        low = self.slider_low.value
        high = self.slider_high.value
        
        if low >= high:
            return
        
        # Apply contrast and cast to uint8
        img_scaled = np.clip((img - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
        
        pad_h = int(self.cell_h)
        pad_w = int(self.cell_w)
        padded_img = np.full((img_scaled.shape[0] + pad_h, img_scaled.shape[1] + pad_w), 128, dtype=np.uint8)
        padded_img[pad_h:, pad_w:] = img_scaled
        
        # Convert to PIL Image and save to BytesIO
        pil_img = Image.fromarray(padded_img, mode='L')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Update trace
        self.fig.data[0].source = f"data:image/png;base64,{b64_str}"
        self.fig.data[0].x0 = -pad_w
        self.fig.data[0].y0 = -pad_h
        self.fig.data[0].dx = 1
        self.fig.data[0].dy = 1

    def _draw_shapes(self):
        shapes = []
        
        # Grid lines (draw over the image padding too!)
        for c in range(self.n_cols + 1):
            x = c * self.cell_w
            shapes.append(dict(
                type='line', x0=x, y0=-self.cell_h, x1=x, y1=self.img_h,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ))
        for r in range(self.n_rows + 1):
            y = r * self.cell_h
            shapes.append(dict(
                type='line', x0=-self.cell_w, y0=y, x1=self.img_w, y1=y,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ))
            
        # Draw the top-left empty corner to distinguish it
        shapes.append(dict(
            type='rect', x0=-self.cell_w, y0=-self.cell_h, x1=0, y1=0,
            fillcolor='rgba(0,0,0,1)', line=dict(width=0)
        ))
            
        # Bad chambers
        for (r, c) in self.bad_chambers:
            x0 = c * self.cell_w
            y0 = r * self.cell_h
            x1 = x0 + self.cell_w
            y1 = y0 + self.cell_h
            shapes.append(dict(
                type='rect', x0=x0, y0=y0, x1=x1, y1=y1,
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(width=0)
            ))
            
        self.fig.layout.shapes = shapes

    def _on_click(self, trace, points, state):
        if not points.xs:
            return
            
        x = points.xs[0]
        y = points.ys[0]
        
        # Check row header
        if x < 0 and 0 <= y < self.img_h:
            row = int(y // self.cell_h)
            row_chambers = {(row, c) for c in range(self.n_cols)}
            # If all are bad, we toggle them to good (remove). Else add all.
            if row_chambers.issubset(self.bad_chambers):
                self._toggle_chambers(row_chambers)
            else:
                to_toggle = row_chambers - self.bad_chambers
                self._toggle_chambers(to_toggle)
            return
            
        # Check col header
        if y < 0 and 0 <= x < self.img_w:
            col = int(x // self.cell_w)
            col_chambers = {(r, col) for r in range(self.n_rows)}
            if col_chambers.issubset(self.bad_chambers):
                self._toggle_chambers(col_chambers)
            else:
                to_toggle = col_chambers - self.bad_chambers
                self._toggle_chambers(to_toggle)
            return
        
        col = int(x // self.cell_w)
        row = int(y // self.cell_h)
        
        if col < 0 or col >= self.n_cols or row < 0 or row >= self.n_rows:
            return
            
        self._toggle_chambers({(row, col)})

    def show(self):
        """Display the widget in the Jupyter Notebook."""
        display(widgets.VBox([self.controls, self.fig]))

    def get_mask(self, data, mask_name='manual_qc_mask'):
        """
        Creates a DataND mask object based on the current selection.
        All selected (FAIL) chambers will be False, and passing chambers True.
        
        Args:
            data: A DataND object (Data4D, Data3D, Data2D) to align shapes with.
            mask_name: Metadata identifier for the created mask.
            
        Returns:
            DataND object representing the custom mask.
        """
        from htbam_analysis.analysis.filter import make_custom_mask
        
        n_conc = data.dep_var.shape[0]
        n_chambers = data.dep_var.shape[1]
        chamber_ids = data.indep_vars.chamber_IDs
        
        chamber_mask = np.ones(n_chambers, dtype=bool)
        for i, cid in enumerate(chamber_ids):
            parts = cid.split(',')
            c = int(parts[0]) - 1 # chamber IDs are 1-indexed string 'col,row'
            r = int(parts[1]) - 1
            if (r, c) in self.bad_chambers:
                chamber_mask[i] = False
                
        # The filter mask expects shape (n_conc, n_chambers, 1)
        full_mask = np.expand_dims(np.tile(chamber_mask, (n_conc, 1)), axis=-1)
        return make_custom_mask(data, full_mask, info=mask_name)
