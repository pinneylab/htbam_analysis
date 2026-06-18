import numpy as np
import pathlib
import logging
from skimage import io, transform
from abc import abstractmethod, ABC
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from basicpy import BaSiC

import os
from typing import Tuple


def ff_subtract(i, ffi, ff_bval, ff_scale):
    ff_result = np.subtract(i, ff_bval) / np.subtract(ffi, ff_bval) * ff_scale
    # TODO figure out where these params are coming from
    result = np.clip(ff_result, 0, 65535).astype("uint16")
    return result


def rotate_image(img, rotation_val) -> np.array:
    rotation_params = {"resize": False, "clip": True, "preserve_range": True}
    return transform.rotate(img, rotation_val, **rotation_params).astype("uint16")


def find_imaging_csv(img_path: pathlib.Path) -> pathlib.Path:
    for parent in img_path.resolve().parents:
        csv_path = parent / "imaging.csv"
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(f"imaging.csv not found in parents of {img_path}")


class RasterParams:
    def __init__(
        self,
        overlap: float,
        size: int,
        acqui_ori: Tuple[int, int],
        rotation: float,
        dims: tuple,
        auto_ff: bool = True,
        ff_type: str = "BaSiC",
        group_feature=0,
    ):
        """
        Parameters describing a single image raster.

        Arguments:
            (float) overlap: overlap fraction (e.g., 0.1)
            (float) rotation: pre-stitch rotation to perform (%)
            (bool) auto_ff: flag to execute FF correction on stitch, if possible
            (int | float | string) group_feature: feature value for RasterGroup

        Returns:
            None

        """
        # This can never be reached!
        # if self._root:
        #     self._parent = list(pathlib.Path(root).parents)[0]
        self._size = size
        self._overlap = overlap
        self._rotation = rotation
        self._acqui_ori = acqui_ori
        self._group_feature = group_feature
        self._auto_ff = auto_ff
        self._ff_type = ff_type

        self._exposure = None
        self._channel = None
        self._parent = None
        self._dims = dims
        self._root = None

    def update_root(self, new_root):
        self._root = new_root
        self._parent = os.path.dirname(new_root)

    @property
    def size(self):
        return self._size

    @property
    def overlap(self):
        return self._overlap

    @property
    def rotation(self):
        return self._rotation

    @property
    def acqui_ori(self):
        return self._acqui_ori

    @property
    def group_feature(self):
        return self._group_feature

    @property
    def auto_ff(self):
        return self._auto_ff

    @property
    def ff_type(self):
        return self._ff_type

    @property
    def exposure(self):
        return self._exposure

    @property
    def channel(self):
        return self._channel

    @property
    def parent(self):
        return self._parent

    @property
    def dims(self):
        return self._dims

    @property
    def root(self):
        return self._root

    @property
    def name(self):
        return f"{self.channel}_{self.exposure}"

    def update_channel(self, new_channel):
        self._channel = new_channel

    def update_exposure(self, new_exposure):
        self._exposure = new_exposure

    def update_dims(self, new_dims):
        self._dims = new_dims

    def update_group_feature(self, new_group_feature):
        self._group_feature = new_group_feature


class Raster(ABC):
    def __init__(self, image_refs: list, params: RasterParams):
        """
        A collection of images (or references) for one rastered image at one set of acquisition
        parameters

        Arguments:
            (list) image_refs: an ordered list of rastered image paths
            (RasterParams) params: rastered image parameters

        Returns:
            None


        """
        self._image_refs = image_refs
        self._params = params
        self._images = None

    @abstractmethod
    def fetch_images(self):
        raise NotImplementedError("Raster Subclass should implement this!")

    @property
    def params(self):
        return self._params

    def apply_ff(self):
        """
        Applies flat-field correction to fetched images

        Arguments:
            None

        Returns:
            None

        """
        raise NotImplementedError("Functionality not ready yet.")
        channel = self._params.channel
        exposure = self._params.exposure
        ff_image = StitchingSettings.ff_images[channel]
        ff_params = StitchingSettings.ff_params[channel][exposure]
        ff_bval = ff_params[0]
        ff_scale = ff_params[1]

        return [ff_subtract(i, ff_image, ff_bval, ff_scale) for i in self._images]

    def applyFF_BaSiC(self, plot: bool):
        """
        Applies flat-field correction to fetched images using BaSiC. This technique simulates a FF and
        dark image based on common shared features across the full raster (e.g. 64 images).

        This version uses gaussian smoothing after fitting.

        Implemented in v2.1.0 by Peter Suzuki, Youngbin Lim

        Arguments:
            None

        Returns:
            None

        """
        # print("Running BaSiC for FF correction...")

        # Load images
        imgarray = []
        for img in self._images:
            # print(img.shape)
            imgarray.append(img)
        images = np.asarray(imgarray)

        # Initialize BaSiC and train
        basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
        basic.fit(images)

        # Smooth FF image and apply transform to each raw image
        ffImage = gaussian_filter(basic.flatfield, sigma=50)
        # StitchingSettings.ffImages[self.params.channel] = ffImage # save ffImage to settings

        ffbval = 0
        ffscale = 1

        manual_smoothed = [ff_subtract(i, ffImage, ffbval, ffscale) for i in imgarray]

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            im = axes[0].imshow(basic.flatfield)
            fig.colorbar(im, ax=axes[0])
            axes[0].set_title("Flatfield")
            im = axes[1].imshow(ffImage)
            fig.colorbar(im, ax=axes[1])
            axes[1].set_title("Smoothed FF")
            axes[2].plot(basic.baseline)
            axes[2].set_xlabel("Frame")
            axes[2].set_ylabel("Baseline")
            fig.tight_layout()
            plt.show()

        return manual_smoothed

    def applyFF_BaSiC_masked(self):
        """
        Applies flat-field correction to fetched images using BaSiC. This technique simulates a FF and
        dark image based on common shared features across the full raster (e.g. 64 images).

        Current version fits a gaussian to the pixels and only uses pixels within 1 s.d. of the mean to
        calculate BaSiC FF image. (Using a mask to cull which pixels are seen by BaSiC during training).
        After fitting, images can be transformed using the FF image either with or w/out smoothing, with
        default setting using gaussian smoothing to eliminate small (wrong) features learned by BaSiC.

        Implemented in v2.1.0 by Peter Suzuki, Youngbin Lim

        Arguments:
            None

        Returns:
            None

        """
        
        raise NotImplementedError("Functionality not ready yet.")

        print("Running BaSiC for FF correction, masking outlier pixels...")

        # Load images
        imgarray = []
        for img in self._images:
            # print(img.shape)
            imgarray.append(img)
        images = np.asarray(imgarray)

        ## Flatten and fit gaussian to background pixels
        plt.figure(figsize=(6, 2))

        data = images.flatten()
        y, x, _ = plt.hist(data, 100, alpha=0.3, label="data")
        x = (x[1:] + x[:-1]) / 2  # for len(x)==len(y)

        def gauss(x, mu, sigma, A):
            return A * np.exp(-((x - mu) ** 2) / 2 / sigma**2)

        expected = (30000, 5000, 1e7)  # , 20000, 5000, 125)
        params, cov = curve_fit(gauss, x, y, expected)
        x_fit = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_fit, gauss(x_fit, *params), color="red", lw=3, label="model")
        plt.legend()
        plt.title("Pixel intensity distribution")
        plt.show()

        # Set mask bounds to train only on background
        top = params[0] + abs(3 * params[1])  # mean + 3sd
        bottom = params[0] - abs(3 * params[1])
        mask = np.zeros(images.shape)
        mask = (images < top) & (
            images > bottom
        )  # threshold for defining bright things defined by mean + 3 s.d.
        print(params)
        print(top)
        print(bottom)
        # masked = mask*images

        # Initialize BaSiC and train
        basic_mask = BaSiC(get_darkfield=False, smoothness_flatfield=1)
        basic_mask.fit(images, fitting_weight=mask)

        # Smooth FF image and apply transform to each raw image
        ffImage = gaussian_filter(basic_mask.flatfield, sigma=50)
        # StitchingSettings.ffImages[self.params.channel] = ffImage # save ffImage to settings

        ffbval = 0
        ffscale = 1

        manual_smoothed = [ff_subtract(i, ffImage, ffbval, ffscale) for i in imgarray]

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        im = axes[0].imshow(basic_mask.flatfield)
        fig.colorbar(im, ax=axes[0])
        axes[0].set_title("Flatfield")
        im = axes[1].imshow(ffImage)
        fig.colorbar(im, ax=axes[1])
        axes[1].set_title("Smoothed FF")
        axes[2].plot(basic_mask.baseline)
        axes[2].set_xlabel("Frame")
        axes[2].set_ylabel("Baseline")
        fig.tight_layout()
        plt.show()

        return manual_smoothed

    def stitch(self, method="cut", plot: bool = False):
        """
        Wrapper for image stitching method selection.
        TODO: Implement 'overlap' method

        Arguments:
            (str) method: stitch method ('cut' | 'overlap' | 'smart' | 'rescue')

        Returns:
            (np.ndarray) A stitched image array

        """

        self.fetch_images()
        if method == "cut":
            return self.cut_stitch(plot)
        elif method == "smart":
            return self.smart_stitch()
        elif method == "overlap":
            return self.overlap_stitch()
        elif method == "rescue":
            return self.coordinate_stitch(plot)
        else:
            raise ValueError(
                'Invalid stitch method. Valid methods are "cut", "smart", "overlap" and "rescue"'
            )

    def smart_stitch(self):
        """
        Returns:
            (np.ndarray) A stitched image array
        """
        raise NotImplementedError("Functionality not ready yet.")

    def cut_stitch(self, plot: bool):
        """
        Stitches a raster via the 'cut' method. Trims borders according to image overlap and
        concatenates along both axes. If RasterParameters.auto_ff is True, performs
        flat-field correction prior to stitching.

        Arguments:
            None

        Returns:
            (np.ndarray) A stitched image array

        """
        imsize = self.params.size
        margin = int(imsize * self.params.overlap / 2)  # Edge dim to trim
        retained = imsize - 2 * margin

        if self.params.overlap < 0 or self.params.overlap >= 1:
            raise ValueError("Overlap must be ≥ 0 and < 1")

        # Catch no overlap case
        if margin == 0:
            border = slice(0, None)
        else:
            border = slice(margin, -margin)

        tiles = self._images
        if self.params.auto_ff and self.params.ff_type == "BaSiC":
            tiles = self.ffCorrectedImages = self.applyFF_BaSiC(plot)
            # print("completed BaSiC FF correction")

            # logging.info(
            #     "BaSiC Flat-Field Corrected Image | Ch: {}, Exp: {}".format(
            #         self.params.channel, self.params.exposure
            #     )
            # )

        # elif self.params.auto_ff and self.params.ff_type == "BaSiC_masked":
        #     tiles = self.ffCorrectedImages = self.applyFF_BaSiC_masked()
        #     print("completed BaSiC FF correction with mask")

        #     logging.info(
        #         "BaSiC Flat-Field Corrected Image | Ch: {}, Exp: {}".format(
        #             self.params.channel, self.params.exposure
        #         )
        #     )

        trimmedTiles = [tile[border, border] for tile in tiles]  # Trim
        tileArray = np.asarray(trimmedTiles)

        arrangedTiles = np.reshape(
            tileArray, (self.params.dims[0], self.params.dims[1], retained, retained)
        )
        if self.params.acqui_ori[0]:  # if origin on on right, flip horizontally (view returned)
            arrangedTiles = np.flip(arrangedTiles, 0)
        if self.params.acqui_ori[1]:  # If origin on bottom, flip vertically (view returned)
            arrangedTiles = np.flip(arrangedTiles, 1)
        rowsStitched = np.concatenate(arrangedTiles, axis=2)  # Stitch rows
        fullStitched = np.concatenate(rowsStitched, axis=0)  # Stitch cols
        return fullStitched

    def overlap_stitch(self):
        """
        #TODO: re-implement overlap stitching method

        """
        raise NotImplementedError("Overlap Stitch not yet implemented")

    def coordinate_stitch(self, plot: bool = False):
        """
        Stitches a raster via stage coordinates from imaging.csv,
        calibrating the pixel size dynamically using a single high-contrast boundary.
        Used primarily for "rescuing" stitched images with poor overlap,
        do to settings or microscope stage errors.

        Returns:
            (np.ndarray) A stitched image array
        """
        import pandas as pd
        from skimage.registration import phase_cross_correlation

        imsize = self.params.size
        overlap = self.params.overlap
        expected_overlap = int(imsize * overlap)

        # Load images
        tiles_list = self._images
        if self.params.auto_ff and self.params.ff_type == "BaSiC":
            tiles_list = self.ffCorrectedImages = self.applyFF_BaSiC(plot)

        # Automatically locate imaging.csv
        # Pulling image.csv from parents is a little yeehaw. Adjust in the future?
        ref_path = pathlib.Path(self._image_refs[0])
        csv_path = find_imaging_csv(ref_path)
        csv_dir = csv_path.parent

        df = pd.read_csv(csv_path)

        # Map to absolute paths for robust matching
        df['abs_path'] = df['image_path'].apply(lambda p: os.path.abspath(csv_dir / p))
        ref_abs = [os.path.abspath(p) for p in self._image_refs]

        df_matched = df[df['abs_path'].isin(ref_abs)].copy()
        df_matched.set_index('abs_path', inplace=True)
        df_matched = df_matched.reindex(ref_abs).reset_index()

        cols_count = self.params.dims[0]
        rows_count = self.params.dims[1]

        # Reconstruct tiles dictionary and coordinates grid
        tiles = {}
        xs = np.zeros((cols_count, rows_count))
        ys = np.zeros((cols_count, rows_count))
        
        for idx, row in df_matched.iterrows():
            c, r = int(row['raster_col_index']), int(row['raster_row_index'])
            tiles[(c, r)] = tiles_list[idx]
            xs[c, r] = row['x']
            ys[c, r] = row['y']

        # Single boundary calibration
        best_std = -1
        best_boundary = None  # Will store (type, c, r, strip1, strip2, dx_stage)
        
        # 1. Search horizontal boundaries
        for r in range(rows_count):
            for c in range(cols_count - 1):
                t1 = tiles[(c, r)].astype(float)
                t2 = tiles[(c+1, r)].astype(float)
                
                crop_margin = int(imsize * 0.15)
                strip1 = t1[crop_margin:-crop_margin, -expected_overlap:]
                strip2 = t2[crop_margin:-crop_margin, :expected_overlap]
                
                std_val = (np.std(strip1) + np.std(strip2)) / 2
                if std_val > best_std:
                    best_std = std_val
                    dx_stage = abs(xs[c+1, r] - xs[c, r])
                    best_boundary = ('H', c, r, strip1, strip2, dx_stage)
                    
        # 2. Search vertical boundaries
        for c in range(cols_count):
            for r in range(rows_count - 1):
                t1 = tiles[(c, r)].astype(float)
                t2 = tiles[(c, r+1)].astype(float)
                
                crop_margin = int(imsize * 0.15)
                strip1 = t1[-expected_overlap:, crop_margin:-crop_margin]
                strip2 = t2[:expected_overlap, crop_margin:-crop_margin]
                
                std_val = (np.std(strip1) + np.std(strip2)) / 2
                if std_val > best_std:
                    best_std = std_val
                    dy_stage = abs(ys[c, r+1] - ys[c, r])
                    best_boundary = ('V', c, r, strip1, strip2, dy_stage)

        pixel_size = None
        if best_boundary is not None and expected_overlap > 0:
            b_type, c, r, strip1, strip2, d_stage = best_boundary
            try:
                shift, error, diffphase = phase_cross_correlation(strip1, strip2, upsample_factor=1)
                if b_type == 'H':
                    d_pixel = (imsize - expected_overlap) + shift[1]
                else:
                    d_pixel = (imsize - expected_overlap) + shift[0]
                
                cand = d_stage / abs(d_pixel)
                if 3.0 < cand < 3.5:
                    pixel_size = cand
            except Exception:
                pass

        if pixel_size is None:
            raise ValueError(f"Could not dynamically assign pixel size for {self.params.name}. ")
        else:
            print(f"Dynamic pixel size assignment for {self.params.name}: {pixel_size}")

        # Determine signs based on acquisition origin
        sign_x = -1 if self.params.acqui_ori[0] else 1
        sign_y = -1 if self.params.acqui_ori[1] else 1

        # Calculate pixel coordinates
        u = sign_x * xs / pixel_size
        v = sign_y * ys / pixel_size

        u_min = np.min(u)
        v_min = np.min(v)

        u_offsets = u - u_min
        v_offsets = v - v_min

        canvas_w = int(np.round(np.max(u_offsets))) + imsize
        canvas_h = int(np.round(np.max(v_offsets))) + imsize

        canvas = np.zeros((canvas_h, canvas_w), dtype=tiles_list[0].dtype)

        # Initialize crop arrays
        trim_left = np.zeros((cols_count, rows_count), dtype=int)
        trim_right = np.zeros((cols_count, rows_count), dtype=int)
        trim_top = np.zeros((cols_count, rows_count), dtype=int)
        trim_bottom = np.zeros((cols_count, rows_count), dtype=int)

        # Calculate horizontal trims
        for r in range(rows_count):
            for c in range(cols_count - 1):
                ol = imsize - abs(u_offsets[c+1, r] - u_offsets[c, r])
                if ol > 0:
                    trim_amount = max(0, int(np.round(ol / 2)) - 1)
                    if u_offsets[c, r] < u_offsets[c+1, r]:
                        trim_right[c, r] = trim_amount
                        trim_left[c+1, r] = trim_amount
                    else:
                        trim_left[c, r] = trim_amount
                        trim_right[c+1, r] = trim_amount

        # Calculate vertical trims
        for c in range(cols_count):
            for r in range(rows_count - 1):
                ol = imsize - abs(v_offsets[c, r+1] - v_offsets[c, r])
                if ol > 0:
                    trim_amount = max(0, int(np.round(ol / 2)) - 1)
                    if v_offsets[c, r] < v_offsets[c, r+1]:
                        trim_bottom[c, r] = trim_amount
                        trim_top[c, r+1] = trim_amount
                    else:
                        trim_top[c, r] = trim_amount
                        trim_bottom[c, r+1] = trim_amount

        # Paste tiles
        for c in range(cols_count):
            for r in range(rows_count):
                u0 = int(np.round(u_offsets[c, r]))
                v0 = int(np.round(v_offsets[c, r]))

                t_l = max(0, min(imsize // 2, trim_left[c, r]))
                t_r = max(0, min(imsize // 2, trim_right[c, r]))
                t_t = max(0, min(imsize // 2, trim_top[c, r]))
                t_b = max(0, min(imsize // 2, trim_bottom[c, r]))

                tile = tiles[(c, r)]
                cropped = tile[t_t : imsize - t_b, t_l : imsize - t_r]

                canvas[v0 + t_t : v0 + imsize - t_b, u0 + t_l : u0 + imsize - t_r] = cropped

        return canvas

    def export_stitch(
        self, method="cut", out_path_name="StitchedImages", manual_target=None
    ):
        """
        Perform stitching and export raster.

        Arguments:
            (str) method: stitch method ('cut' | 'overlap')
            (str) out_path_name: Name of folder to house stitched raster. Typically 'StitchedImages'

        Returns:
            None

        """
        stitchedRaster = self.stitch(method=method)

        features = [
            self._params.exposure,
            self._params.channel,
            self._params.group_feature,
        ]
        rasterName = "StitchedImg_{}_{}_{}.tif".format(*features)
        if manual_target:
            stitchDir = pathlib.Path(manual_target)
        else:
            stitchDir = pathlib.Path(os.path.join(self._params.parent, out_path_name))
        stitchDir.mkdir(exist_ok=True)
        outDir = os.path.join(stitchDir, rasterName)
        io.imsave(outDir, stitchedRaster, check_contrast=False)#, plugin="tifffile", check_contrast=False)
        logging.debug("Stitching Complete")

    def __lt__(self, other):
        selfstem = pathlib.Path(self.image_refs[0]).stem
        otherstem = pathlib.Path(other.image_refs[0]).stem
        return selfstem < otherstem


class FlatRaster(Raster):
    def __init__(self, image_refs, params):
        super().__init__(image_refs, params)

    def fetch_images(self):
        """
        Fetches (loads into memory) and rotates images (if indicated by raster parameters)

        Arguments:
            None

        Returns:
            None

        """
        # r = self._params.rotation
        #
        # images = [io.imread(img) for img in self.image_refs]
        #
        # if r:
        #     images = [rotate_image(img, self._params.rotation) for img in images]
        #
        # self._images = images

        self._images = [io.imread(img) for img in self._image_refs]

        if self._params.rotation:
            # if rotation value provided, rotate all images
            self._images = [
                rotate_image(img, self._params.rotation) for img in self._images
            ]

