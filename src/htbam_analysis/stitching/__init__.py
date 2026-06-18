# -*- coding: utf-8 -*-

"""Top-level package for stitching."""

from htbam_analysis.stitching import rastering

__author__ = """Jonathan Zhang"""
__email__ = ""
__version__ = "1.3.0"


import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io, filters, transform
from pathlib import Path
from typing import Tuple, Union, List, Optional
import matplotlib.pyplot as plt
import warnings
import fnmatch

def _get_grid_angle(image_path: Path, debug=False) -> float:
    """Helper function to find the grid rotation angle of a single raw image."""
    img = io.imread(image_path)
    
    # We only care about the center part to speed up and avoid edge effects
    h, w = img.shape
    crop = img[h//4:3*h//4, w//4:3*w//4]
    
    # Normalize the crop to handle different exposures
    crop = crop.astype(float)
    ptp = crop.max() - crop.min()
    
    if ptp < 1000:
        raise ValueError("Image lacks sufficient contrast/features (basically flat)")
        
    # make a plot of the raw crop
    if debug:
        plt.figure()
        plt.imshow(crop)
        plt.title("Raw Crop")
        plt.show()
    
    crop = (crop - crop.min()) / ptp
        
    ## We can consider doing edge detection on smoothed crop to suppress high-frequency noise
    ## Instead, to filter out images that are nearly all noise, we're requiring an absolute contrast of 1000 above.
    #blurred = filters.gaussian(crop, sigma=5.0)
    #edges = filters.sobel(blurred)
    edges = filters.sobel(crop)

    # make a plot of the edges
    if debug:
        plt.figure()
        plt.imshow(edges)
        plt.title("Edges")
        plt.show()

    # Does our image have structured edges? Or is it mostly featureless/noise?
    # Here, we check that at least 1% of the image has edges
    if np.mean(edges > 0.05) < 0.01:
        raise ValueError("No edges detected in image")
    
    # Test angles from -5 to 5 degrees
    angles = np.linspace(-5, 5, 200, endpoint=False)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", 
            category=UserWarning, 
            message="Radon transform: image must be zero outside the reconstruction circle"
        )
        sinogram = transform.radon(edges, theta=angles, circle=True)
    
    variances = np.var(sinogram, axis=0)

    # # Plot sinogram:
    # if debug:
    #     plt.figure()
    #     plt.imshow(sinogram, extent=[angles.min(), angles.max(), 0, sinogram.shape[0]], aspect="auto")
    #     plt.title("Sinogram")
    #     plt.xlabel("Angle (degrees)")
    #     plt.ylabel("Projection")
    #     plt.show()
    
    # Check if a clear dominant grid line angle was found
    peak_prominence = np.max(variances) / (np.median(variances) + 1e-8)
    if peak_prominence < 1.1:
        raise ValueError(f"No distinct grid angle detected (prominence: {peak_prominence:.3f})")

    # Make a plot of prominance vs angle
    if debug:
        plt.figure()
        plt.plot(angles, variances)
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Prominence")
        plt.title("Prominence vs Angle")
        plt.show()
        
    best_angle = angles[np.argmax(variances)]

    if best_angle < -4.9 or best_angle > 4.9:
        raise ValueError(f"Failed to find a valid rotation angle between -5, 5 degrees. This could be because the real rotation is too large, or because auto-rotation finding failed on your images.")
    
    # Return negative angle because of rotation sign convention
    return -best_angle


def stitch_single_raster(
    raster: List[Path],
    raster_params: rastering.RasterParams,
    stitched_image_path: Path,
    method: str = "cut",
) -> Tuple[bool, Path, str]:
    """
    Process a single raster group (directory + list of image paths).
    
    Args:
        args: Tuple of (dir, raster_list, outdir, raster_params)
    
    Returns:
        Tuple of (success: bool, output_file: Path, error_msg: str)
    """

    n_raster = int(raster_params.dims[0] * raster_params.dims[1])
    assert len(raster) == n_raster
    
    # Create the raster and stitch
    raster_obj = rastering.FlatRaster(
        image_refs=raster,
        params=raster_params
    )
    stitched_image = raster_obj.stitch(method=method)
    
    # Save the stitched image
    io.imsave(stitched_image_path, stitched_image, check_contrast=False)#, plugin="tifffile"
        
    return True, stitched_image_path, None


class ImageStitcher:

    def __init__(
            self,
            root_path,
            ):

        self.root_path = root_path
        self._raw_images_path = root_path / 'raw_images'
        self._stitched_images_path = root_path / 'stitched_images'

        self.raster_data = self.load_raster_data()
        self.stitched_image_data = None
        
    def load_raster_data(self):

        raster_data = pd.read_csv(self.root_path / 'imaging.csv')
    
        # make absolute image paths
        raster_data['image_path'] = raster_data['image_path'].apply(Path)
        raster_data['image_path'] = raster_data['image_path'].apply(lambda p: self.root_path / p)
        raster_data['image_path_parent'] = raster_data['image_path'].apply(lambda p: p.parent)

        # force integer type for width and height of rasters
        raster_data['raster_width'] = raster_data['raster_width'].astype(int)
        raster_data['raster_height'] = raster_data['raster_height'].astype(int)

        # configure flat-field correction (default is False)
        # TODO: add methods to help override defaults
        raster_data['apply_ff_correction'] = [False] * len(raster_data)

        # sort
        raster_data.sort_values(by=['image_path_parent', 'raster_col_index', 'raster_row_index'], inplace=True)

        return raster_data

    def find_optimal_rotation(self, raster_path: Union[Path, str], max_samples: int = 5) -> float:
        """
        Automatically determine the optimal rotation for a given raster
        by analyzing the grid angle of a few sample raw images.
        """
        raster_path = Path(raster_path) if not isinstance(raster_path, Path) else raster_path
        mask = self.raster_data['image_path_parent'] == raster_path
        df = self.raster_data[mask]
        
        if df.empty:
            raise ValueError(f"No raster data found for {raster_path}")
            
        # Try to sample from the center of the raster to avoid edge artifacts
        try:
            width = df['raster_width'].iloc[0]
            height = df['raster_height'].iloc[0]
            center_df = df[
                (df['raster_col_index'] > 0) & (df['raster_col_index'] < width - 1) &
                (df['raster_row_index'] > 0) & (df['raster_row_index'] < height - 1)
            ]
            if not center_df.empty:
                # Use all center images, shuffled
                sample_images = center_df['image_path'].sample(frac=1.0, random_state=42).to_list()
            else:
                sample_images = df['image_path'].sample(frac=1.0, random_state=42).to_list()
        except KeyError:
            # Fallback if raster_col_index or width aren't present
            sample_images = df['image_path'].sample(frac=1.0, random_state=42).to_list()
        
        angles = []
        for img_path in sample_images:
            try:
                angle = _get_grid_angle(img_path)
                angles.append(angle)
                if len(angles) >= max_samples:
                    break
            except Exception as e:
                print(f"Warning: Failed to compute grid angle for {img_path.name}: {e}")
                
        if not angles:
            raise ValueError(f"Failed to auto-detect rotation from any sample in {raster_path}")
            
        return float(np.median(angles))

    def stitch_images(self, 
        rotation: Optional[float] = None, 
        get_rotation_from: str = '*', 
        acqui_origin: Tuple[bool] = (True, False),
        method: str = "cut"
    ):

        assert isinstance(self.raster_data, pd.DataFrame), 'Raster data has not been set.'

        # TODO: make this not hard-coded
        SIZE = 1600
        raster_headers = ['image_path', 'image_path_parent', 'raster_index', 'x', 'y', 'z', 'frame_time', 'raster_width', 'raster_height', 'raster_overlap', 'raster_row_index', 'raster_col_index']
        stitched_image_data = []

        grouped = self.raster_data.groupby('image_path_parent')

        if rotation is None:
            matched_group_paths = [
                gp for gp in grouped.groups.keys()
                if fnmatch.fnmatch(gp.name, get_rotation_from) or 
                   fnmatch.fnmatch(str(gp.relative_to(self._raw_images_path)), get_rotation_from)
            ]
            if not matched_group_paths:
                raise ValueError(f"No raster groups matched the pattern '{get_rotation_from}'. Available groups: {[gp.name for gp in grouped.groups.keys()]}")
            
            for group_path in matched_group_paths:
                print(f"Auto-detecting optimal rotation from raster: {group_path}")
                try:
                    rotation = self.find_optimal_rotation(group_path)
                    print(f"Using optimal rotation: {rotation:.3f} degrees from {group_path}")
                    break
                except ValueError as e:
                    print(f"Warning: {e}. Trying next raster group...")
            
            if rotation is None:
                raise ValueError(f"Failed to auto-detect rotation from any matching raster group (pattern: '{get_rotation_from}'). \n You may want to manually provide a rotation angle by setting the 'rotation' parameter in stitch_images().")

        success_counter = 0
        for image_parent, df in tqdm(grouped, desc='Stitching images.'):

            try:
                # format outpath and make sure directory to write it to exists
                outpath = self._stitched_images_path / image_parent.relative_to(self._raw_images_path).with_suffix('.tif')
                outpath.parent.mkdir(parents=True, exist_ok=True)

                # extract params, stitch image
                overlap, width, height, ff_correction = df[['raster_overlap', 'raster_width', 'raster_height', 'apply_ff_correction']].iloc[0]
                width, height = int(width), int(height)
                raster = df['image_path'].to_list()
                params = rastering.RasterParams(overlap, size=SIZE, acqui_ori=acqui_origin, rotation=rotation, dims=(width, height), auto_ff=ff_correction, ff_type='BaSiC') 
                stitch_single_raster(raster, params, outpath, method=method)

                # carry over metadata
                row = df.drop_duplicates(subset=['image_path_parent']).drop(columns=raster_headers)
                row['image_path'] = outpath.relative_to(self._stitched_images_path)
                stitched_image_data.append(row)
                success_counter += 1

            except Exception as e:
                print('Failed stitching of {}: {}'.format(image_parent, e))

        print('Successfully stitched {} / {} images.'.format(success_counter, len(grouped)))

        self.stitched_image_data = pd.concat(stitched_image_data).reset_index(drop=True)
        self.stitched_image_data.to_csv(self._stitched_images_path / 'stitched_images.csv', index=False)

    def test_stitching_rotations(self, raster_path: Path, rotations: List[float], acqui_origin: Tuple[bool, bool], outdir: Path):
        """
        Test stitching a specific raster with different rotation parameters.
        
        Args:
            raster_path: Path to the directory containing the raster images
            rotations: List of rotation angles to test
            acqui_ori: Acquisition origin tuple (e.g., (True, False))
            output_dir: Directory to save the stitched images
        """
        # Filter raster data for this specific raster
        raster_path = Path(raster_path) if not isinstance(raster_path, Path) else raster_path
        mask = self.raster_data['image_path_parent'] == raster_path
        df = self.raster_data[mask]
        
        if df.empty:
            raise ValueError(f"No raster data found for {raster_path}")
        
        # Extract parameters
        overlap, width, height, ff_correction = df[['raster_overlap', 'raster_width', 'raster_height', 'apply_ff_correction']].iloc[0]
        width, height = int(width), int(height)
        raster = df['image_path'].to_list()
        
        # Hard-coded size (consistent with stitch_images method)
        SIZE = 1600
        
        # Ensure output directory exists
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Stitch for each rotation
        for rotation in rotations:
            params = rastering.RasterParams(
                overlap, 
                size=SIZE, 
                acqui_ori=acqui_origin, 
                rotation=rotation, 
                dims=(width, height), 
                auto_ff=ff_correction, 
                ff_type='BaSiC'
            )
            
            # Create output path with rotation in filename
            outpath = outdir / f"stitched_rotation_{rotation}.tif"
            
            # Stitch and save
            success, stitched_path, error_msg = stitch_single_raster(raster, params, outpath)
            
            if success:
                print(f"Successfully stitched with rotation {rotation}: {stitched_path}")
            else:
                print(f"Failed to stitch with rotation {rotation}: {error_msg}")


def backgroud_subtract(background_image, target_image):

    # TODO: figure out why on earth this is hard-coded
    MAX_VALUE = 65535

    subtracted = np.subtract(target_image.astype("float"), background_image.astype("float"))
    subtracted_clipped = np.clip(subtracted, 0, MAX_VALUE).astype("uint16")
    
    return subtracted_clipped


class BackgroundSubtractor:

    def __init__(self, root_path):

        self.root_path = root_path
        self._stitched_image_path = root_path / 'stitched_images'
        self._bgsub_images_path = root_path / 'bgsub_images'

        self.stitched_image_data = self.load_stitched_image_data()
        self.bgsub_data = None

    def load_stitched_image_data(self):

        stitched_image_data = pd.read_csv(self._stitched_image_path / 'stitched_images.csv')

        # make absolute image paths
        stitched_image_data['image_path'] = stitched_image_data['image_path'].apply(Path)
        stitched_image_data['image_path'] = stitched_image_data['image_path'].apply(lambda p: self._stitched_image_path / p)

        return stitched_image_data
    
    def _background_image_check(self, background_images: List[Union[Path, str]]):

        reformatted = []
        for image in background_images:
            image = Path(image) if not isinstance(image, Path) else image
            assert image.exists(), 'Background image not found.'

            mask = self.stitched_image_data['image_path'] == image
            n_images = mask.sum()

            if n_images < 1:
                raise Exception('{} not found in `stitched_images.csv`.'.format(image))
            
            if n_images > 1:
                raise Exception('Duplicate entries for {} found in `stitched_images.csv`.'.format(image))
            
            reformatted.append(image)
            
        return reformatted
    
    def subtract(
            self, 
            background_images: List[Union[Path, str]],
            settings_to_match: List[str] = ['temp', 'hum','setup', 'dname', 'lightsource', 'channel', 'exposure', 'camera_mode', 'binning', 'nosepiece'],
            verbose: bool = True,
            dry_run: bool = False
            ):

        # input sanity check
        background_images = self._background_image_check(background_images)

        bgsub_data = []
        grouped = self.stitched_image_data.groupby(by=settings_to_match, dropna=False)

        for settings, group in grouped:

            # region background image sanity checks

            print('Attempting background subtracting on images with the following settings:')
            print(' | '.join(['{}: {}'.format(setting, value) for setting, value in zip(settings_to_match, settings)]))

            # make a target df
            background_image = None
            group['background_image'] = [background_image] * len(group)

            # check if a unique background image can be found
            background_mask = group['image_path'].isin(background_images)
    
            if sum(background_mask) == 0:
                print('No background image found. Subtracting with zero array.')

            else:
                # grab background image (should just be one row)
                background_image_path = group[background_mask].iloc[0]['image_path']
                background_image = io.imread(background_image_path)
                print('Using {} as background image'.format(background_image_path))

                # update dataframe
                group = group.copy() # to stop pandas from complaining 
                group['background_image'] = [background_image_path] * len(group)
                group = group[~background_mask]

            # endregion

            success_mask = np.ones(len(group), dtype=bool)
            for i in tqdm(range(len(group)), desc='Running background subtraction.'):

                target_image_path = group['image_path'].iloc[i]
                outfile = self._bgsub_images_path / target_image_path.relative_to(self._stitched_image_path)
                outfile.parent.mkdir(parents=True, exist_ok=True)

                if not dry_run:

                    if isinstance(background_image, np.ndarray):
            
                        try: 
                            # subtract and save
                            target_image = io.imread(target_image_path)
                            subtracted_image = backgroud_subtract(background_image, target_image)
                            io.imsave(outfile, subtracted_image, check_contrast=False)#, plugin="tifffile"

                        except Exception as e:
                            # "subtract" a zero array
                            # or in other words, just copy target image
                            success_mask[i] = False
                            shutil.copy(target_image_path, outfile)
                            if verbose:
                                print(f"Error background subtracting {target_image_path}: {e}")
                            
                    else:
                        # "subtract" a zero array
                        # or in other words, just copy target image
                        success_mask[i] = False
                        shutil.copy(target_image_path, outfile)   
                             
            print("{successes} / {total} images were successfully background subtracted".format(successes=(success_mask).sum(), total=len(success_mask)))
            print()
            # update failure rows
            group.loc[~success_mask, 'column_name'] = None
            bgsub_data.append(group)

        if not dry_run:

            # format and save the background subtraction dataframe
            self.bgsub_data = pd.concat(bgsub_data, ignore_index=True)
            self.bgsub_data['image_path'] = self.bgsub_data['image_path'].apply(lambda f: f.relative_to(self._stitched_image_path))
            self.bgsub_data['background_image'] = self.bgsub_data['background_image'].apply(lambda f: None if not f else f.relative_to(self._stitched_image_path))
            self.bgsub_data.to_csv(self._bgsub_images_path / 'bgsub_images.csv', index=False)

