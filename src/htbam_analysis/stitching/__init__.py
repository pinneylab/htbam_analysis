# -*- coding: utf-8 -*-

"""Top-level package for stitching."""

from htbam_analysis.stitching import rastering

__author__ = """Daniel Mokhtari"""
__email__ = ""
__version__ = "0.1.0"


import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
from pathlib import Path
from typing import Tuple, Union, List


def stitch_single_raster(
    raster: List[Path],
    raster_params: rastering.RasterParams,
    stitched_image_path: Path,
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
    stitched_image = raster_obj.stitch()
    
    # Save the stitched image
    io.imsave(stitched_image_path, stitched_image, plugin="tifffile", check_contrast=False)
        
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
        raster_data['image_path'] = raster_data['image_path'].apply(lambda p: self._raw_images_path / p)
        raster_data['image_path_parent'] = raster_data['image_path'].apply(lambda p: p.parent)
        raster_data.sort_values(by=['image_path_parent', 'raster_col_index', 'raster_row_index'], inplace=True)

        return raster_data

    def stitch_images(self, rotation: float, acqui_origin: Tuple[bool] = (True, False)):

        assert self.raster_data, 'Raster data has not been set.'

        # TODO: make this not hard-coded
        SIZE = 1600
        raster_headers = ['image_path', 'image_path_parent', 'raster_index', 'x', 'y', 'z', 'frame_time', 'raster_width', 'raster_height', 'raster_overlap', 'raster_row_index', 'raster_col_index']
        stitched_image_data = []

        grouped = self.raster_data.groupby('image_path_parent')

        for image_parent, df in grouped:

            # format outpath and make sure directory to write it to exists
            outpath = self._stitched_images_path / image_parent.relative_to(self._raw_images_path).with_suffix('.tif')
            outpath.parent.mkdir(exist_ok=True)

            # extract params, stitch image
            overlap, width, height = df[['raster_overlap', 'raster_width', 'raster_height']].iloc[0]
            raster = df['image_path'].to_list()
            params = rastering.RasterSettings(overlap, size=SIZE, acqui_ori=acqui_origin, rotation=rotation, dims=(width, height), auto_ff=True, ff_type='BaSiC') 
            stitch_single_raster(raster, params, outpath)

            # carry over metadata
            row = df.drop_duplicates(subset=['image_path_parent']).drop(columns=raster_headers)
            row['image_path'] = outpath.relative_to(self._stitched_images_path)
            stitched_image_data.append(row)

        self.stitched_image_data = pd.concat(stitched_image_data).reset_index(drop=True)
        self.stitched_image_data.to_csv(self._stitched_images_path / 'stitched_images.csv')


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
            verbose: bool = True
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

                if isinstance(background_image, np.ndarray):
                    try: 
                        # subtract and save
                        target_image = io.imread(target_image_path)
                        subtracted_image = backgroud_subtract(background_image, target_image)
                        io.imsave(outfile, subtracted_image, plugin="tifffile", check_contrast=False)

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

        # format and save the background subtraction dataframe
        self.bgsub_data = pd.concat(bgsub_data, ignore_index=True)
        self.bgsub_data['image_path'] = self.bgsub_data['image_path'].apply(lambda f: f.relative_to(self._stitched_image_path))
        self.bgsub_data['background_image'] = self.bgsub_data['background_image'].apply(lambda f: None if not f else f.relative_to(self._stitched_image_path))
        self.bgsub_data.to_csv(self._bgsub_images_path / 'bgsub_images.csv', index=False)

