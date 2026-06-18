# -*- coding: utf-8 -*-

"""Top-level package for processing."""

from htbam_analysis.processing import chip, experiment

__author__ = """Daniel Mokhtari"""
__email__ = ""
__version__ = "0.1.0"

import skimage
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union, List, Tuple


class Processor:
    
    def __init__(self, experiment: experiment.Experiment, image_data: pd.DataFrame, features: str):
        self.experiment = experiment

        self.image_data = image_data
        self.image_data['corners'] = [None] * len(self.image_data)
        self.image_data['reference_image'] = [None] * len(self.image_data)

        self.features = features
        assert self.features in ('button', 'chamber', 'all'), "features argument must be 'button', 'chamber', or 'all'."

        self.reference_images = {dname: None for dname in self.experiment.devices}

        self.summary_image_dir = self.experiment.root / 'summary_images'

        # print('Loaded the following images:')
        # for image in image_data['image_path']:
        #     print(image)

    def _update_reference_image(self, dname: str, chip_image: chip.ChipImage):
        self.reference_images[dname] = chip_image

    def _process_features(self, chip_image: chip.ChipImage, coerce_chamber_center: bool = False):
        """Extract feature processing logic into a reusable method."""
        if self.features == 'chamber':
            chip_image.findChambers(coerce_center=coerce_chamber_center)
        elif self.features == 'button':
            chip_image.findButtons()
        elif self.features == 'all':
            chip_image.findChambers(coerce_center=coerce_chamber_center)
            chip_image.findButtons()

    def _save_summary_image_if_needed(self, chip_image: chip.ChipImage, summary_image_path: Union[str, Path]):
        """Save summary image if path is provided."""
        if summary_image_path:
            skimage.io.imsave(summary_image_path, chip_image.summary_image(stamptype=self.features))

    def set_corners(self, dname: str, corners: List[Tuple]):
        assert dname in self.experiment.devices, '{} not found in experiment.'.format(dname)
        assert len(corners) == 4
        for item in corners:
            assert isinstance(item, tuple)
            assert len(item) == 2
            x, y = item
            assert isinstance(x, int)
            assert isinstance(y, int)

        device_mask = self.image_data['dname'] == dname
        for idx in self.image_data[device_mask].index:
            self.image_data.at[idx, 'corners'] = corners

    def set_reference(
        self, 
        image: Union[Path, str], 
        save_summary_images: bool = True,
        coerce_chamber_center: bool = False
    ):
        """Set reference image for a device."""


        image = Path(image) if not isinstance(image, Path) else image

        # Get device number and corners via metadata lookup 
        data = self.image_data[self.image_data['image_path'] == image]
        assert len(data) > 0, 'Image not found!'
        assert len(data) == 1, 'Duplicate images found!'
        dname = data.iloc[0]['dname']
        corners = data.iloc[0]['corners']

        # Set reference_image column
        device_mask = self.image_data['dname'] == dname
        self.image_data.loc[device_mask, 'reference_image'] = image

        # Create and process chip image
        chip_image = chip.ChipImage(self.experiment.devices[dname], image, corners)
        chip_image.stamp()
        self._process_features(chip_image, coerce_chamber_center)
        
        self._update_reference_image(dname, chip_image)

        if save_summary_images:
            outpath = self.summary_image_dir / image.relative_to(self.experiment.root)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            skimage.io.imsave(outpath, chip_image.summary_image(stamptype=self.features))

    def process(
        self, 
        *, 
        use_reference: bool = False, 
        save_summary_images: bool = True,
        coerce_chamber_center: bool = False, 
        **kwargs
    ):
        """High-level dispatcher that handles the mutual exclusivity logic."""
        
        if use_reference:
            return self._process_from_reference(save_summary_images=save_summary_images, **kwargs)
        
        else:

            no_corners_mask = self.image_data['corners'].isna()
            cornerless_devices = set(self.image_data['dname'][no_corners_mask].to_list())
            assert len(cornerless_devices) == 0, 'ERROR: the following devices have no corners set: ' + ', '.join(cornerless_devices)

            return self._process_manually(coerce_chamber_center=coerce_chamber_center, save_summary_images=save_summary_images, **kwargs)
        
    def _process_manually(
        self, 
        save_summary_images: bool = False,
        coerce_chamber_center: bool = False
    ):
        """Process images with manually provided corners."""
        data = []
        for i in tqdm(range(len(self.image_data)), desc='Processing images', leave=False):

            dname, image, c = self.image_data[['dname', 'image_path', 'corners']].iloc[i]
            chip_image = chip.ChipImage(self.experiment.devices[dname], image, c)
            chip_image.stamp()
            self._process_features(chip_image, coerce_chamber_center)

            processed_data = chip_image.summarize().reset_index()
            metadata = pd.DataFrame([self.image_data.iloc[i]] * len(processed_data)).reset_index(drop=True)
            merged = pd.concat([metadata, processed_data], axis=1)
            data.append(merged)
            
            if save_summary_images:
                outpath = self.summary_image_dir / image.relative_to(self.experiment.root)
                outpath.parent.mkdir(parents=True, exist_ok=True)
                skimage.io.imsave(outpath, chip_image.summary_image(stamptype=self.features))
        
        return pd.concat(data, ignore_index=False)

    def _process_from_reference(
        self, 
        save_summary_images: bool = True
    ):
        """Process images by mapping from reference images."""

        data = []
        for i in tqdm(range(len(self.image_data)), desc='Processing images', leave=False):

            dname, image = self.image_data[['dname', 'image_path']].iloc[i]

            reference = self.reference_images.get(dname)
            if reference is None:
                raise ValueError(f"No reference image set for device {dname}. Call set_reference() first.")
            
            chip_image = chip.ChipImage(self.experiment.devices[dname], image, reference.corners)
            chip_image.stamp()
            reference.mapto(chip_image, features=self.features)
            
            processed_data = chip_image.summarize().reset_index()
            metadata = pd.DataFrame([self.image_data.iloc[i]] * len(processed_data)).reset_index(drop=True)
            merged = pd.concat([metadata, processed_data], axis=1)
            data.append(merged)
            
            if save_summary_images:
                outpath = self.summary_image_dir / image.relative_to(self.experiment.root)
                outpath.parent.mkdir(parents=True, exist_ok=True)
                skimage.io.imsave(outpath, chip_image.summary_image(stamptype=self.features))

        return pd.concat(data, ignore_index=False)

    def _save_summary_image(
        self, 
        summary_image_dir: Union[Path, str], 
        image_path: Union[Path, str], 
        chip_image: chip.ChipImage, 
        features: str
    ):
        """Save summary images for debugging/review."""
        summary_image_dir = Path(summary_image_dir) if not isinstance(summary_image_dir, Path) else summary_image_dir
        image_path = Path(image_path) if not isinstance(image_path, Path) else image_path
        summary_image_dir.mkdir(parents=True, exist_ok=True)  # Fixed: removed redundant argument

        base_name = image_path.with_suffix('').name
        
        if features in ('button', 'all'):
            summary_image = chip_image.summary_image(stamptype='button')
            path = summary_image_dir / f'{base_name}_ButtonSummaryImage.tif'
            skimage.io.imsave(path, summary_image)

        if features in ('chamber', 'all'):
            summary_image = chip_image.summary_image(stamptype='chamber')
            path = summary_image_dir / f'{base_name}_ChamberSummaryImage.tif'
            skimage.io.imsave(path, summary_image)

