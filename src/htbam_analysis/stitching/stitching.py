"""
Parallel stitching utilities for processing multiple raster groups.
"""
import os
import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from shutil import copy
from skimage import io
from multiprocessing import Pool
from typing import Union, List, Tuple
from htbam_analysis.stitching import rastering


# def process_single_raster(args: Tuple) -> Tuple[bool, Path, str]:
#     """
#     Process a single raster group (directory + list of image paths).
    
#     Args:
#         args: Tuple of (dir, raster_list, outdir, raster_params)
    
#     Returns:
#         Tuple of (success: bool, output_file: Path, error_msg: str)
#     """
#     dir, raster, outdir, raster_params = args
#     n_raster = int(raster_params.dims[0] * raster_params.dims[1])
    
#     try:
#         if not os.path.isdir(dir):
#             return False, None, f'{dir} does not exist.'
        
#         if len(raster) != n_raster:
#             return False, None, f'{dir} has {len(raster)} images, expected {n_raster}.'
        
#         # Create the raster and stitch
#         raster_obj = raster.FlatRaster(
#             image_refs=raster,
#             params=raster_params
#         )
#         stitched_image = raster_obj.stitch()
        
#         # Save the stitched image
#         outfile = (outdir / dir.relative_to(outdir.parent)).with_suffix('.tif')
#         outfile.parent.mkdir(parents=True, exist_ok=True)
#         io.imsave(outfile, stitched_image, plugin="tifffile", check_contrast=False)
        
#         return True, outfile, None
#     except Exception as e:
#         return False, None, f'Error processing {dir}: {str(e)}'


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
    stitched_image_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(stitched_image_path, stitched_image, plugin="tifffile", check_contrast=False)
        
    return True, stitched_image_path, None


def stitch_images(
    root: Union[Path, str],
    rotation: float,
    acqui_ori: Tuple[bool, bool] = (True, False),
    verbose: bool = True
) -> None:
        
    if not isinstance(root, Path):
            root = Path(root)

    assert root.exists(), f"Root directory {root} does not exist"

    raw_images = root / 'raw_images'
    assert raw_images.exists(), f"Raw images directory {raw_images} does not exist"

    # load image df
    image_csv = os.path.join(root, 'imaging.csv')
    image_df = pd.read_csv(image_csv)

    # make absolute image paths
    image_df['image_path'] = image_df['image_path'].apply(Path)
    image_df['image_path'] = image_df['image_path'].apply(lambda p: raw_images / p)
    image_df['image_path_parent'] = image_df['image_path'].apply(lambda p: p.parent)

    # sort image df by parent dir and raster position
    image_df.sort_values(by=['image_path_parent', 'raster_col_index', 'raster_row_index'], inplace=True)

    # get a list of rasters
    _grouped = image_df.groupby('image_path_parent')
    image_dirs = list(_grouped.groups.keys())
    rasters = _grouped['image_path'].apply(list).tolist()

    # make a df for stitched images
    stitched_image_df = _grouped.first().reset_index()

    # make raster settings
    raster_settings_list = [rastering.RasterParams(
        overlap=o,
        size=1600,
        acqui_ori = acqui_ori,
        rotation=rotation,
        dims = (w, h),
        auto_ff=True,
        ff_type='BaSiC'
    )
    for o, w, h in zip(stitched_image_df['raster_overlap'], stitched_image_df['raster_width'], stitched_image_df['raster_height'])
    ]

    # drop uneeded columns
    stitched_image_df.reset_index(drop=True)
    stitched_image_df.drop(columns=['raster_index', 'frame_time', 'raster_width', 'raster_height', 'raster_start_time', 'x', 'y', 'z', 'raster_overlap', 'raster_row_index', 'raster_col_index'], inplace=True)

    # make an output directory for stitched images
    stitched_images = root / 'stitched_images'
    stitched_images.mkdir(exist_ok=True)

    # stitch images
    success_mask = np.ones(len(image_dirs), dtype=bool)
    for i in tqdm.tqdm(range(len(image_dirs)), desc='Stitching image rasters'):
        outpath = stitched_images / image_dirs[i].relative_to(raw_images).with_suffix('.tif')
        try:
            stitch_single_raster(rasters[i], raster_settings_list[i], outpath)
        except Exception as e:
            success_mask[i] = False
            if verbose:
                print(f"Error stitching {image_dirs[i]}: {e}")


    # print report
    print("{successes} / {total} rasters were successfully stitched".format(successes=(success_mask).sum(), total=len(success_mask)))

    # format and save stitched image dataframe
    success_df = stitched_image_df.loc[success_mask].copy()
    success_df['image_path'] = success_df['image_path_parent'].apply(lambda f: f.relative_to(raw_images).with_suffix('.tif'))
    success_df.drop(columns=['image_path_parent'], inplace=True)
    success_df.to_csv(stitched_images / 'stitched_images.csv', index=False)

    # carry over metadata
    for src in raw_images.rglob('*index.csv'):
        dst = stitched_images / src.relative_to(raw_images)
        copy(src, dst)


def backgroud_subtract(background_image, target_image):

    subtracted = np.subtract(target_image.astype("float"), background_image.astype("float"))

    # TODO figure out why these params are set this way
    subtracted_clipped = np.clip(subtracted, 0, 65535).astype("uint16")
    
    return subtracted_clipped


def background_subtract_images(
    root: Union[Path, str],
    background_images: List[Union[Path, str]],
    settings_to_match: List[str] = ['temp', 'hum','setup', 'dname', 'lightsource', 'channel', 'exposure', 'camera_mode', 'binning', 'nosepiece'],
    verbose: bool = True
):
    
    #region IO and sanity checks

    # check that root is a Path object and exists
    if not isinstance(root, Path):
        root = Path(root)
    assert root.exists(), f"Root directory {root} does not exist"

    # check that background images are Path objects and exist
    for i in range(len(background_images)):
        if not isinstance(background_images[i], Path):
            background_images[i] = Path(background_images[i])
        assert background_images[i].exists(), f"Background image {background_images[i]} does not exist"

    # check that stitched images directory and csv exist
    stitched_images = root / 'stitched_images'
    assert stitched_images.exists(), f"Stitched images directory {stitched_images} does not exist"

    stitched_images_csv = stitched_images / 'stitched_images.csv'
    assert stitched_images_csv.exists(), f"Stitched images csv {stitched_images_csv} does not exist"

    # load stitched images dataframe
    stitched_image_df = pd.read_csv(stitched_images_csv)
    stitched_image_df['image_path'] = stitched_image_df['image_path'].apply(Path)
    stitched_image_df['image_path'] = stitched_image_df['image_path'].apply(lambda f: stitched_images / f)

    #endregion

    # group rows by imaging settings
    grouped = stitched_image_df.groupby(by=settings_to_match, dropna=False)

    bgsub_images = root / 'bgsub_images'
    bgsub_df = []
    for key, group in grouped:

        # region background image sanity checks

        print('Attempting background subtracting on images with the following settings:')
        print(' | '.join(['{}: {}'.format(setting, value) for setting, value in zip(settings_to_match, key)]))

        # check if a unique background image can be found
        background_mask = group['image_path'].isin(background_images)

        if sum(background_mask) == 0:
            print('No background image found. Continuing.\n')
            continue

        elif sum(background_mask) > 1:
            print('Too many background images found. Continuing.\n')    
            continue

        # endregion

        # df for background image (should just be one row)
        background_df = group.loc[background_mask].copy()
        background_image_path = group[background_mask].iloc[0]['image_path']
        background_image = io.imread(background_image_path)

        # df for target images
        target_df = group.loc[~background_mask].copy()
        target_df['background_image'] = [background_image_path] * len(target_df)
        target_images = target_df['image_path'].tolist()

        success_mask = np.ones(len(target_df), dtype=bool)
        for i in tqdm.tqdm(range(len(target_images)), desc='Running background subtraction.'):
            target_image_path = target_images[i]
            try: 
                target_image = io.imread(target_image_path)
                subtracted_image = backgroud_subtract(background_image, target_image)
                
                # save the stitched image
                outfile = bgsub_images / target_image_path.relative_to(stitched_images)
                outfile.parent.mkdir(parents=True, exist_ok=True)
                io.imsave(outfile, subtracted_image, plugin="tifffile", check_contrast=False)

            except Exception as e:
                success_mask[i] = False

                if verbose:
                    print(f"Error background subtracting {target_image_path}: {e}")
        
        print("{successes} / {total} images were successfully background subtracted".format(successes=(success_mask).sum(), total=len(success_mask)))
        print()

        bgsub_df.append(target_df[success_mask])

    # format and save the background subtraction dataframe
    bgsub_df = pd.concat(bgsub_df, ignore_index=True)
    bgsub_df['image_path'] = bgsub_df['image_path'].apply(lambda f: f.relative_to(stitched_images))
    bgsub_df['background_image'] = bgsub_df['background_image'].apply(lambda f: f.relative_to(stitched_images))
    bgsub_df.to_csv(bgsub_images / 'bgsub_images.csv', index=False)

    # carry over metadata
    for src in stitched_images.rglob('*index.csv'):
        dst = bgsub_images / src.relative_to(stitched_images)
        copy(src, dst)


### JSZ WIP CODE ###

# """
# Pythonic image stitching function for raster image processing.

# Provides a functional interface for stitching image rasters using various methods.
# """

# import numpy as np
# from typing import Union, List, Dict, Any, Optional, Tuple
# from pathlib import Path
# from skimage import io, transform
# from PIL import Image, ImageSequence
# from scipy.ndimage import gaussian_filter
# from basicpy import BaSiC
# import logging
# from multiprocessing import Pool, cpu_count

# from htbam_analysis.stitching.rastering.raster_params import RasterParams


# def _load_images(
#     image_paths: List[Union[str, Path]],
#     rotation: Optional[float] = None,
#     stack_index: Optional[int] = None,
# ) -> List[np.ndarray]:
#     """
#     Load images from disk with optional rotation and frame selection.

#     Arguments:
#         image_paths: List of paths to image files
#         rotation: Optional rotation angle in degrees to apply to all images
#         stack_index: Optional frame index for multi-frame images (TIFF stacks)

#     Returns:
#         List of loaded image arrays
#     """
#     images = []

#     for path in image_paths:
#         if stack_index is not None:
#             # Load specific frame from multi-frame image
#             with Image.open(path) as img:
#                 frame_data = np.asarray(ImageSequence.Iterator(img)[stack_index])
#             images.append(frame_data)
#         else:
#             # Load single image
#             images.append(io.imread(path))

#     # Apply rotation if specified
#     if rotation is not None and rotation != 0:
#         rotation_params = {"resize": False, "clip": True, "preserve_range": True}
#         images = [
#             transform.rotate(img, rotation, **rotation_params).astype("uint16")
#             for img in images
#         ]

#     return images


# def _ff_subtract(
#     image: np.ndarray,
#     ff_image: np.ndarray,
#     ff_bval: float,
#     ff_scale: float,
# ) -> np.ndarray:
#     """
#     Apply flat-field correction to an image.

#     Arguments:
#         image: Input image array
#         ff_image: Flat-field reference image
#         ff_bval: Flat-field baseline value
#         ff_scale: Flat-field scaling factor

#     Returns:
#         Flat-field corrected image
#     """
#     ff_result = np.subtract(image, ff_bval) / np.subtract(ff_image, ff_bval) * ff_scale
#     return np.clip(ff_result, 0, 65535).astype("uint16")


# def apply_flat_field_basic(
#     images: List[np.ndarray],
#     sigma: float = 50,
#     visualize: bool = False,
# ) -> List[np.ndarray]:
#     """
#     Apply BaSiC (Basic Illumination Correction) flat-field correction to images.

#     Uses Gaussian smoothing to eliminate small incorrect features learned by BaSiC.

#     Arguments:
#         images: List of image arrays to correct
#         sigma: Gaussian smoothing sigma parameter
#         visualize: Whether to display correction visualization (requires matplotlib)

#     Returns:
#         List of flat-field corrected images
#     """
#     logging.info("Running BaSiC for flat-field correction...")

#     images_array = np.asarray(images)

#     # Initialize and fit BaSiC
#     basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
#     basic.fit(images_array)

#     # Smooth flat-field image
#     ff_image = gaussian_filter(basic.flatfield, sigma=sigma)

#     # Apply flat-field correction to all images
#     ff_bval = 0
#     ff_scale = 1
#     corrected = [_ff_subtract(img, ff_image, ff_bval, ff_scale) for img in images]

#     if visualize:
#         try:
#             from matplotlib import pyplot as plt

#             fig, axes = plt.subplots(1, 3, figsize=(9, 3))
#             im = axes[0].imshow(basic.flatfield)
#             fig.colorbar(im, ax=axes[0])
#             axes[0].set_title("Flatfield")
#             im = axes[1].imshow(ff_image)
#             fig.colorbar(im, ax=axes[1])
#             axes[1].set_title("Smoothed FF")
#             axes[2].plot(basic.baseline)
#             axes[2].set_xlabel("Frame")
#             axes[2].set_ylabel("Baseline")
#             fig.tight_layout()
#             plt.show()
#         except ImportError:
#             logging.warning("matplotlib not available; skipping visualization")

#     logging.info("BaSiC flat-field correction complete")
#     return corrected


# def cut_stitch(
#     images: List[np.ndarray],
#     tile_grid_shape: tuple,
#     overlap: float,
#     origin: tuple = (False, False),
# ) -> np.ndarray:
#     """
#     Stitch image tiles using the 'cut' method.

#     Trims borders according to specified overlap and concatenates tiles along both axes.

#     Arguments:
#         images: List of tile image arrays (ordered in raster order)
#         tile_grid_shape: Tuple of (rows, cols) describing the tile grid arrangement
#         overlap: Overlap fraction between tiles (0 <= overlap < 1)
#         origin: Tuple of (right_origin, bottom_origin) flags for flip correction

#     Returns:
#         Stitched image array

#     Raises:
#         ValueError: If overlap is invalid
#     """
#     if overlap < 0 or overlap >= 1:
#         raise ValueError("Overlap must be ≥ 0 and < 1")

#     # Get tile size from first image
#     tile_size = images[0].shape[0]

#     # Calculate trim dimensions
#     margin = int(tile_size * overlap / 2)
#     retained = tile_size - 2 * margin

#     # Define slice for trimming
#     if margin == 0:
#         border = slice(0, None)
#     else:
#         border = slice(margin, -margin)

#     # Trim all tiles
#     trimmed_tiles = [tile[border, border] for tile in images]
#     tile_array = np.asarray(trimmed_tiles)

#     # Reshape into grid arrangement
#     rows, cols = tile_grid_shape
#     arranged_tiles = np.reshape(tile_array, (rows, cols, retained, retained))

#     # Apply origin corrections (flips)
#     if origin[0]:  # right origin
#         arranged_tiles = np.flip(arranged_tiles, axis=0)
#     if origin[1]:  # bottom origin
#         arranged_tiles = np.flip(arranged_tiles, axis=1)

#     # Concatenate along both axes
#     rows_stitched = np.concatenate(arranged_tiles, axis=2)
#     full_stitched = np.concatenate(rows_stitched, axis=0)

#     return full_stitched


# def stitch(
#     image_paths: List[Union[str, Path]],
#     params: Union[RasterParams, Dict[str, Any]],
#     method: str = "cut",
#     stack_index: Optional[int] = None,
#     apply_flat_field: bool = False,
#     ff_kwargs: Optional[Dict[str, Any]] = None,
# ) -> np.ndarray:
#     """
#     Stitch a raster of images into a single stitched image.

#     Pythonic functional interface for image stitching with support for various correction
#     and stitching methods.

#     Arguments:
#         image_paths: List of paths to image tiles in raster order (left-to-right, top-to-bottom)
#         params: RasterParams object or dict with keys:
#             - 'overlap': float, overlap fraction (e.g., 0.1)
#             - 'size': int, tile size in pixels (square tiles assumed)
#             - 'tile_dims': tuple, (rows, cols) of tile grid
#             - 'rotation': float, rotation angle in degrees (optional, default 0)
#             - 'origin': tuple, (right_origin, bottom_origin) bool flags (optional, default (False, False))
#         method: Stitching method: 'cut' or 'overlap' (default: 'cut')
#         stack_index: Optional frame index for multi-frame images (e.g., TIFF stacks)
#         apply_flat_field: Whether to apply BaSiC flat-field correction
#         ff_kwargs: Keyword arguments to pass to flat-field correction function

#     Returns:
#         Stitched image array (np.ndarray)

#     Raises:
#         ValueError: If method is not supported or params are invalid
#         FileNotFoundError: If image files cannot be found

#     Example:
#         >>> from pathlib import Path
#         >>> from htbam_analysis.stitching.rastering.raster_params import RasterParams
#         >>>
#         >>> # Define parameters
#         >>> params = RasterParams(
#         ...     overlap=0.1,
#         ...     size=512,
#         ...     acqui_ori=(False, False),
#         ...     rotation=0,
#         ...     auto_ff=True
#         ... )
#         >>> params.update_dims((8, 8))  # 8x8 tile grid
#         >>>
#         >>> # Load image paths
#         >>> image_dir = Path('path/to/tiles')
#         >>> image_paths = sorted(image_dir.glob('*.tif'))
#         >>>
#         >>> # Stitch
#         >>> stitched = stitch(
#         ...     image_paths,
#         ...     params,
#         ...     method='cut',
#         ...     apply_flat_field=True
#         ... )
#     """
#     # Parse parameters
#     if isinstance(params, dict):
#         overlap = params.get("overlap")
#         tile_size = params.get("size")
#         tile_dims = params.get("tile_dims")
#         rotation = params.get("rotation", 0)
#         origin = params.get("origin", (False, False))
#     elif isinstance(params, RasterParams):
#         overlap = params.overlap
#         tile_size = params.size
#         tile_dims = params.dims
#         rotation = params.rotation
#         origin = params.acqui_ori
#     else:
#         raise TypeError("params must be RasterParams instance or dict")

#     # Validate parameters
#     if tile_dims is None:
#         raise ValueError("tile_dims (or params.dims) must be specified")
#     if overlap is None:
#         raise ValueError("overlap must be specified")
#     if tile_size is None:
#         raise ValueError("size (tile size) must be specified")

#     # Validate method
#     if method not in ("cut", "overlap"):
#         raise ValueError(f"Invalid stitch method '{method}'. Must be 'cut' or 'overlap'")

#     if method == "overlap":
#         raise NotImplementedError("Overlap stitching method not yet implemented")

#     # Convert paths to Path objects if needed
#     image_paths = [Path(p) for p in image_paths]

#     # Load images
#     logging.info(f"Loading {len(image_paths)} images...")
#     images = _load_images(image_paths, rotation=rotation, stack_index=stack_index)

#     # Apply flat-field correction if requested
#     if apply_flat_field:
#         ff_kwargs = ff_kwargs or {}
#         images = apply_flat_field_basic(images, **ff_kwargs)

#     # Perform stitching
#     logging.info(f"Stitching images using '{method}' method...")
#     if method == "cut":
#         stitched = cut_stitch(images, tile_dims, overlap, origin)

#     logging.info("Stitching complete")
#     return stitched


# def _stitch_worker(
#     args: Tuple[
#         List[Union[str, Path]],
#         Union[RasterParams, Dict[str, Any]],
#         str,
#         Optional[int],
#         bool,
#         Optional[Dict[str, Any]],
#     ]
# ) -> np.ndarray:
#     """
#     Worker function for parallel stitching.

#     Unpacks arguments and calls stitch(). Designed to be used with multiprocessing.Pool.

#     Arguments:
#         args: Tuple of (image_paths, params, method, stack_index, apply_flat_field, ff_kwargs)

#     Returns:
#         Stitched image array
#     """
#     image_paths, params, method, stack_index, apply_flat_field, ff_kwargs = args
#     return stitch(
#         image_paths,
#         params,
#         method=method,
#         stack_index=stack_index,
#         apply_flat_field=apply_flat_field,
#         ff_kwargs=ff_kwargs,
#     )


# def stitch_parallel(
#     stitch_tasks: List[Tuple[List[Union[str, Path]], Union[RasterParams, Dict[str, Any]]]],
#     method: str = "cut",
#     stack_index: Optional[int] = None,
#     apply_flat_field: bool = False,
#     ff_kwargs: Optional[Dict[str, Any]] = None,
#     num_workers: Optional[int] = None,
#     chunksize: Optional[int] = None,
# ) -> List[np.ndarray]:
#     """
#     Stitch multiple image rasters in parallel using multiprocessing.

#     Distributes stitching tasks across multiple CPU cores for improved performance when
#     processing many rasters.

#     Arguments:
#         stitch_tasks: List of tuples, each containing:
#             - image_paths: List of paths to image tiles
#             - params: RasterParams object or dict with stitching configuration
#         method: Stitching method: 'cut' or 'overlap' (default: 'cut')
#         stack_index: Optional frame index for multi-frame images
#         apply_flat_field: Whether to apply BaSiC flat-field correction to each raster
#         ff_kwargs: Keyword arguments for flat-field correction function
#         num_workers: Number of worker processes (default: CPU count)
#         chunksize: Number of tasks per worker batch (default: auto)

#     Returns:
#         List of stitched image arrays, in the same order as input tasks

#     Example:
#         >>> from pathlib import Path
#         >>> from htbam_analysis.stitching.rastering.stitch import stitch_parallel
#         >>> from htbam_analysis.stitching.rastering.raster_params import RasterParams
#         >>>
#         >>> # Prepare multiple stitching tasks
#         >>> base_dir = Path('path/to/rasters')
#         >>> tasks = []
#         >>> for raster_dir in base_dir.iterdir():
#         ...     params = RasterParams(overlap=0.1, size=512, acqui_ori=(False, False), rotation=0)
#         ...     params.update_dims((8, 8))
#         ...     image_paths = sorted(raster_dir.glob('*.tif'))
#         ...     tasks.append((image_paths, params))
#         >>>
#         >>> # Stitch all rasters in parallel
#         >>> stitched_images = stitch_parallel(
#         ...     tasks,
#         ...     method='cut',
#         ...     apply_flat_field=True,
#         ...     num_workers=4
#         ... )
#     """
#     if not stitch_tasks:
#         logging.warning("No stitching tasks provided")
#         return []

#     # Determine number of workers
#     if num_workers is None:
#         num_workers = cpu_count()
    
#     logging.info(
#         f"Starting parallel stitching: {len(stitch_tasks)} tasks on {num_workers} workers"
#     )

#     # Prepare arguments for worker function
#     worker_args = [
#         (image_paths, params, method, stack_index, apply_flat_field, ff_kwargs)
#         for image_paths, params in stitch_tasks
#     ]

#     # Perform parallel stitching
#     try:
#         with Pool(processes=num_workers) as pool:
#             stitched_images = pool.map(
#                 _stitch_worker,
#                 worker_args,
#                 chunksize=chunksize,
#             )
#     except Exception as e:
#         logging.error(f"Error during parallel stitching: {e}")
#         raise

#     logging.info(f"Parallel stitching complete: {len(stitched_images)} images processed")
#     return stitched_images


# def stitch_parallel_with_names(
#     stitch_tasks: Dict[str, Tuple[List[Union[str, Path]], Union[RasterParams, Dict[str, Any]]]],
#     method: str = "cut",
#     stack_index: Optional[int] = None,
#     apply_flat_field: bool = False,
#     ff_kwargs: Optional[Dict[str, Any]] = None,
#     num_workers: Optional[int] = None,
#     chunksize: Optional[int] = None,
# ) -> Dict[str, np.ndarray]:
#     """
#     Stitch multiple image rasters in parallel, preserving task names/identifiers.

#     Distributes stitching tasks across multiple CPU cores. Useful when you need to track
#     which output corresponds to which input raster.

#     Arguments:
#         stitch_tasks: Dict mapping task names/identifiers to tuples of:
#             - image_paths: List of paths to image tiles
#             - params: RasterParams object or dict with stitching configuration
#         method: Stitching method: 'cut' or 'overlap' (default: 'cut')
#         stack_index: Optional frame index for multi-frame images
#         apply_flat_field: Whether to apply BaSiC flat-field correction
#         ff_kwargs: Keyword arguments for flat-field correction function
#         num_workers: Number of worker processes (default: CPU count)
#         chunksize: Number of tasks per worker batch (default: auto)

#     Returns:
#         Dict mapping task names to stitched image arrays

#     Example:
#         >>> from pathlib import Path
#         >>> from htbam_analysis.stitching.rastering.stitch import stitch_parallel_with_names
#         >>> from htbam_analysis.stitching.rastering.raster_params import RasterParams
#         >>>
#         >>> # Prepare multiple stitching tasks with identifiers
#         >>> base_dir = Path('path/to/rasters')
#         >>> tasks = {}
#         >>> for raster_dir in sorted(base_dir.iterdir()):
#         ...     params = RasterParams(overlap=0.1, size=512, acqui_ori=(False, False), rotation=0)
#         ...     params.update_dims((8, 8))
#         ...     image_paths = sorted(raster_dir.glob('*.tif'))
#         ...     tasks[raster_dir.name] = (image_paths, params)
#         >>>
#         >>> # Stitch all rasters in parallel
#         >>> stitched_dict = stitch_parallel_with_names(
#         ...     tasks,
#         ...     method='cut',
#         ...     apply_flat_field=True,
#         ...     num_workers=4
#         ... )
#         >>> for name, stitched_img in stitched_dict.items():
#         ...     print(f"{name}: {stitched_img.shape}")
#     """
#     if not stitch_tasks:
#         logging.warning("No stitching tasks provided")
#         return {}

#     # Convert dict to list while preserving names
#     task_names = list(stitch_tasks.keys())
#     task_list = [stitch_tasks[name] for name in task_names]

#     # Perform parallel stitching
#     stitched_images = stitch_parallel(
#         task_list,
#         method=method,
#         stack_index=stack_index,
#         apply_flat_field=apply_flat_field,
#         ff_kwargs=ff_kwargs,
#         num_workers=num_workers,
#         chunksize=chunksize,
#     )

#     # Reconstruct dict with names
#     return {name: img for name, img in zip(task_names, stitched_images)}
