import htbam.image.stitching.file_handler as fh
import logging
import pathlib
import os
from copy import deepcopy
from tqdm import tqdm


def stitch_image(path, params):
    """
    Stitch and export a single raster at the path with given parameters (exposure is overwritten)

    """

    start_message = (
        "Stitching images | Ch: {ch}, Exp: {ex}, Overlap: {o}, Rot: {r}".format(
            ch=params.channel, ex=params.exposure, o=params.overlap, r=params.rotation
        )
    )
    logging.info(start_message)
    raster = fh.parse_single_channel_folder(params, path)
    raster.exportStitchAll()


def stitch_kinetics(path, params):
    """
    Stitch and export a timecourse of rasters at the path with given parameters

    """

    startmessage = "Starting Kinetic Stitch"
    logging.debug(startmessage)
    k = fh.parse_kinetic_folder(params, path)  # Returns KineticImaging
    k.order()
    k.export_stitch()


def stitch_standard(path, params, handlesIDs):
    """
    Stitch and export a standard curve of rasters at the path with given parameters

    """
    startmessage = "Starting Standard Curve Stitch"
    logging.debug(startmessage)

    glob_pattern = "*_{}*/*/*/"

    r = pathlib.Path(path)

    standards = []
    for handle, ID in handlesIDs:
        img_folders = [(ID, i) for i in list(r.glob(glob_pattern.format(handle)))]
        standards.extend(img_folders)

    for ident, p in tqdm(dict(standards).items(), desc="Stitching Standard"):
        par = deepcopy(params)

        par.update_group_feature(ident)
        par.update_channel(p.parent.name)

        raster = fh.parse_single_channel_folder(par, p)

        target = pathlib.Path(os.path.join(r, "Analysis"))
        target.mkdir(exist_ok=True)
        rexport_params = {"manual_target": target}
        raster.exportStitchAll(**rexport_params)


def walk_and_stitch(path, stitch_settings, params, stitch_type="kinetic"):
    """
    Walk a directory structure of identical types (all single acquisitions or all kinetic)
    and stitch all images beneath. The folder to stitch must be wrapped in a folder
    with a name among the channels found in the class variable StitchingSettings.channels.
    Ignores folders with the keyword "StitchedImages" in them.

    Arguments:
        (str) path: filepath to top-level folder
        (StitchingSettings) stitch_settings: stitching settings
        (RasterParams) params: raster parameters.
        (str) stitch_type: type of rasters contained ('single' | 'kinetic')

    Returns:
        None

    """
    found_stitchable = False
    for root, dirs, files in os.walk(path):
        # check if current base-level directory is in valid channel list
        if os.path.basename(root) in stitch_settings.channels:

            found_stitchable = True

            curr_channel = os.path.basename(root)

            # files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == "."]

            tidy_dirs = [
                direct for direct in sorted(dirs) if "StitchedImages" not in direct
            ]

            # channel = os.path.basename(root)
            if stitch_type == "kinetic":
                new_params = deepcopy(params)
                new_params.update_channel(curr_channel)
                stitch_kinetics(root, new_params)

            elif stitch_type == "single":
                for direct in tidy_dirs:
                    new_params = deepcopy(params)
                    new_params.update_channel(curr_channel)
                    target = os.path.join(root, direct)
                    new_params.update_root(target)
                    # print("Stiching image now! : {}".format(target))
                    stitch_image(target, new_params)
            else:
                raise ValueError('Valid Stitch Types are "single" or "kinetic"')
    if not found_stitchable:
        raise ValueError(
            """
            No stitchable folder found. This is possibly \
            because the 'channel name' hasn't been added to the \
            StitchingSettings object. That can be done like this: \
            stitcher.StitchingSettings.channels.update({'new_channel_name'}) \
            """
        )


def mm_stitch_stacks(path, params, channelExposureMap, channelRemap=None):
    """ """

    raster_metadata = MMFileHandler.parseMMStackedFolder(
        path, channelExposureMap, remapChannels=channelRemap
    )

    channels = raster_metadata.channel.unique().tolist()
    exposures = raster_metadata.exp.unique().tolist()
    delay_times = sorted(raster_metadata.delay_time.unique().tolist())
    params.size = raster_metadata.dims.unique().tolist()[0][0]
    params.dims = (max(raster_metadata.x) + 1, max(raster_metadata.y) + 1)

    rasters = []
    for channel in channels:
        # stitch a set of timepoints
        for time in tqdm(
            delay_times,
            desc="Stitching Kinetics | {} | {}".format(
                channel, pathlib.Path(path).stem
            ),
        ):
            if channelRemap:
                invertedChannelRemap = dict(
                    zip(channelRemap.values(), channelRemap.keys())
                )
                exposure = channelExposureMap[invertedChannelRemap[channel]]
            else:
                exposure = channelExposureMap[channel]
            toStitch = (
                raster_metadata.loc[
                    (raster_metadata.channel == channel)
                    & (raster_metadata.exp == exposure)
                    & (raster_metadata.delay_time == time)
                ]
                .sort_values(["x", "y"])
                .reset_index(drop=True)
                .set_index("exp")
            )
            paths = toStitch.path.values.tolist()
            stackIndex = toStitch.stack_index.unique()[0]

            newparams = deepcopy(params)
            newparams.exposure = exposure
            newparams.channel = channel
            newparams.group_feature = time

            stacked_raster = StackedRaster(paths, stackIndex, newparams)
            stacked_raster.fetch_images()

            stitchedRaster = stacked_raster.stitch()

            features = [
                stacked_raster.params.exposure,
                stacked_raster.params.channel,
                int(time),
            ]
            rasterName = "StitchedImg_{}_{}_{}.tif".format(*features)

            stitchDir = pathlib.Path(os.path.join(path, pathlib.Path("StitchedImages")))
            stitchDir.mkdir(exist_ok=True)
            outDir = os.path.join(stitchDir, rasterName)
            external.tifffile.imsave(outDir, stitchedRaster)
