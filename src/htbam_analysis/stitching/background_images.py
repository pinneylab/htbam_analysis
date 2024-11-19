import logging
import pathlib
import numpy as np
import os
from skimage import io


class BackgroundImages:
    def __init__(self):
        """
        A simple class for background subtraction.
        Stores a collection of background images and permits scripted background subtraction

        Arguments:
            None

        Returns:
            None

        """

        self.images = {}

    def add(self, path, index, channel, exposure):
        """
        Adds a background image, described by an index described by (index, channel, exposure)
        where index is typically device name (i.e., 'd1')

        Arguments:
            (str) path: Background image path
            (str) index: arbitrary index (typically device name (i.e., 'd1'))
            (str) channel: Background image channel
            (int) exposure: Background image exposure time (ms)

        Returns:
            None

        """
        # TODO Should think about if this makes sense to overwrite
        k = (index, channel, exposure)
        if k in self.images.keys():
            logging.warning(
                "Background image with features {} already exists. Overwriting...".format(
                    k
                )
            )
        self.images[(index, channel, exposure)] = io.imread(path)

    def remove(self, index, channel, exposure):
        """
        Removes a background image.
        Arguments:
            (str) index: arbitrary index (typically device name (i.e., 'd1'))
            (str) channel: Background image channel
            (int) exposure: Background image exposure time (ms)

        Returns:
            None

        """

        del self.images[(index, channel, exposure)]

    def subtract_background(
        self,
        target_image_path,
        target_index,
        target_channel,
        target_exposure,
        prefix="BGSubtracted_",
    ):
        """
        Subtracts a background image from a target image of matching index, channel, and exposure.
        Resulting images are prefixed with an optional prefix, typically 'BGSubtracted_'

        Arguments:
            (str) target_image_path: path of image to background subtract
            (str) target_index: arbitrary index to match to backgrounds (typ. device name (i.e., 'd1'))
            (str) target_channel: target image channel
            (int) target_exposure:: target image exposure
            (str) prefix: resulting image filename prefix

        Returns:
            None


        """
        mp = {"ch": target_channel, "ex": target_exposure, "i": target_index}
        logging.info(
            "Background Subtracting | Ch: {ch}, Ex: {ex}, Index: {i}".format(**mp)
        )
        img_dir = pathlib.Path(target_image_path)
        target = io.imread(img_dir)

        bg_image = self.images[(target_index, target_channel, target_exposure)]
        bg_sub = np.subtract(target.astype("float"), bg_image.astype("float"))

        # TODO figure out why these params are set this way
        bg_sub_clipped = np.clip(bg_sub, 0, 65535).astype("uint16")

        outPath = os.path.join(img_dir.parents[0], "{}{}".format(prefix, img_dir.name))
        io.imsave(outPath, bg_sub_clipped, plugin="tifffile", check_contrast=False)

        logging.debug("Background Subtraction Complete")

    def walk_and_bg_subtract(self, path, index, channel, manual_exposure=None):
            """
            Walks a directory structure, find images to background subtract, and executes subtraction

            Arguments:
                (str) path: path from hwere to walk
                (str) index: arbitrary index to select background image
                (str) channel: channel to select background image

            Returns:
                None

            """

            parse = lambda f: tuple(f.split(".")[0].split("_")[1:3] + [".".join(f.split(".")[0].split("_")[3:])])

            for root, dirs, files in os.walk(path):
                if ("StitchedImages" in root) | ("Analysis" in root):
                    parsed_files = {
                        parse(f): os.path.join(root, f) for f in files
                    }

                    for params, file in parsed_files.items():

                        # exposure, channel, and features for image
                        e, c, f = params

                        # check that the channel in the file handle matches that passed as an arg
                        if c == channel:
                            if manual_exposure:  # in case filenames corrupted
                                e = manual_exposure
                            self.subtract_background(file, index, channel, int(e))