import logging
import io
import matplotlib.pyplot as plt
import os


class StitchingSettings:

    # ff_paths = {}
    # ff_params = None
    # ff_images = None
    # Raster origin. bottomleft = (0, 1), topright = (1, 0)
    # acqui_ori = (True, False)
    # tile_dim = None

    def __init__(self, ff_paths=None, ff_params=None, tile_dim=1024, setup_num=1):
        """
        StitchingSettings for general stitching parameters. Assumes a square image.
        The raster pattern is always assumed to traverse rows, then columns in a serpentine
        pattern starting at the specified origin (acqui_ori).

        Arguments:
            (dict) ff_paths: A dict mapping channel name to flat-field image
                (i.e., {'4egfp': setup_3_eGFP_ffPath})
            (ff_params) ff_params: Channel and exposure time specific flat-field correction
                parameters of the form
                    {channel-name_1:
                        {exposure-time_1: (FFDarkValue-c1e1, FFScaleValue-c1e1),
                         exposure-time_2: (FFDarkValue-c1e2, FFScaleValue-c1e2),
                         ...},
                     channel-name_2:
                         {
                         ...}
                     ...}
            (int) tile_dim: Number of pixels defining the width/height of an image tile.
                Assumes square images.

            (int) setup_num: Index of the setup from which to stitch. Needed to define
                The origin and direction of the stage rastering.

        Returns:
            None
        """
        if ff_paths is None:
            self.ff_paths = self.define_ff_paths({})
        else:
            self.ff_paths = self.define_ff_paths(ff_paths)

        self.ff_images = self.read_ff_images()
        self.ff_params = ff_params
        self.tile_dimensions = tile_dim

        self.channels = {
            "1pbp",
            "2bf",
            "3dapi",
            "4egfp",
            "5cy5",
            "6mcherry",
            "pbp",
            "bf",
            "dapi",
            "egfp",
            "cy5",
            "mcherry",
            "yfp",
            "fura",
            '1',
            '2',
            '3',
            '4',
            '5',
            '6'
        }

        if setup_num == 2:
            self.acqui_ori = (False, True)
        else:
            self.acqui_ori = (True, False)
        self.initialize_logger()

    def read_ff_images(self):
        """
        Loads FF images

        Arguments:
            None

        Returns:
            None
        """

        result = {}
        for channel, path in self.ff_paths.items():
            result[channel] = io.imread(path)
        return result

    def define_ff_paths(self, ff_channel_paths):
        """
        Error checking for FF paths dictionary

        Arguments:
            (dict) ff_channel_paths: dictionary of channels:ffpath

        Returns:
            (dict) Dictionary of channel:ffpath

        """
        allPaths = {}
        for channel, path in ff_channel_paths.items():
            if os.path.isfile(path):
                allPaths[channel] = path
            else:
                note = "Error: The Flat-Field Image for {} at {} does not exist"
                raise ValueError(note.format(channel, path))
        return allPaths

    def show_ff_images(self, vmin=0, vmax=65535):
        """
        Displays loaded FF images

        Arguments:
            (int) vmin: intensity minimum
            (int) vmax: intensity maximum

        Returns;
            None

        """
        for channel, image in self.ff_images.items():
            fig = plt.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("RFU")
            plt.title("{} FF Correction Image".format(channel), weight="bold")
            plt.axis("off")
            plt.show()

    @staticmethod
    def initialize_logger():
        """
        Initializes the logger

        Arguments:
            None

        Return:
            None
        """
        logfmt = "%(asctime)s %(levelname)-8s %(message)s"
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.basicConfig(
            format=logfmt, level=logging.INFO, datefmt="%y-%m-%d %H:%M:%S"
        )
        logging.captureWarnings(True)

    # TODO add an "add_channel" functionality
