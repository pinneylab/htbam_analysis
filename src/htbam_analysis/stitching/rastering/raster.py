import numpy as np
import pathlib
import os
import logging
from skimage import io, transform
from abc import abstractmethod, ABC
from htbam_analysis.stitching.rastering.raster_params import RasterParams
from PIL import Image, ImageSequence
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from basicpy import BaSiC
from basicpy import datasets as bdata


def ff_subtract(i, ffi, ff_bval, ff_scale):
    ff_result = np.subtract(i, ff_bval) / np.subtract(ffi, ff_bval) * ff_scale
    # TODO figure out where these params are coming from
    result = np.clip(ff_result, 0, 65535).astype("uint16")
    return result


def rotate_image(img, rotation_val) -> np.array:
    rotation_params = {"resize": False, "clip": True, "preserve_range": True}
    return transform.rotate(img, rotation_val, **rotation_params).astype("uint16")


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

    def applyFF_BaSiC(self):
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
        print("Running BaSiC for FF correction...")

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

    def stitch(self, method="cut"):
        """
        Wrapper for image stitching method selection.
        TODO: Implement 'overlap' method

        Arguments:
            (str) method: stitch method ('cut' | 'overlap')

        Returns:
            (np.ndarray) A stitched image array

        """

        self.fetch_images()
        if method == "cut":
            return self.cut_stitch()
        elif method == "overlap":
            return self.overlap_stitch()
        else:
            raise ValueError(
                'Invalid stitch method. Valid methods are "cut" and "overlap"'
            )

    def cut_stitch(self):
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
            tiles = self.ffCorrectedImages = self.applyFF_BaSiC()
            print("completed BaSiC FF correction")

            logging.info(
                "BaSiC Flat-Field Corrected Image | Ch: {}, Exp: {}".format(
                    self.params.channel, self.params.exposure
                )
            )

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
        if self.params.acqui_ori[
            0
        ]:  # If origin on right, flip horizontally (view returned)
            arrangedTiles = np.flip(arrangedTiles, 0)
        if self.params.acqui_ori[
            1
        ]:  # If origin on bottom, flip vertically (view returned)
            arrangedTiles = np.flip(arrangedTiles, 1)
        rowsStitched = np.concatenate(arrangedTiles, axis=2)  # Stitch rows
        fullStitched = np.concatenate(rowsStitched, axis=0)  # Stitch cols
        return fullStitched

    def overlap_stitch(self):
        """
        #TODO: re-implement overlap stitching method

        """
        raise NotImplementedError("Overlap Stitch not yet implemented")

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
        io.imsave(outDir, stitchedRaster, plugin="tifffile", check_contrast=False)
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


class StackedRaster(Raster):
    def __init__(self, image_refs, stackIndex, params):
        """

        Arguments:

        Returns:
            None

        """
        super().__init__(image_refs, params)
        self.stackIndex = stackIndex

    def fetch_images(self):
        """
        Fetches (loads into memory) and rotates images (if indicated by raster parameters)

        Arguments:
            None

        Returns:
            None

        """
        r = self._params.rotation

        rotation_params = {"resize": False, "clip": True, "preserve_range": True}
        rotate_image = lambda i: transform.rotate(i, r, **rotation_params).astype(
            "uint16"
        )

        def readImage(path):
            with Image.open(path) as img:
                data = np.asarray(ImageSequence.Iterator(img)[self.stackIndex])
            return data

        images = [readImage(i) for i in self.image_refs]

        if r:
            images = [rotate_image(i) for i in images]

        self._images = images


class RasterSet:
    def __init__(self, rasters):
        """
        A set of rasters

        """
        self.rasters = rasters

    def exportStitchAll(self, **args):
        """
        Basic stitching and export of raster set
        """
        while self.rasters:
            r = self.rasters.pop()
            r.export_stitch(**args)


class RasterGroup:
    def __init__(self, root, rasters):
        """ """
        # Dict of feature (time)
        self.root = root
        self.rasters = rasters  # list of rasters

    def add(self):
        # add raster to group
        return

    def stitch(self):
        while self.rasters:
            self.rasters.pop().stitch()


class KineticImaging(RasterGroup):
    def __init__(self, root, rasters):
        super().__init__(root, rasters)
        self.referenced_times = None
        self.ordered_rasters = None

    def order(self):
        """
        Orders the set of rasters as a dictionary of time:raster entries

        Arguments:
            None

        Returns:
            None

        """
        # TODO: why reference the raster params here, instead of sel
        sorted_times = sorted([raster.params.group_feature for raster in self.rasters])
        self.referenced_times = [t - min(sorted_times) for t in sorted_times]
        self.ordered_rasters = {
            raster.params.group_feature - min(sorted_times): raster
            for raster in self.rasters
        }

    def export_stitch(self, method="cut", outPathName="StitchedImages"):
        """
        Stitches and exports each of the rasters, appending the raster time onto the filename

        Arguments:
            (str) method: stitch method ('cut' | 'overlap')
            (outPathname) out_path_name: Name of folder to house stitched raster. Typically 'StitchedImages'

        Returns:
            None

        """
        p = pathlib.Path(self.root)
        pathstem = p.stem
        pathparent_stem = pathlib.Path(p.parent).stem
        for dt, raster in tqdm(
            self.ordered_rasters.items(),
            desc="Stitching Kinetics | {}".format(pathparent_stem),
        ):
            time = dt.total_seconds()
            stitchedRaster = raster.stitch(method=method)

            features = [raster.params.exposure, raster.params.channel, int(time)]
            rasterName = "StitchedImg_{}_{}_{}.tif".format(*features)

            stitchDir = pathlib.Path(os.path.join(self.root, outPathName))
            stitchDir.mkdir(exist_ok=True)
            outDir = os.path.join(stitchDir, rasterName)
            io.imsave(outDir, stitchedRaster, plugin="tifffile", check_contrast=False)

            mp = {"t": time, "ch": raster.params.channel, "ex": raster.params.exposure}
            logging.debug("Stitch Saved: (Time: {t}, Ch: {ch}, Ex: {ex})".format(**mp))
        logging.debug("Kinetic Stitches Complete")
