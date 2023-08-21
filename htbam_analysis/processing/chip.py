# title             : chip.py
# description       :
# authors           : Daniel Mokhtari
# credits           : Craig Markin
# date              : 20180615
# version update    : 20180615
# version           : 0.1.0
# usage             : With permission from DM
# python_version    : 3.6

# General Python
import gc
import warnings
from copy import deepcopy
from collections import namedtuple
from htbam.image.processing import experiment

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import pandas as pd

import cv2
import skimage
from skimage import io


class ChipImage:
    stampWidth = 100
    # stampWidth = 60

    def __init__(
        self, device, raster, ids, corners, pinlist, channel, exposure, attrs=None
    ):
        """
        Constructor for a ChipImage object. A ChipImage represents a single rastered chip image
        and all associated information.

        Arguments:
            (experiment.Device) device: device object
            (str | pathlib.Path) raster: path of the rasted image file
            (dict) ids: dict of ChipImage identifiers and values (i.e., time, concentration, etc.)
            (tuple | namedtuple) corners: cornerpositions of the form
                ((ULx, ULy),(URx, URy),(LLx, LLy),(LRx, LRy))
            (pd.DataFrame) pinlist:
            (str) channel: imaging channel
            (int) exposure: exposure time (ms)
            (dict) attrs: arbitrary metadata

        Returns:
            None

        """

        self.device = device
        self.data_ref = raster  # reference
        self.ids = ids
        self.stampWidth = ChipImage.stampWidth
        self.pinlist = pinlist
        self.attrs = attrs
        self.stamps = None
        self.centers = []

        if not isinstance(corners, type(namedtuple)):
            self.corners = experiment.Device._corners(corners)
        else:
            self.corners = corners

        self.__grid()

    def __grid(self, altCorners=None):
        """
        Calculates chamber center positions interpolating the corner positions

        Arguments:
            (tuple | namedtuple) corners: cornerpositions of the form
                ((ULx, ULy),(URx, URy),(LLx, LLy),(LRx, LRy))

        Returns:
            None

        """

        if altCorners:
            if not isinstance(altCorners, namedtuple):
                corners = experiment.Device._corners(altCorners)
            else:
                corners = altCorners
            self.centers = self.quadrilateralInterp(altCorners, self.device.dims)
        self.centers = self.quadrilateralInterp(self.corners, self.device.dims)

    def stamp(self):
        """
        Wrapper class for chamber stamping. Stamps ChipImage using calculated center positions.

        Arguments:
            None

        Returns:
            None

        """
        self.stamps = self._stamp()

    def _stamp(self):
        """
        Stamps the chipImage using calculated center positions.

        Arguments:
            None

        Returns:
            (np.ndarray) an array of Stamp objects

        """

        pinlistFeatureTitle = "MutantID"
        img = skimage.io.imread(self.data_ref)

        def makeSlice(center, width):
            s = (
                slice(center[1] - (width // 2), center[1] + (width // 2)),
                slice(center[0] - (width // 2), center[0] + (width // 2)),
            )
            return s

        slices = np.apply_along_axis(makeSlice, 2, self.centers, *[(self.stampWidth)])

        def stampImg(slices, img):
            return img[tuple(slices)]

        xdim = self.device.dims.x
        ydim = self.device.dims.y
        imgstamps = np.apply_along_axis(stampImg, 2, slices, *[(img)])
        indices = np.array(
            [[x, y] for x in range(0, xdim) for y in range(0, ydim)]
        ).reshape(xdim, ydim, 2)

        a = np.empty((xdim, ydim), dtype=np.object)
        for x, y in indices.reshape((xdim * ydim, 2)):
            stamp = imgstamps[x, y]
            center = self.centers[x, y]
            s = tuple(slices[x, y])
            a[x, y] = Stamp(
                stamp,
                center,
                s,
                (x + 1, y + 1),
                self.pinlist.loc[x + 1, y + 1][pinlistFeatureTitle],
            )
        return a

    @staticmethod
    def quadrilateralInterp(corners, dims):
        """
        Grid the chip using the corners as vertices and the dims as the number of latice points
        in the x and y directions.

        Arguments:
            (tuple | namedtuple) corners: cornerpositions of the form
                ((ULx, ULy),(URx, URy),(LLx, LLy),(LRx, LRy))
            (tuple) dims: chip dimensions (num columns, num rows)

        Returns:
            (np.ndarray) a 2-D array of latice (x, y) coordinates
        """

        def interp(p1, p2, divs):
            y = np.linspace(p1[0], p2[0], divs, dtype=int)
            x = np.linspace(p1[1], p2[1], divs, dtype=int)
            return np.stack((x, y), axis=1)

        if not isinstance(corners, type(namedtuple)):
            corners = experiment.Device._corners(corners)

        left = interp(corners.ul, corners.bl, dims.y)
        right = interp(corners.ur, corners.br, dims.y)
        mesh = [interp(p1, p2, dims.x) for p1, p2 in zip(left, right)]
        return np.stack(mesh, axis=1)

    def mapto(self, target, features="all"):
        """
        Maps the chamber and/or button parameters to the target ChipImage, and generates the
        Chamber and/or Button objects for those features.

        Arguments:
            (ChipImage) target:
            (str) features; features to map ('chamber', 'button', 'all')

        Returns:
            None

        """

        dims = self.device.dims
        indices = [(i, j) for i in range(dims.x) for j in range(dims.y)]
        for i in indices:
            x, y = i
            s = self.stamps[x, y]
            t = target.stamps[x, y]
            if features == "chamber":
                t.defineChamber(s.chamber.center, s.chamber.radius)
            elif features == "button":
                t.defineButton(
                    s.button.center, s.button.disk_radius, s.button.annulus_radii
                )
            elif features == "all":
                t.defineChamber(s.chamber.center, s.chamber.radius)
                t.defineButton(
                    s.button.center, s.button.disk_radius, s.button.annulus_radii
                )
            else:
                raise ValueError(
                    'Invalid feature name. Choices are "chamber", "button", or "all".'
                )

    def findChambers(self, coerce_center=False):
        """
        Performs chamber finding for each of the Stamps in the ChipImage. Uses a Hough transform.

        Arguments:
            None

        Returns:
            None

        """

        for c in self.stamps.flatten():
            c.findChamber(coerce_center=coerce_center)

    def findButtons(self):
        """
        Performs button finding for each of the Stamps in the ChipImage. Uses a Hough transform.

        Arguments:
            None

        Returns:
            None

        """

        for c in tqdm(self.stamps.flatten(), desc="Finding Buttons"):
            c.findButton()

    def summarize(self):
        """
        Summarizes the ChipImage feature parameters and returns a Pandas DataFrame of the result.

        Arguments:
            None

        Returns:
            (pd.DataFrame) a Pandas DataFrame summarizing the ChipImage feature parameters

        """

        records = {}
        for s in self.stamps.flatten():
            records[s.index] = s.summarize()
        s = pd.DataFrame.from_dict(records, orient="index").sort_index()
        s.index.rename(["x", "y"], inplace=True)
        return s

    def _delete_stamps(self):
        """
        Deletes and forces garbage collection on the image data contained in the ChipImage stamps.
        Data include the stamp data, the chamber and/or button stamp data, and the feature masks.

        Arguments:
            None

        Returns:
            None

        """

        for s in self.stamps.flatten():
            del s.data
            s.data = None
            if s.chamber:
                del s.chamber.stampdata
                del s.chamber.disk
                s.chamber.disk = None
                s.chamber.stampdata = None
            if s.button:
                del s.button.stampdata
                del s.chamber.disk
                del s.chamber.annulus
                s.chamber.disk = None
                s.chamber.annulus = None
                s.button.stampdata = None
        gc.collect()

    def summary_image(self, stamptype):
        """
        Generates a "deflated" chip image as a numpy ndarray. Returns the
        the ChipImage stamps concatenated into a single array (image)

        Arguments:
            (str) stamptype: parameterized feature type to draw onto stamp ('chamber' | 'button')

        Returns:
            (np.ndarray) a 2-D numpy ndarray of the concatenated stamps

        """

        return ChipImage.stitch2D(self._summary_image_arr(stamptype))

    def _summary_image_arr(self, stamptype):
        """
        Generates a 2-d numpy array of stamp images (2-d numpy arrays), indexed by their chip indices.

        Arguments:
            (str) stamptype: parameterized feature type to draw onto stamp ('chamber' | 'button')

        Returns:
            (np.ndarray) a 2-D numpy array of stamp data (ndarrays)
        """

        tiles = self.stamps.flatten()
        stampdims = tiles[0].summary_stamp(stamptype).shape

        arrShape = [
            self.device.dims[0],
            self.device.dims[1],
            stampdims[0],
            stampdims[1],
        ]
        r = np.array([s.summary_stamp(stamptype) for s in tiles]).reshape(*arrShape)
        return r

    @staticmethod
    def stitch2D(arr):
        """
        Concatenates a 2-d array of 2-d numpy arrays (nested) into a single 2-d array (concatenates
        in both directions)

        Arguments:
            (np.ndarray) arr: nested ndarray of the form[[arr1.1, arr1.1,...], [arr2.1, arr2.2,...], ...]

        Returns:
            (np.ndarray) concatenated 2-d array

        """

        rowsStitched = np.concatenate(arr, axis=2)  # Stitch rows
        fullStitched = np.concatenate(rowsStitched, axis=0)  # Stitch cols
        return fullStitched

    def repo_dump(self, stamptype, target_root, title, as_ubyte=False):
        """
        Saves the stamps of the ChipImage to the target directory as a stamp repository (repo).

        Arguments:
            (str) stamptype:parameterized feature type to draw onto stamp ('chamber' | 'button')
            (str) target_root: repo directory, or directory in which to instantiate repo
            (str) title: stem of filename
            (bool) as_ubyte: flag to save stamp as uint8 image

        Returns:
            None

        """
        # saves each stamp to a repo of the form root->id->index
        import os
        from pathlib import Path

        for stamp in self.stamps.flatten():
            sid = stamp.id
            index = "{}_{}".format(*stamp.index)
            s = stamp.summary_stamp(stamptype)  # uint8 for export (space saving)
            if as_ubyte:
                s = skimage.img_as_ubyte(s)
            target = Path(os.path.join(target_root, sid, index, "{}.png".format(title)))
            os.makedirs(target.parent, exist_ok=True)
            skimage.io.imsave(target, s)

    def __str__(self):
        return "IDs: {}, Device: {}, ImageReference: {}".format(
            self.ids, str((self.device.setup, self.device.dname)), self.data_ref
        )


class Stamp:

    chamberrad = 16
    outerchamberbound = 5
    circlePara1Index = 50
    circlePara2Index = 40

    def __init__(self, img, center, slice, index, id):
        """
        Constructor for a Stamp object, which contains feature data and parameters and permits
        feature finding.

        Arguments:
            (np.ndarray) img:
            (tuple) center
            (tuple) slice
            (tuple) index
            (str | int) id:

        Returns:
            None

        """

        self.data = img  # the actual stamp data
        self.index = index
        self.slice = slice
        self.id = id
        self.chamber = None
        self.button = None

    def defineChamber(self, center, radius):
        """
        Manually defines a chamber. The passed center coordinates are with respect to the stamp
        coordinate system (origin upper left).

        Arguments:
            (tuple) center:
            (int) radius:

        Returns:
            None

        """
        if center != center:
            self.chamber = Chamber.BlankChamber()
        else:
            p = Stamp.circularSubsection(self.data, center, radius)
            self.chamber = Chamber(self.data, p["mask"], p["center"], p["radius"])

    def defineButton(self, center, radius, annulus_radii):
        """
        Manually defines a Button. The passed center coordinates are with respect to the stamp
        coordinate system (origin upper left).

        Arguments:
            (tuple) center:
            (int) radius:
            (tuple) annulus radii:

        Returns:
            None

        """

        b = Stamp.circularSubsection(self.data, center, radius)
        o = Stamp.circularSubsection(
            self.data, center, annulus_radii[1]
        )  # The circles can extend past the edge of the image

        b_mask = b["mask"]
        o_mask = o["mask"]
        annulus_mask = ~(o_mask ^ b_mask)
        self.button = Button(
            self.data,
            b_mask,
            annulus_mask,
            b["center"],
            b["radius"],
            (b["radius"], b["radius"]),
        )

    def summarize(self):
        """
        Summarizes a stamp as a dictionary of parameterized chamber, button, and stamp features.

        Arguments:
            None

        Returns:
            (dict) stamp features

        """

        c_r = {}
        b_r = {}
        if self.chamber:
            c_r = self.chamber.summary
        if self.button:
            b_r = self.button.summary
        stampInfo = {
            "xslice": (self.slice[0].start, self.slice[0].stop),
            "yslice": (self.slice[1].start, self.slice[1].stop),
            "id": self.id,
        }
        return {**c_r, **b_r, **stampInfo}

    def summary_stamp(self, stamptype):
        """
        Annotes a stamp image and overlays chamber or button borders.

        Arguments:
            (str) stamptype: parameterized feature type to draw onto stamp ('chamber' | 'button')

        Returns:
            (np.ndarray) the annotated stamp image array

        """
        # stamptype: ('chamber', 'button')
        if stamptype == "chamber":
            circles = [[self.chamber.radius, self.chamber.center]]
            index = "{}.{} | {}".format(self.index[0], self.index[1], self.id)
            return annotateStamp(self.data, circles, index, "")
        elif stamptype == "button":
            circles = [
                [self.button.disk_radius, self.button.center],
                [self.button.annulus_radii[1], self.button.center],
            ]
            index = "{}.{} | {}".format(self.index[0], self.index[1], self.id)
            val = "{}, {}".format(
                int(self.button.summary["summed_button_BGsub"]),
                int(self.button.summary["summed_button_annulus_normed"]),
            )
            return annotateStamp(self.data, circles, index, val)
        else:
            raise ValueError(
                'Invalid stamp type. Valid values are "chamber" or "button"'
            )

    @staticmethod
    def circularSubsection(img, center, radius):
        """
        Given an image stamp, chamber/button center position and radius, returns the raw and
        analyzed chamber/button pixel values for that chamber/button. Uses a mask.

        Arguments:
            (np.ndarray) image:
            (tuple) center: x,y center location of the image
            (int) radius: radius of the chamber/button to be masked

        Returns:
            (dict) circularSubsection features

        """

        imageCopy = img.copy()
        mask = np.zeros(imageCopy.shape)
        cv2.circle(mask, center, radius, 1, -1)  # Warning: MODIFIES mask IN PLACE!!
        mask = mask.astype(np.bool)

        insert = np.where(mask)
        intensities = img[insert]

        return {
            "mask": ~mask,
            "intensities": intensities,
            "center": center,
            "radius": int(radius),
        }

    def findChamber(self, coerce_center=False):
        """
        Uses Hough transform to find a chamber.

        Arguments:
            (np.ndarray) imageCopy: chamber stamp image.

        Returns:
            (dict) optimizedSpotParams: optimal found chamber border parameters

        """

        chamberRadius = Stamp.chamberrad
        outerChamberBound = Stamp.outerchamberbound
        img = self.data

        if coerce_center:
            p = Stamp.circularSubsection(
                img, (int(img.shape[0] / 2), int(img.shape[0] / 2)), chamberRadius
            )
            self.chamber = Chamber(img, p["mask"], p["center"], p["radius"])
            return

        else:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # Will throw warning due to precision loss
                cimg = skimage.img_as_ubyte(img, force_copy=True)

            # searchRadii
            minRad = chamberRadius
            maxRad = minRad + outerChamberBound

            # find circles
            circles = cv2.HoughCircles(
                cimg,
                cv2.HOUGH_GRADIENT,
                2,
                10,
                param1=self.circlePara1Index,
                param2=self.circlePara2Index,
                minRadius=minRad,
                maxRadius=maxRad,
            )

            circlePara1Index = self.circlePara1Index
            # If no circles found, loosen gradient threshold
            while type(circles) is not np.ndarray and circlePara1Index > 5:
                circles = cv2.HoughCircles(
                    cimg,
                    cv2.HOUGH_GRADIENT,
                    2,
                    10,
                    param1=circlePara1Index,
                    param2=self.circlePara2Index,
                    minRadius=minRad + 1,
                    maxRadius=maxRad + 2,
                )
                circlePara1Index -= 1

            # If still none found, return a blank chamber (failed)
            if not np.any(circles):
                m = "No chamber border found for chamber {}".format(str(self.index))
                warnings.warn(m)
                self.chamber = Chamber.BlankChamber()
                return

            # Else, round the resulting circle params
            circles = np.around(circles)

            # Then select circle of highest summed I from those found
            bestCircle = None
            bestIntensity = 0
            for i in circles[0, :]:
                if len(circles[0, :]) == 1:
                    bestCircle = i
                    break
                # pick the set of circles that maximizes intensity inside of it (may have ties)
                circleResultsSum = np.sum(
                    Stamp.circularSubsection(cimg, (int(i[0]), int(i[1])), int(i[2]))[
                        "intensities"
                    ]
                )
                if circleResultsSum > bestIntensity:
                    bestIntensity = circleResultsSum
                    bestCircle = i

            # Calculate the final parameters
            p = Stamp.circularSubsection(
                img, (int(bestCircle[0]), int(bestCircle[1])), int(bestCircle[2])
            )
            self.chamber = Chamber(img, p["mask"], p["center"], p["radius"])

    def findButton(self):
        """
        Button finding algorithm using Craig's "grid search" optimization.
        Searches sparse grid of tile position centers, finds optimum, then refines by searching local
        neighborhood. Then, fits the radius and re-fits the centerposition after each decrease in radius.
        Terminates when either the minRadius is reached or finds a bright circle with small standard deviation
        within the found circle border falls below specified threshold.

        Arguments:
            None

        Returns:
            (dict) bestSpotParams: {'mask': ~mask, 'intensities': intensities, 'center': center, 'radius': int(radius)}

        """

        ######## DEFAULTS ########
        searchSpacing = 7
        radius = 15  # was 2x2 = 15, 4x4 = 7
        tileWidth = 110
        tileHeight = 110
        refiningRange = 7
        minRadius = 9
        stdCutoff = 0.9
        boundingInsetRatio = 0.3
        ###########################

        imagestamp = self.data

        maxI = 0
        bestSpotParams = None
        fitRadius = radius
        localBGRadius = radius * 2

        boundingInset = int(tileWidth * boundingInsetRatio)

        # Crude initial fit of center position (sparse initial serach grid, entire image stamp) by maximizing summed intensity
        for xIndex in range(boundingInset, tileWidth - boundingInset, searchSpacing):
            for yIndex in range(
                boundingInset, tileHeight - boundingInset, searchSpacing
            ):
                features = Stamp.circularSubsection(
                    imagestamp, (xIndex, yIndex), fitRadius
                )
                summedI = np.nansum(features["intensities"])
                if summedI > maxI:
                    maxI = deepcopy(summedI)
                    bestSpotParams = deepcopy(features)

        # If the image is perfectly black in the bounding region, it's necessary to just pick the center position as a placeholder
        if not bestSpotParams:
            warnmsg = "No intensity observed for chamber {}".format(self.index)
            warnings.warn(warnmsg)
            bestSpotParams = deepcopy(
                Stamp.circularSubsection(
                    imagestamp, (int(tileWidth / 2), int(tileHeight / 2)), fitRadius
                )
            )
            buttonBound = Stamp.circularSubsection(
                imagestamp, bestSpotParams["center"], radius
            )
            outerBound = Stamp.circularSubsection(
                imagestamp, bestSpotParams["center"], localBGRadius
            )  # The circles can extend past the edge of the image
            b_mask = buttonBound["mask"]
            o_mask = outerBound["mask"]
            annulus_mask = ~(o_mask ^ b_mask)
            self.button = Button(
                imagestamp,
                b_mask,
                annulus_mask,
                buttonBound["center"],
                buttonBound["radius"],
                (buttonBound["radius"], outerBound["radius"]),
            )
            return

        # Fine-tuning center position (dense local array for search grid) by maximizing summed intensity
        for xIncrement in np.linspace(
            bestSpotParams["center"][0] - refiningRange,
            bestSpotParams["center"][0] + refiningRange,
            num=2 * refiningRange,
            dtype=int,
        ):
            for yIncrement in np.linspace(
                bestSpotParams["center"][1] - refiningRange,
                bestSpotParams["center"][1] + refiningRange,
                num=2 * refiningRange,
                dtype=int,
            ):
                features = Stamp.circularSubsection(
                    imagestamp, (xIncrement, yIncrement), fitRadius
                )
                summedI = np.nansum(features["intensities"])
                if summedI > maxI:
                    maxI = deepcopy(summedI)
                    bestSpotParams = deepcopy(features)

        refStdDev = np.nanstd(bestSpotParams["intensities"])

        # Refines center position by optimizing radius via watershed method
        while bestSpotParams["radius"] > minRadius:
            fitRadius -= 1

            maxIAtRadius = 0
            bestParamsAtRadius = bestSpotParams

            for xIncrement in np.linspace(
                bestParamsAtRadius["center"][0] - refiningRange,
                bestParamsAtRadius["center"][0] + refiningRange,
                num=2 * refiningRange,
                dtype=int,
            ):
                for yIncrement in np.linspace(
                    bestParamsAtRadius["center"][1] - refiningRange,
                    bestParamsAtRadius["center"][1] + refiningRange,
                    num=2 * refiningRange,
                    dtype=int,
                ):
                    features = Stamp.circularSubsection(
                        imagestamp, (xIncrement, yIncrement), fitRadius
                    )
                    summedI = np.nansum(features["intensities"])
                    if summedI > maxIAtRadius:
                        maxIAtRadius = deepcopy(summedI)
                        bestParamsAtRadius = deepcopy(features)
            # If the radius has shrunk the optimal circle to w/in bright bounds, stop the fitting and use that circle center)
            if np.nanstd(bestParamsAtRadius["intensities"]) < stdCutoff * refStdDev:
                bestSpotParams = Stamp.circularSubsection(
                    imagestamp, bestParamsAtRadius["center"], radius
                )
                break
            else:
                maxI = deepcopy(maxIAtRadius)
                bestSpotParams = deepcopy(bestParamsAtRadius)

        # If radius fitting didn't work, just go with the parameters from before
        buttonBound = Stamp.circularSubsection(
            imagestamp, bestSpotParams["center"], radius
        )
        outerBound = Stamp.circularSubsection(
            imagestamp, bestSpotParams["center"], localBGRadius
        )  # The circles can extend past the edge of the image

        b_mask = buttonBound["mask"]
        o_mask = outerBound["mask"]
        annulus_mask = ~(o_mask ^ b_mask)

        self.button = Button(
            imagestamp,
            b_mask,
            annulus_mask,
            buttonBound["center"],
            buttonBound["radius"],
            (buttonBound["radius"], outerBound["radius"]),
        )

    def __str__(self):
        return "Stamp| ID:{}, Index:{}".format(self.id, self.index)

    def _repr_pretty_(self, p, cycle=True):
        p.text("<{}>".format(self.__str__()))


class Chamber:
    def __init__(self, stampdata, disk, center, radius, empty=False):
        """
        Constructor for a Chamber object

        Arguments:
            (np.ndarray) stampdata: the original stamp data
            (np.ndarray) disk: a boolean mask for the stampdata FALSE within the found chamber
            (tuple) center: chamber center coordinates, with respect to stampdata coord. system
            (int) radius: chamber radius
            (bool) empty: flag for empty chamber

        Returns:
            None

        """

        self.blankFlag = empty
        self.stampdata = stampdata  # uint16 ndarray
        self.disk = disk  # a mask
        self.disk_intensities = ma.compressed(ma.array(stampdata, mask=disk))
        self.center = center
        self.radius = radius
        self.summary = self.summarize()

    def get_disk(self):
        """
        Generates the masked array corresponding to the chamber.

        Arguments:
            None

        Returns:
            (np.ma) masked array of the stamp chamber

        """

        return ma.array(self.stampdata, mask=self.disk)

    def summarize(self):
        """
        Summarizes a chamber as a dictionary of paramters mapped from their descriptors

        Arguments:
            None

        Returns:
            (dict) summary of chamber parameters

        """

        features = [
            "median_chamber",
            "sum_chamber",
            "std_chamber",
            "x_center_chamber",
            "y_center_chamber",
            "radius_chamber",
        ]

        if self.blankFlag:
            return dict(zip(features, list(np.full(len(features), np.nan))))

        disk = self.get_disk()
        medI = int(ma.median(disk))
        sumI = int(disk.sum())
        sdI = int(disk.std())

        vals = [medI, sumI, sdI, self.center[0], self.center[1], self.radius]
        return dict(zip(features, vals))

    @classmethod
    def BlankChamber(cls):
        """
        Factory fresh empty chamber, hot off the press
        """
        bc = cls(*([np.nan] * 4), **{"empty": True})
        return bc


class Button:
    localBG_Margin = 10

    def __init__(
        self, stampdata, disk, annulus, center, disk_radius, annulus_radii, empty=False
    ):
        """
        Constructor for a Button object

        Arguments:
            (np.ndarray) stampdata: the original stamp data
            (np.ndarray) disk: a boolean mask for the stampdata FALSE within the found chamber
            (np.ndarray) annulus: a boolean mask for the stampdata FALSE within the button annulus
                (local background)
            (tuple) center: chamber center coordinates, with respect to stampdata coord. system
            (int) disk_radius: button radius
            (tuple) annulus radii: inner and outer radii of the annulus (innerrad, outerrad)
            (bool) empty: flag for empty button

        Returns:
            None

        """

        self.blankFlag = empty
        self.stampdata = stampdata  # uint16 ndarray
        self.disk = disk  # a mask
        self.disk_intensities = ma.compressed(self.get_disk())
        self.annulus = annulus  # a mask
        self.annulus_intensities = ma.compressed(self.get_annulus())
        try:
            self.annulus_to_disk_ratio = len(self.annulus_intensities) / len(
                self.disk_intensities
            )
        except:
            warnings.warn(
                "Annulus ratio could not be calculated.\nButton Intensities Are Of Length Zero.\
                            Annulus to disk ratio is NaN"
            )
            self.annulus_to_disk_ratio = np.nan
        self.center = center
        self.disk_radius = disk_radius
        self.annulus_radii = annulus_radii
        self.summary = self.summarize()

    def get_disk(self):
        """
        Generates the masked array corresponding to the button.

        Arguments:
            None

        Returns:
            (np.ma) masked array of the stamp button

        """

        return ma.array(self.stampdata, mask=self.disk)

    def get_annulus(self):
        """
        Generates the masked array corresponding to the annulus.

        Arguments:
            None

        Returns:
            (np.ma) masked array of the stamp button annulus

        """

        return ma.array(self.stampdata, mask=self.annulus)

    def summarize(self):
        """
        Summarizes a button as a dictionary of paramters mapped from their descriptors

        Arguments:
            None

        Returns:
            (dict) summary of chamber parameters

        """

        features_disk = [
            "median_button",
            "summed_button",
            "summed_button_BGsub",
            "std_button",
            "x_button_center",
            "y_button_center",
            "radius_button_disk",
        ]
        features_ann = [
            "median_button_annulus",
            "summed_button_annulus_normed",
            "std_button_annulus_localBG",
            "inner_radius_button_annulus",
            "outer_radius_button_annulus",
        ]

        if self.blankFlag:
            return dict(
                zip(
                    features_disk + features_ann,
                    list(np.full(len(features_disk + features_ann), np.nan)),
                )
            )

        disk = self.get_disk()
        annulus = self.get_annulus()
        medI_disk = int(ma.median(disk))
        sumI_disk = int(disk.sum())
        sdI_disk = int(disk.std())
        medI_ann = int(ma.median(annulus))
        sumI_ann_normed = int(annulus.sum() / self.annulus_to_disk_ratio)
        sdI_ann = int(annulus.std())
        sumI_BGsub = sumI_disk - sumI_ann_normed

        vals_disk = [
            medI_disk,
            sumI_disk,
            sumI_BGsub,
            sdI_disk,
            self.center[0],
            self.center[1],
            self.disk_radius,
        ]
        vals_ann = [
            medI_ann,
            sumI_ann_normed,
            sdI_ann,
            self.annulus_radii[0],
            self.annulus_radii[1],
        ]
        return dict(zip(features_disk + features_ann, vals_disk + vals_ann))

    @classmethod
    def BlankButton(cls):
        bb = cls(*([np.nan] * 6), **{"empty": True})
        return bb


def annotateStamp(data, circles, index, val):
    """
    Annotates a stamp image with an index, a feature value, and arbitrary circles
    (for chamber and button feature drawing). Text is drawn in white. Circles are drawn in
    black and white

    The index a(nd value) is/are drawn on the top of the stamp.

    Arguments:
        (tuple) circle paramters of the form (radius, (centerx, centery))
        (str) index: stamp index
        (str) val: stamp value

    Returns:
        (np.ndarray) annotated stamp image

    """

    working = deepcopy(data)
    d = cv2.copyMakeBorder(deepcopy(data), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=60000)
    cv2.putText(d, index, (2, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, 60000)  # Index
    cv2.putText(d, val, (2, len(d) - 4), cv2.FONT_HERSHEY_PLAIN, 0.7, 60000)  # Index
    for rad, center in circles:
        if center != center:
            pass
        else:
            cv2.circle(d, center, rad + 2, 0, thickness=1)
            cv2.circle(d, center, rad + 1, (2**16) - 1, thickness=1)
            cv2.circle(d, center, rad, 0, thickness=1)
    return d
