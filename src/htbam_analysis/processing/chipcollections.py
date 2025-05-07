# title             : chipcollections.py
# description       :
# authors           : Daniel Mokhtari
# credits           : Craig Markin
# date              : 20180615
# version update    : 20191001
# version           : 0.1.0
# usage             : With permission from DM
# python_version    : 3.7

# General Python
import os
import logging
from glob import glob
from pathlib import Path
from collections import namedtuple, OrderedDict
import pandas as pd

from tqdm import tqdm
from skimage import io
import re
from typing import Callable, Optional

from htbam_analysis.processing.chip import ChipImage


class ChipSeries:
    def __init__(self, device, series_index, attrs=None):
        """
        Constructor for a ChipSeries object.

        Arguments:
            (experiment.Device) device:
            (int) series_index:
            (dict) attrs: arbitrary ChipSeries metdata

        Returns:
            Return

        """

        self.device = device  # Device object
        self.attrs = attrs  # general metadata fro the chip
        self.series_indexer = series_index
        self.description = description
        self.chips = {}
        self.series_root = None
        logging.debug("ChipSeries Created | {}".format(self.__str__()))

    def add_file(self, identifier, path, channel, exposure):
        """
        Adds a ChipImage of the image at path to the ChipSeries, mapped from the passed identifier.

        Arguments:
            (Hashable) identifier: a unique chip identifier
            (str) path: image file path
            (str) channel: imaging channel
            (int) exposure: imaging exposure time (ms)

        Returns:
            None

        """

        source = Path(path)
        chipParams = (self.device.corners, self.device.pinlist, channel, exposure)
        self.chips[identifier] = ChipImage(
            self.device, source, {self.series_indexer: identifier}, *chipParams
        )
        logging.debug("Added Chip | Root: {}/, ID: {}".format(source, identifier))

    def load_files(self, root, channel, exposure, indexes=None, custom_glob=None):
        """
        Loads indexed images from a directory as ChipImages.
        Image filename stems must be of the form *_index.tif.

        Arguments:
            (str) root: directory path containing images
            (str) channel: imaging channel
            (int) exposure: imaging exposure time (ms)
            (list | tuple) indexes: custom experimental inde

        Returns:
            None

        """

        self.series_root = root

        glob_pattern = "*BGSubtracted_StitchedImg*.tif"
        if custom_glob:
            glob_pattern = custom_glob

        if not indexes:
            r = Path(root)
            img_files = [
                i
                for i in list(r.glob(glob_pattern))
                if not "ChamberBorders" in i.stem or "Summary" in i.stem
            ]
            
            if len(img_files) < 1:
                raise ProcessingException(f"No images found! Looked in directory \"{root}\" for images that matched the pattern: \"{glob_pattern}\"")
            img_files = [img for img in img_files if img.parts[-1][0] != "."]
            img_paths = [Path(os.path.join(r.parent, img)) for img in img_files]
            pattern= re.compile("^([a-z|A-Z]*_){1,2}([0-9]*_){1,3}[0-9]*$")
            
            correct_setting_imgs = []

            for img in img_paths:
                if pattern.match(img.stem):
                    params = re.sub("^([A-Z|a-z]*_){1,2}", "", img.stem)
                    img_exposure = params.split("_")[0]
                    img_channel = params.split("_")[1]
            
                    if img_exposure == str(exposure) and img_channel == str(channel):
                        correct_setting_imgs.append(img)
                    else:
                        pass
                else:
                    raise ProcessingException(f"Malformed image name found \"{img.stem}\". Make sure any decimals in concentration are replaced with underscores.")
         
            
            record = {float(".".join(re.sub("^([A-Z|a-z]*_){1,2}", "", path.stem).split("_")[2:])): path for path in correct_setting_imgs}
            chipParams = (self.device.corners, self.device.pinlist, channel, exposure)
            self.chips = {
                identifier: ChipImage(
                    self.device, source, {self.series_indexer: identifier}, *chipParams
                )
                for identifier, source in record.items()
            }
            
            keys = list(self.chips.keys())
            print(keys)
            logging.debug("Loaded Series | Root: {}/, IDs: {}".format(root, keys))

    def summarize(self):
        """
        Summarize the ChipSeries as a Pandas DataFrame for button and/or chamber features
        identified in the chips contained.

        Arguments:
            None

        Returns:
            (pd.DataFrame) summary of the ChipSeries

        """

        summaries = []
        for i, r in self.chips.items():
            df = r.summarize()
            df[self.series_indexer] = i
            summaries.append(df)
        return pd.concat(summaries).sort_index()

    def map_from(self, reference, mapto_args={}):
        """
        Maps feature positions from a reference chip.ChipImage to each of the ChipImages in the series.
        Specific features can be mapped by passing the optional mapto_args to the underlying
        mapper.

        Arguments:
            (chip.ChipImage) reference: reference image (with found button and/or chamber features)
            (dict) mapto_args: dictionary of keyword arguments passed to ChipImage.mapto().

        Returns:
            None

        """

        for chip in tqdm(
            self.chips.values(),
            desc="Series <{}> Stamped and Mapped".format(self.description),
        ):
            chip.stamp()
            reference.mapto(chip, **mapto_args)

    def from_record():
        """
        TODO: Import imaging from a Stitching record.
        """
        return

    def _repr_pretty_(self, p, cycle=True):
        p.text("<{}>".format(self.device.__str__()))

    def save_summary(self, outPath=None):
        """
        Generates and exports a ChipSeries summary Pandas DataFrame as a bzip2 compressed CSV file.

        Arguments:
            (str) outPath: target directory for summary

        Returns:
            None

        """

        target = self.series_root
        if outPath:
            target = outPath
        df = self.summarize()
        fn = "{}_{}_{}.csv.bz2".format(
            self.device.dname, self.description, "ChipSeries"
        )
        df.to_csv(os.path.join(target, fn), compression="bz2")

    def save_summary_images(self, outPath=None, featuretype="chamber"):
        """
        Generates and exports a stamp summary image (chip stamps concatenated)

        Arguments:
            (str) outPath: user-define export target directory
            (str) featuretype: type of feature overlay ('chamber' | 'button')

        Returns:
            None

        """

        target_root = self.series_root
        if outPath:
            target_root = outPath
        target = os.path.join(target_root, "SummaryImages")  # Wrapping folder
        os.makedirs(target, exist_ok=True)
        for c in self.chips.values():
            image = c.summary_image(featuretype)
            name = "{}_{}.tif".format("Summary", c.data_ref.stem)
            outDir = os.path.join(target, name)
            io.imsave(outDir, image, plugin="tifffile")
        logging.debug("Saved Summary Images | Series: {}".format(self.__str__()))

    def _delete_stamps(self):
        """
        Deletes and forces garbage collection of stamps for all ChipImages

        Arguments:
            None

        Returns:
            None

        """

        for c in self.chips.values():
            c._delete_stamps()

    def repo_dump(self, target_root, title, as_ubyte=False, featuretype="button"):
        """
        Save the chip stamp images to the target_root within folders title by chamber IDs

        Arguments:
            (str) target_root:
            (str) title:
            (bool) as_ubyte:

        Returns:
            None

        """

        for i, c in self.chips.items():
            title = "{}{}_{}".format(self.device.setup, self.device.dname, i)
            c.repo_dump(featuretype, target_root, title, as_ubyte=as_ubyte)

    def __str__(self):
        return "Description: {}, Device: {}".format(
            self.description, str((self.device.setup, self.device.dname))
        )


class StandardSeries(ChipSeries):
    def __init__(self, device, description, attrs=None):
        """
        Constructor for a StandardSeries object.

        Arguments:
            (experiment.Device) device: Device object
            (str) description: Terse description (e.g., 'cMU')
            (dict) attrs: arbitrary StandardSeries metadata

        Returns:
            None

        """

        self.device = device  # Device object
        self.attrs = attrs  # general metadata fro the chip
        self.series_indexer = "concentration_uM"
        self.description = description
        self.chips = None
        self.series_root = None
        logging.debug("StandardSeries Created | {}".format(self.__str__()))

    def get_hs_key(self):
        return max(self.chips.keys())

    def get_highstandard(self):
        """
        Gets the "maximal" (high standard) chip object key

        Arguments:
            None

        Returns:
            None

        """

        return self.chips[self.get_hs_key()]

    def map_from_hs(self, mapto_args={}):
        """
        Maps the chip image feature position from the StandardSeries high standard to each
        other ChipImage

        Arguments:
            (dict) mapto_args: dictionary of keyword arguments passed to ChipImage.mapto().

        Returns:
            None

        """

        reference_key = {self.get_hs_key()}
        all_keys = set(self.chips.keys())
        hs = self.get_highstandard()

        for key in tqdm(
            all_keys - reference_key,
            desc="Processing Standard <{}>".format(self.__str__()),
        ):
            self.chips[key].stamp()
            hs.mapto(self.chips[key], **mapto_args)

    def process(self, featuretype="chamber", coerce_center=False):
        """
        A high-level (script-like) function to execute analysis of a loaded Standard Series.
        Processes the high-standard (stamps and finds chambers) and maps processed high standard
        to each other ChipImage

        Arguments:
            (str) featuretype: stamp feature to map

        Returns:
            None

        """

        hs = self.get_highstandard()
        hs.stamp()
        hs.findChambers(coerce_center=coerce_center)
        self.map_from_hs(mapto_args={"features": featuretype})

    def process_summarize(self):
        """
        Simple wrapper to process and summarize the StandardSeries Data

        Arguments:
            None

        Returns:
            None

        """

        self.process()
        df = self.summarize()
        return df

    def save_summary(self, outPath=None):
        """
        Generates and exports a StandardSeries summary Pandas DataFrame as a bzip2 compressed CSV file.

        Arguments:
            (str | None) outPath: target directory for summary. If None, saves to the series root.

        Returns:
            None

        """

        target = self.series_root
        if outPath:
            target = outPath
        df = self.summarize()
        fn = "{}_{}_{}.csv.bz2".format(
            self.device.dname, self.description, "StandardSeries_Analysis"
        )
        df.to_csv(os.path.join(target, fn), compression="bz2")
        logging.debug(
            "Saved StandardSeries Summary | Series: {}".format(self.__str__())
        )


class Timecourse(ChipSeries):
    def __init__(self, device, description, attrs=None):
        """
        Constructor for a Timecourse object.

        Arguments:
            (experiment.Device) device:
            (str) description: user-define description
            (dict) attrs: arbitrary metadata

        Returns:
            None

        """

        self.device = device  # Device object
        self.attrs = attrs  # general metadata fro the chip
        self.description = description
        self.series_indexer = "time_s"
        self.chips = None
        self.series_root = None
        logging.debug("Timecourse Created | {}".format(self.__str__()))

    def process(self, reference, featuretype="chamber"):
        """
        Map chamber positions (stamp, feature mapping) from the provided reference

        Arguments:
            (ChipImage) chamber_reference: reference ChipImage for chamber or button position mapping
            (str) featuretype: type of feature to map ('chamber' | 'button' | 'all')

        Returns:
            None

        """

        self.map_from(reference, mapto_args={"features": featuretype})

    def process_summarize(self, reference):
        """

        Process (stamp, positions and features mapping) and summarize the resulting image data
        as a Pandas DataFrame

        Arguments:
            (ChipImage) reference: reference ChipImage for chamber ro button position mapping

        Returns:
            (pd.DataFrame) DataFrame of chip feature information

        """

        self.process(reference)
        df = self.summarize()
        return df

    def save_summary(self, outPath=None):
        """

        Arguments:
            (str) outPath: target directory for summary

        Returns:
            None

        """

        target = self.series_root
        if outPath and os.isdir(outPath):
            target = outPath
        df = self.summarize()
        fn = "{}_{}_{}.csv.bz2".format(
            self.device.dname, self.description, "Timecourse"
        )
        df.to_csv(os.path.join(target, fn), compression="bz2")
        logging.debug(
            "Saved Timecourse Summary | Timecourse: {}".format(self.__str__())
        )


class Titration(ChipSeries):
    # TODO
    pass


class ChipQuant:
    def __init__(self, device, description, attrs=None):
        """
        Constructor for a ChipQuant object

        Arguments:
            (experiment.Device) device: device object
            (str) description: terse user-define description
            (dict) attrs: arbitrary metadata

        Returns:
            None

        """

        self.device = device
        self.description = description
        self.attrs = attrs
        self.chip = None
        self.processed = False
        logging.debug("ChipQuant Created | {}".format(self.__str__()))

    def load_file(self, path, channel, exposure):
        """
        Loads an image file as a ChipQuant.

        Arguments:
            (str) path: path to image
            (str) channel: imaging channel
            (int) exposure: exposure time (ms)

        Returns:
            None

        """

        p = Path(path)
        chipParams = (self.device.corners, self.device.pinlist, channel, exposure)
        self.chip = ChipImage(self.device, p, {}, *chipParams)
        logging.debug("ChipQuant Loaded | Description: {}".format(self.description))

    def process(self, reference=None, mapped_features="button", coerce_center=False):
        """
        Processes a chip quantification by stamping and finding buttons. If a reference is passed,
        button positions are mapped.

        Arguments:
            (ChipImage) button_ref: Reference ChipImage
            (st) mapped_features: features to map from the reference (if button_ref)

        Returns:
            None

        """

        self.chip.stamp()
        if not reference:
            if mapped_features == "button":
                self.chip.findButtons()
            elif mapped_features == "chamber":
                self.chip.findChambers(coerce_center=coerce_center)
            elif mapped_features == "all":
                self.chip.findButtons()
                self.chip.findChambers(coerce_center=coerce_center)
            else:
                raise ValueError(
                    'Must specify valid feature name to map ("button", "chamber", or "all"'
                )
        else:
            reference.mapto(self.chip, features=mapped_features)
        self.processed = True
        logging.debug("Features Processed | {}".format(self.__str__()))

    def summarize(self):
        """
        Summarize the ChipQuant as a Pandas DataFrame for button features
        identified in the chips contained.

        Arguments:
            None

        Returns:
            (pd.DataFrame) summary of the ChipSeries

        """

        if self.processed:
            return self.chip.summarize()
        else:
            raise ValueError("Must first process ChipQuant")

    def process_summarize(self, reference=None, process_kwrds={}):
        """
        Script-like wrapper for process() and summarize() methods

        Arguments:
            (chip.ChipImage) reference: ChipImage to use as a reference
            (dict) process_kwrds: keyword arguments passed to ChipQuant.process()

        Returns:
            (pd.DataFrame) summary of the ChipSeries


        """
        self.process(reference=reference, **process_kwrds)
        return self.summarize()

    def save_summary_image(self, outPath_root=None, feature_type="button"):
        """
        Generates and exports a stamp summary image (chip stamps concatenated)

        Arguments:
            (str) outPath_root: path of user-defined export root directory

        Returns:
            None

        """

        outPath = self.chip.data_ref.parent
        if outPath_root:
            if not os.isdir(outPath_root):
                em = "Export directory does not exist: {}".format(outPath_root)
                raise ValueError(em)
            outPath = Path(outPath_root)

        target = os.path.join(outPath, "SummaryImages")  # Wrapping folder
        os.makedirs(target, exist_ok=True)

        c = self.chip
        image = c.summary_image(feature_type)
        name = "{}_{}.tif".format("Summary", c.data_ref.stem)
        outDir = os.path.join(target, name)
        io.imsave(outDir, image, plugin="tifffile")
        logging.debug(
            "Saved ChipQuant Summary Image | ChipQuant: {}".format(self.__str__())
        )

    def repo_dump(self, outPath_root, as_ubyte=False):
        """
        Export the ChipQuant chip stamps to a repository (repo). The repo root contains a
        directory for each unique pinlist identifier (MutantID, or other) and subdirs
        for each chamber index. Stamps exported as .png

        Arguments:
            (str): outPath_root: path of user-defined repo root directory
            (bool) as_ubyte: flag to export the stamps as uint8 images

        Returns:
            None

        """

        title = "{}{}_{}".format(self.device.setup, self.device.dname, self.description)
        self.chip.repo_dump("button", outPath_root, title, as_ubyte=as_ubyte)

    def __str__(self):
        return "Description: {}, Device: {}".format(
            self.description, str((self.device.setup, self.device.dname))
        )


class Assay:
    def __init__(self, device, description, attrs=None):
        """
        Constructor for an Assay class.

        Arguments:
            (experiment.Device) device:
            (str) description: user-defined assay description
            (dict) attrs: arbitrary metadata

        Returns:
            None

        """

        self.device = device  # Device object
        self.attrs = attrs  # general metadata for the chip
        self.description = description
        self.series = None
        self.quants = []

    def add_series(self, c):
        """
        Setter to add an arbitary ChipSeries to the assay

        Arguments:
            (ChipSeries) c: a chipseries (or subclass)

        Returns:
            None

        """

        if isinstance(c, ChipSeries):
            self.series = c
        else:
            raise TypeError("Must provide a valid ChipSeries")

    def add_quant(self, c):
        """
        Setter to add an arbitry ChipQuant to the Assay.

        Arguments:
            (ChipQuant) c: a chipquant

        Returns:
            None


        """
        self.quants.append(c)


class TurnoverAssay(Assay):
    def merge_summarize(self):
        """
        A script-like method to summarize each quantification and join them with summary
        of the ChipSeries.

        Arguments:
            None

        Returns:
            (pd.DataFrame) a Pandas DataFrame summarizing the Assay

        """

        quants_cleaned = []
        for quant in self.quants:
            desc = quant.description
            summary = quant.chip.summarize()
            toAdd = summary.drop(columns=["id"])
            quants_cleaned.append(
                summary.add_suffix("_{}".format(desc.replace(" ", "_")))
            )

        kinSummary = self.series.summarize()
        merged = kinSummary.join(
            quants_cleaned, how="left", lsuffix="_kinetic", rsuffix="_buttonquant"
        )
        return merged


class AssaySeries:
    def __init__(
        self, device, descriptions, chamber_ref, button_ref, attrs=None, assays_attrs=[]
    ):
        """
        Constructor for and AssaySeries, a high-level class representing a collection of related TurnoverAssays.
        Holds arbitrary ordered TurnoverAssays as a dictionary. Designed specficially for eMITOMI use.
        TurnoverAssays are generated when the object is constructed, but must be populated after with
        kinetic and quantificationd data.

        Arguments:
            (experiment.Device) device: Device object
            (list | tuple) descriptions: Descriptions assocated with assays
            (chip.ChipImage) chamber_ref: a ChipImage object with found chambers for referencing
            (ChipQuant) button_ref: a ChipQuant object with found buttons for referencing
            (dict) attrs:arbitrary StandardSeries metadata

        Returns:
            None

        """

        self.device = device
        self.assays = OrderedDict(
            [
                (description, TurnoverAssay(device, description))
                for description in descriptions
            ]
        )
        self.chamber_ref = chamber_ref
        self.button_ref = button_ref
        self.chamber_root = None
        self.button_root = None
        self.attrs = attrs
        logging.debug("AssaySeries Created | {}".format(self.__str__()))
        logging.debug(
            "AssaySeries Chamber Reference Set | {}".format(chamber_ref.__str__())
        )
        logging.debug(
            "AssaySeries Button Reference Set | {}".format(button_ref.__str__())
        )

    def load_kin(self, descriptions, paths, channel, exposure, custom_glob=None):
        """
        Loads kinetic imaging and descriptions into the AssaySeries.

        Given paths of imaging root directories, creates Timecourse objects and associates with
        the passed descriptions. Descriptions and paths must be of equal length. Descriptions and
        paths are associated on their order (order matters)

        Arguments:
            (list | tuple) descriptions: descriptions of the imaging (paths)
            (list | tuple) paths: paths to directories containing timecourse imaging
            (str) channel: imaging channel
            (int) exposure: exposure time (ms)

        Returns:
            None

        """

        len_series = len(self.assays)
        len_descriptions = len(descriptions)

        if len_descriptions != len_series:
            raise ValueError(
                "Descriptions and series of different lengths. Number of assays and descriptions must match."
            )
        kin_refs = list(
            zip(descriptions, paths, [channel] * len_series, [exposure] * len_series)
        )
        for desc, p, chan, exp in kin_refs:
            t = Timecourse(self.device, desc)
            t.load_files(p, chan, exp, custom_glob=custom_glob)
            self.assays[desc].series = t

    def load_quants(self, descriptions, paths, channel, exposure):
        """
        Loads chip quantification imaging and associates with Timecourse data for existing Assay objects

        Arguments:
            (list | tuple) descriptions: descriptions of the imaging (paths)
            (list | tuple) paths: paths to directories containing quantification imaging
            (str) channel: imaging channel
            (int) exposure: exposure time (ms)

        Returns:
            None

        """

        if len(descriptions) != len(paths):
            raise ValueError("Descriptions and paths must be of same length")

        len_series = len(self.assays)

        if len(descriptions) == 1:
            descriptions = self.assays.keys()
            paths = list(paths) * len_series

        bq_refs = list(
            zip(descriptions, paths, [channel] * len_series, [exposure] * len_series)
        )
        for desc, p, chan, exp in bq_refs:
            q = ChipQuant(self.device, "Button_Quant")
            q.load_file(p, chan, exp)
            self.assays[desc].add_quant(q)

    def parse_kineticsFolders(
        self, root, file_handles, descriptors, channel, exposure, pattern=None, custom_glob=None
    ):
        """
        Walks down directory tree, matches the passed file handles to the Timecourse descriptors,
        and loads kinetic imaging data. Default pattern is "*_{}*/*/StitchedImages", with {}
        file_handle

        Arguments:
            (str) root: path to directory Three levels above the StitchedImages folders (dir
                above unique assay folders)
            (list | tuple) file_handles: unique file handles to match to dirs in the root.
            (list | tuple) descriptors: unique kinetic imaging descriptors, order-matched to
                the file_handles
            (str) channel: imaging channel
            (int) exposure: exposure time (ms)
            (bool) pattern: custom UNIX-style pattern to match when parsing dirs

        Returns:
            None

        """

        self.chamber_root = root
        if not pattern:
            pattern = "*_{}*/*/StitchedImages"

        p = lambda f: glob(os.path.join(root, pattern.format(f)))[0]
        files = {
            (handle, desc): p(handle) for handle, desc in zip(file_handles, descriptors)
        }
        
        self.load_kin(descriptors, files.values(), channel, exposure, custom_glob=custom_glob)

    def parse_quantificationFolders(
        self, root, file_handles, descriptors, channel, exposure, pattern=None
    ):
        """
        Walks down directory tree, matches the passed file handles to the ChipQuant descriptors,
        and loads button quantification imaging data. Default pattern is "*_{}*/*/StitchedImages/
        BGSubtracted_StitchedImg*.tif", with {} file_handle

        Arguments:
            (str) root: path to directory Three levels above the StitchedImages folders (dir
                above unique assay folders)
            (list | tuple) file_handles: unique file handles to match to dirs in the root.
            (list | tuple) descriptors: unique kinetic imaging descriptors, order-matched to
                the file_handles. MUST BE THE SAME USED FOR parse_kineticsFolders
            (str) channel: imaging channel
            (int) exposure: exposure time (ms)
            (bool) pattern: custom UNIX-style pattern to match when parsing dirs

        Returns:
            None

        """

        if not pattern:
            pattern = "*_{}*/*/StitchedImages/BGSubtracted_StitchedImg*.tif"

        try:
            p = lambda f: glob(os.path.join(root, pattern.format(f)))[0]
            files = {
                (handle, desc): p(handle)
                for handle, desc in zip(file_handles, descriptors)
            }
        except:
            raise ValueError(
                "Error parsing filenames for quantifications. Glob pattern is: {}".format(
                    pattern
                )
            )

        self.load_quants(descriptors, files.values(), channel, exposure)

    def summarize(self):
        """
        Summarizes an AssaySeries as a Pandas DataFrame.

        Arguments:
            None

        Returns:
            (pd.DataFrame) summary of the AssaySeries

        """

        summaries = []
        for tc in self.assays.values():
            s = tc.merge_summarize()
            s["series_index"] = tc.description
            summaries.append(s)
        return pd.concat(summaries).sort_index()

    def process_quants(self, subset=None):
        """
        Processes the chip quantifications and saves summary images for each of, or a subset of,
        the assays.

        Arguments:
            (list | tuple) subset: list of assay descriptors (a subset of the assay dictionary keys)

        Returns:
            None

        """

        if not subset:
            subset = self.assays.keys()
        for key in tqdm(subset, desc="Mapping and Processing Buttons"):
            for quant in self.assays[key].quants:
                quant.process(reference=self.button_ref, mapped_features="button")
                quant.save_summary_image()

    def process_kinetics(self, subset=None, low_mem=True):
        """
        Processes the timecourses and saves summary images for each of, or a subset of,
        the assays.

        Arguments:
            (list | tuple) subset: list of assay descriptors (a subset of the assay dictionary keys)
            (bool) low_mem: flag to delete and garbage collect stamp data of all ChipImages
                after summarization and export

        Returns:
            None

        """

        if not subset:
            subset = self.assays.keys()
        for key in subset:
            s = self.assays[key].series
            s.process(self.chamber_ref)
            s.save_summary()
            s.save_summary_images(featuretype="chamber")
            if low_mem:
                s._delete_stamps()

    def save_summary(self, description=None, outPath=None):
        """
        Saves a CSV summary of the AssaySeries to the specified path.

        Arguments:
            (str) outPath: path of directory to save summary

        Returns:
            None

        """

        if not outPath:
            outPath = self.chamber_root
        df = self.summarize()
        if not description:
            fn = "{}_{}.csv.bz2".format(self.device.dname, "TitrationSeries_Analysis")
        else:   
            fn = "{}_{}_{}.csv.bz2".format(self.device.dname, description, "TitrationSeries_Analysis")
        df.to_csv(os.path.join(outPath, fn), compression="bz2")

    def __str__(self):
        return "Assays: {}, Device: {}, Attrs: {}".format(
            list(self.assays.keys()),
            str((self.device.setup, self.device.dname)),
            self.attrs,
        )


class ButtonChamberAssaySeries:
    # This class permits simultaneous kinetic imaging and analysis of chambers and buttons
    # It consists of a collection of kinetic imaging (one or more timecourses) in one or more channels

    def __init__(
        self,
        device,
        descriptions,
        chamber_ref,
        button_ref,
        channels,
        attrs=None,
        assays_attrs=[],
    ):
        """ """

        self.device = device
        self.channels = channels
        self.assays = OrderedDict(
            [
                ((description, channel), TurnoverAssay(device, description))
                for description in descriptions
                for channel in channels
            ]
        )
        self.chamber_ref = chamber_ref
        self.button_ref = button_ref
        self.root = None
        self.attrs = attrs
        logging.debug("AssaySeries Created | {}".format(self.__str__()))
        logging.debug(
            "AssaySeries Chamber Reference Set | {}".format(chamber_ref.__str__())
        )
        logging.debug(
            "AssaySeries Button Reference Set | {}".format(button_ref.__str__())
        )

    def parse_kineticsFolders(
        self, root, file_handles, descriptors, channel, exposure, pattern=None
    ):
        """
        Walks down directory tree, matches the passed file handles to the Timecourse descriptors,
        and loads kinetic imaging data. Default pattern is "*_{}*/*/StitchedImages", with {}
        file_handle

        Arguments:
            (str) root: path to directory Three levels above the StitchedImages folders (dir
                above unique assay folders)
            (list | tuple) file_handles: unique file handles to match to dirs in the root.
            (list | tuple) descriptors: unique kinetic imaging descriptors, order-matched to
                the file_handles
            (str) channel: imaging channel
            (int) exposure: exposure time (ms)
            (bool) pattern: custom UNIX-style pattern to match when parsing dirs

        Returns:
            None

        """

        self.root = root
        if not pattern:
            pattern = "*_{}*/{}/StitchedImages"

        p = lambda f: glob(os.path.join(root, pattern.format(f, channel)))[0]
        files = {
            (handle, desc, channel): p(handle)
            for handle, desc in zip(file_handles, descriptors)
        }

        self.load_kin(descriptors, files.values(), channel, exposure)

    def load_kin(self, descriptions, paths, channel, exposure):
        """
        Loads kinetic imaging and descriptions into the ButtonChamberAssaySeries.
            None

        """

        len_series = len(self.assays)
        len_descriptions = len(descriptions)

        if len_descriptions != len_series:
            raise ValueError(
                "Descriptions and series of different lengths. Number of assays and descriptions must match."
            )
        kin_refs = list(
            zip(descriptions, paths, [channel] * len_series, [exposure] * len_series)
        )
        for desc, p, chan, exp in kin_refs:
            t = Timecourse(self.device, desc)
            t.load_files(p, chan, exp)
            self.assays[(desc, chan)].series = t

    def process_kinetics(
        self,
        subset=None,
        featuretype="chamber",
        save_summary=True,
        save_images=True,
        low_mem=True,
    ):
        """
        Processes the timecourses and saves summary images for each of, or a subset of,
        the assays.

        Arguments:
            (list | tuple) subset: list of assay descriptors (a subset of the assay dictionary keys)
            (bool) low_mem: flag to delete and garbage collect stamp data of all ChipImages
                after summarization and export

        Returns:
            None

        """

        if not subset:
            subset = self.assays.keys()
        for key in subset:
            s = self.assays[key].series
            if featuretype == "chamber":
                try:
                    s.process(self.chamber_ref)
                except:
                    raise ValueError(
                        "No chamber ref provided (did you provice button ref instead?)"
                    )
            if featuretype == "button":
                try:
                    s.process(self.button_ref, featuretype='button')
                except:
                    raise ValueError(
                        "No button ref provided (did you provice chamber ref instead?)"
                    )
            if save_summary:
                s.save_summary()
            if save_images:
                s.save_summary_images(featuretype=featuretype)
            if low_mem:
                s._delete_stamps()

    def summarize(self):
        """
        Summarizes an ButtonChamberAssaySeries as a Pandas DataFrame.

        Arguments:
            None

        Returns:
            (pd.DataFrame) summary of the ButtonChamberAssaySeries

        """

        summaries = []
        for tc in self.assays.values():
            s = tc.merge_summarize()
            s["series_index"] = tc.description
            summaries.append(s)
        return pd.concat(summaries).sort_index()

    def save_summary(self, outPath=None):
        """
        Saves a CSV summary of the ButtonChamberAssaySeries to the specified path.

        Arguments:

        Returns:
            None

        """

        if not outPath:
            outPath = self.root
        df = self.summarize()
        fn = "{}_{}.csv.bz2".format(
            self.device.dname, "ButtonChamberAssaySeries_Analysis"
        )
        df.to_csv(os.path.join(outPath, fn), compression="bz2")

    def __str__(self):
        return "Assays: {}, ..., Device: {}, Channels: {}".format(
            list(self.assays.keys())[0], str((self.device.setup, self.device.dname)), str(self.channels)
        )

    def _repr_pretty_(self, p, cycle=True):
        p.text("<{}>".format(self.__str__()))


class ButtonBindingSeries:
    def __init__(
            self,
            device,
            button_ref,
            prey_channel: str,
            prey_exposure: int,
            bait_channel: str = '2',
            bait_exposure: int = 5
            ) -> None:
        """Initializes a ButtonBindingSeries object.

        Args:
            device: The imaging device used.
            button_ref: A reference object containing chip information.
            prey_channel (str): Channel used to detect prey.
            prey_exposure (int): Exposure time for prey imaging.
            bait_channel (str, optional): Channel used to detect bait. Defaults to '2'.
            bait_exposure (int, optional): Exposure time for bait imaging. Defaults to 5.
        
        Returns:
            None
        """
        self.device = device
        self.button_ref = button_ref
        self.prey_channel = prey_channel
        self.prey_exposure = prey_exposure 
        self.bait_channel = bait_channel
        self.bait_exposure = bait_exposure

    def grab_binding_images(self, binding_path: str, verbose: bool=True):
        """Grabs images from a directory structure for PreWash and PostWash conditions.

        Args:
            binding_path (str): Root directory containing image data.
            verbose (bool, optional): Whether to print paths to found images. Defaults to True.

        Returns:
            None
        """

        # utility function that globs images generated with a specified exposure, channel, etc
        def get_images(parent_path: str, exposure: int, channel: int, postwash: bool = True):
            wash_timing = 'PostWash' if postwash else 'PreWash'
            handle = '*{wash_timing}_Quant/{channel}/StitchedImages/BGSubtracted_StitchedImg_{exp}_{channel}_0.tif'.format(
                wash_timing=wash_timing,
                channel=channel, 
                exp=exposure
                )
            return glob(os.path.join(parent_path, handle))
        
        self.prewash_bait_images = get_images(binding_path, self.bait_exposure, self.bait_channel, postwash=False)
        self.postwash_bait_images = get_images(binding_path, self.bait_exposure, self.bait_channel, postwash=True)
        self.postwash_prey_images = get_images(binding_path, self.prey_exposure, self.prey_channel, postwash=True)
        self.binding_path = binding_path

        if verbose:
            print('PREWASH BAIT IMAGES:\n' + '\n'.join(self.prewash_bait_images) + '\n')
            print('POSTWASH BAIT IMAGES:\n' + '\n'.join(self.postwash_bait_images) + '\n')
            print('POSTWASH PREY IMAGES:\n' + '\n'.join(self.postwash_prey_images))

    def process(
            self, 
            concentration_parser: Optional[Callable[[str], float]] = None
            ) -> None:
        """Processes binding images to quantify signal across concentrations.

        Args:
            concentration_parser (callable, optional): Function to extract concentration from file path.
                If None, a default parser will be used.

        Returns:
            None

        Raises:
            ValueError: If the concentration cannot be extracted from the file name.
        """

        # default function for grabbing concentrations from filenames using regex  
        def concentration_parser_default(file: str):
            parent_path = self.binding_path

            # use regex to pull out note
            handle = os.path.relpath(file, parent_path).split('/')[0]
            note_match = re.search(r"\d{8}-\d{6}-d\d+_(.+?)_(PreWash|PostWash)_Quant", handle)
            if not note_match:
                raise ValueError(f"Could not extract note from file path: {file}")
            
            note = note_match.group(1)

            # then from note, grab the concentration
            concentration_match = re.search(r"([^\W_]+)(?=[a-z]M)", note)
            if not concentration_match:
                raise ValueError(f"Could not extract concentration from note: {note}")

            # convert numeric string to a float
            concentration = concentration_match.group(1)
            concentration = float(concentration.replace('_', '.'))

            return concentration

        concentration_parser = concentration_parser if concentration_parser else concentration_parser_default

        # utility function for processing binding images
        def process_binding_images(
                images: str,
                concentrations: list,
                channel: int, 
                exposure: int,
                reference_device, 
                reference_chip, 
                ):
            """Processes a set of images with known concentrations.

            Args:
                images (str): List of image file paths.
                concentrations (list): List of concentrations corresponding to each image.
                channel (int): Imaging channel.
                exposure (int): Exposure time.
                reference_device: Device object used for reference.
                reference_chip: Chip object used as a reference.

            Returns:
                pd.DataFrame: Concatenated data from all processed images.
            """
            data = []
            for f, c in tqdm(zip(images, concentrations), total=len(images)):
                chip_quant = ChipQuant(reference_device, 'ButtonReference')
                chip_quant.load_file(f, channel, exposure)
                chip_quant.process(reference=reference_chip)
                chip_quant.save_summary_image()

                _data = chip_quant.summarize()
                _data['concentration'] = [c] * len(_data)
                data.append(_data)

            data = pd.concat(data)
            return data

        print('Processing Pre-Wash Bait Images...')
        self.prewash_bait_data = process_binding_images(
            self.prewash_bait_images, 
            [concentration_parser(f) for f in self.prewash_bait_images],
            self.bait_channel,
            self.bait_exposure, 
            self.device,
            self.button_ref.chip
            )
        self.prewash_bait_data.sort_values(by=['x', 'y', 'concentration'], inplace=True)

        print('Processing Post-Wash Bait Images...')
        self.postwash_bait_data = process_binding_images(
            self.postwash_bait_images, 
            [concentration_parser(f) for f in self.postwash_bait_images],
            self.bait_channel,
            self.bait_exposure, 
            self.device,
            self.button_ref.chip
            )
        self.postwash_bait_data.sort_values(by=['x', 'y', 'concentration'], inplace=True)

        print('Processing Post-Wash Prey Images...')
        self.postwash_prey_data = process_binding_images(
            self.postwash_prey_images, 
            [concentration_parser(f) for f in self.postwash_prey_images],
            self.prey_channel,
            self.prey_exposure, 
            self.device,
            self.button_ref.chip
            )
        self.postwash_prey_data.sort_values(by=['x', 'y', 'concentration'], inplace=True)

    def save_summary(self, outpath: str, description: Optional[str] = None):
        """Saves a CSV summary of the binding data to the specified path.

        Args:
            outpath (str): Path to the directory where summary files should be saved.
            description (str, optional): Optional description to include in the filenames.

        Returns:
            None
        """
        if not description:
            fn = "{dname}_TitrationSeries_Analysis".format(dname=self.device.dname)
        else:   
            fn = "{dname}_{desc}_TitrationSeries_Analysis".format(dname=self.device.dname, desc=description)

        self.prewash_bait_data.to_csv(os.path.join(outpath, fn + "_prewash_bait.csv.bz2"), compression="bz2", index=True)
        self.postwash_bait_data.to_csv(os.path.join(outpath, fn + "_postwash_bait.csv.bz2"), compression="bz2", index=True)
        self.postwash_prey_data.to_csv(os.path.join(outpath, fn + "_postwash_prey.csv.bz2"), compression="bz2", index=True)


class ProcessingException(Exception):
    pass