# title             : experiment.py
# description       :
# authors           : Daniel Mokhtari
# credits           : Craig Markin
# date              : 20180615
# version update    : 20180615
# version           : 0.1.0
# usage             : With permission from DM
# python_version    : 3.7


# General Python
import os
from pathlib import Path
import logging
from collections import namedtuple

# Scientific Data Structures and Plotting
import pandas as pd


class Experiment:
    def __init__(self, description, root, operator, repoName="Repo"):
        """
        Constructor for the Experiment class.

        Arguments:
            (str) description: User-defined xperimental description
            (str) root: Root path of the experimental data
            (str) operator: Workup user name or initials
            (str) repoName: Name of the stamp repo

        Returns:
            None

        """
        self.info = description
        self.root = root
        self.devices = []
        self.operator = operator
        self.repoName = repoName
        self.repoRoot = Path(os.path.join(root, repoName))
        self._initializeLogger()
        logging.info("Experiment Initialized | {}".format(self.__str__()))

    def _initializeLogger(self):
        """
        Initializes the logger, which logs to both a file (>DEBUG) and the console (>INFO)

        Arguments:
            None

        Return:
            None

        """

        logFile = os.path.join(self.root, "Workup.log")
        # verbose logging to file
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)-4s %(name)-4s %(levelname)-4s %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            filename=logFile,
            filemode="a+",
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # simpler console logging
        formatter = logging.Formatter(
            "%(levelname)-8s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
        )
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

        # chiplogger = logging.getLogger('experiment.chip') # FUTURE

    def addDevices(self, devices):
        """

        Arguments:
            (list | tuple) devices: list or tuple of Device objects to add to the experiment

        Returns:
            None

        """

        def add(d):
            if isinstance(d, Device):
                self.devices.append(d)
            else:
                raise ValueError("Must add an experimental Device object")

        if isinstance(devices, tuple) or isinstance(devices, list):
            for d in devices:
                if any([d == device for device in self.devices]):
                    logging.warn(
                        "Device already added |  Device: {}".format(d.__str__())
                    )
                else:
                    add(d)
                    logging.info("Added Device | Device: {}".format(d.__str__()))
        else:
            raise ValueError("Must add devices as a list or tuple")

    @staticmethod
    def read_pinlist(pinlistPath):
        pl = pd.read_csv(pinlistPath)
        pl["Indices"] = pl.Indices.apply(eval)
        pl["x"] = pl.Indices.apply(lambda x: x[0])
        pl["y"] = pl.Indices.apply(lambda x: x[1])
        sorted_pinlist = pl.set_index(["x", "y"], drop=True, inplace=False).sort_index()
        return sorted_pinlist

    def __str__(self):
        return "Description: {}, Operator: {}".format(self.info, self.operator)


class Device:
    def __init__(
        self, setup, dname, dims, pinlist, corners, operators="FordyceLab", attrs=None
    ):
        """
        Constructor for the Device class.

        Arguments:
            (str) setup:
            (str) dname:
            (tuple) dims:
            (pd.DataFrame) pinlist: Pinlist indexed by (x, y) chamber inddices, and ID column "MutantID"
            (tuple) corners: nested tuple of chip corner positions of the form
                ((ULx, ULy),(URx, URy),(LLx, LLy),(LRx, LRy))
            (str) operators: Name(s) of device operators
            (attrs) dict: arbitrary device metdata

        Returns:
            None

        """

        self.setup = setup
        self.dname = dname
        self.dims = namedtuple("ChipDims", ["x", "y"])(*dims)
        self.pinlist = pinlist
        self.operators = operators
        self.attrs = attrs  # arbitrary metadata, as a dict
        self.experiments = None
        self.corners = Device._corners(corners)

    @staticmethod
    def _corners(corners):
        """
        Generates a corners namedtuple from a set of cornerpositions.

        Arguments:
            (tuple) corners: corner positions of the form ((ULx, ULy),(URx, URy),(LLx, LLy),(LRx, LRy))

        Returns:
            (namedtuple) a namedtuple for the cornerpositions

        """
        chipCorners = namedtuple("Corners", ["ul", "ur", "bl", "br"])
        return chipCorners(*corners)

    def __str__(self):
        return "{}, {}, {}".format(self.operators, self.setup, self.dname)

    def __eq__(self, other):
        if isinstance(other, Device):
            return (
                (self.setup == other.setup)
                and (self.dname == other.dname)
                and (self._corners == other._corners)
            )
        else:
            return NotImplemented

    def _repr_pretty_(self, p, cycle=True):
        p.text("<{}>".format(self.__str__()))
