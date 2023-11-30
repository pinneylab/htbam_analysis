import pathlib
import os
from typing import Tuple


class RasterParams:
    def __init__(
        self,
        overlap: float,
        size: int,
        acqui_ori: Tuple[int, int],
        rotation: float,
        auto_ff: bool = False,
        ff_type: str = "BaSiC",
        group_feature=0,
    ):
        """
        Parameters describing a single image raster.

        Arguments:
            (float) overlap: overlap fraction (e.g., 0.1)
            (float) rotation: pre-stitch rotation to perform (%)
            (bool) auto_ff: flag to execute FF correction on stitch, if possible
            (int | float | string) group_feature: feature value for RasterGroup

        Returns:
            None

        """
        # This can never be reached!
        # if self._root:
        #     self._parent = list(pathlib.Path(root).parents)[0]
        self._size = size
        self._overlap = overlap
        self._rotation = rotation
        self._acqui_ori = acqui_ori
        self._group_feature = group_feature
        self._auto_ff = auto_ff
        self._ff_type = ff_type

        self._exposure = None
        self._channel = None
        self._parent = None
        self._dims = None
        self._root = None

    def update_root(self, new_root):
        self._root = new_root
        self._parent = os.path.dirname(new_root)

    @property
    def size(self):
        return self._size

    @property
    def overlap(self):
        return self._overlap

    @property
    def rotation(self):
        return self._rotation

    @property
    def acqui_ori(self):
        return self._acqui_ori

    @property
    def group_feature(self):
        return self._group_feature

    @property
    def auto_ff(self):
        return self._auto_ff

    @property
    def ff_type(self):
        return self._ff_type

    @property
    def exposure(self):
        return self._exposure

    @property
    def channel(self):
        return self._channel

    @property
    def parent(self):
        return self._parent

    @property
    def dims(self):
        return self._dims

    @property
    def root(self):
        return self._root

    def update_channel(self, new_channel):
        self._channel = new_channel

    def update_exposure(self, new_exposure):
        self._exposure = new_exposure

    def update_dims(self, new_dims):
        self._dims = new_dims

    def update_group_feature(self, new_group_feature):
        self._group_feature = new_group_feature
