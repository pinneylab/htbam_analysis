# title             : experiment.py
# description       :
# authors           : Daniel Mokhtari
# credits           : Craig Markin
# date              : 20180615
# version update    : 20180615
# version           : 0.1.0
# usage             : With permission from DM
# python_version    : 3.7


import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, List
from collections import namedtuple


def read_pinlist(pl):
    pl["Indices"] = pl.Indices.apply(eval)
    pl["x"] = pl.Indices.apply(lambda x: x[0])
    pl["y"] = pl.Indices.apply(lambda x: x[1])
    sorted_pinlist = pl.set_index(["x", "y"], drop=True, inplace=False).sort_index()
    return sorted_pinlist


def make_dummy_pinlist(block_descriptions: dict):

    def get_block(c):
        return ((c // 8) + 1)

    pinlist_dict = []
    for c in range(32):
        for r in range(56):
            block = get_block(c)
            mutant = block_descriptions[block]
            pinlist_dict.append({'Indices': (c + 1, r + 1), 'MutantID': mutant})
    pinlist = pd.DataFrame(pinlist_dict)
    pinlist['Indices'] = pinlist['Indices'].astype(str)

    pinlist = read_pinlist(pinlist)

    return pinlist


class Device:

    def __init__(self, setup: str, dname: str, dims: Tuple[int, int] = (32, 56)):
        self.setup = setup
        self.dname = dname
        self.dims = namedtuple("ChipDims", ["x", "y"])(*dims)
        self.pinlist = None

    def __str__(self):
        return "{}, {}, {}".format(self.setup, self.dname, self.dims)

    def set_pinlist(self, pinlist_path: Union[str, Path] = None, block_descriptions: dict = None):

        if pinlist_path:
            pinlist = read_pinlist(pd.read_csv(pinlist_path))

        if block_descriptions:
            pinlist = make_dummy_pinlist(block_descriptions)

        self.pinlist = pinlist

        return


class Experiment:

    def __init__(self, root: Union[str, Path]):

        root = Path(root) if not isinstance(root, Path) else root
        self.root = root
        self.devices = {}

        self.image_df = self._load_imaging_data()
        self.stitched_image_df = None
        

    def _load_imaging_data(self):
        
        path = self.root / 'imaging.csv'
        assert path.exists()

        image_df = pd.read_csv(path)
        image_df.sort_values(by=['image_path_parent', 'raster_col_index', 'raster_row_index'], inplace=True)

        return image_df
    

    def add_device(self, device: Device):
        self.devices[device.dname] = device


def series_to_dataframe(series_dict):
    """
    Recursively flattens a nested series dictionary into a Pandas DataFrame.
    """
    rows = []

    def walk(item):
        if item.get("item_type") == "image":
            # Create a flat dictionary: {'id': 'img_01', 'metadat_col1': val, ...}
            row = {"identifier": item["identifier"]}
            row.update(item.get("metadata", {}))
            rows.append(row)
        
        elif item.get("item_type") == "series":
            # If it's a series, recurse into its items
            for sub_item in item.get("items", []):
                walk(sub_item)

    # Start the recursion
    walk(series_dict)
    
    return pd.DataFrame(rows)


# def df_to_nested_series(df, group_cols, series_id="root"):
#     """
#     Converts a flat DataFrame into a nested dictionary structure.
    
#     Args:
#         df: The pandas DataFrame.
#         group_cols: A list of column names to group by (e.g., ['replicate', 'channel']).
#         series_id: The ID for the current series level.
#     """
#     # Base Case: No more groups to create, these are individual images
#     if not group_cols:
#         items = []
#         for _, row in df.iterrows():
#             # Separate the 'id' from the rest of the metadata
#             img_id = str(row['id'])
#             # Convert row to dict and remove 'id' so it only contains metadata
#             metadata = row.drop('id').to_dict()
            
#             items.append({
#                 "id": img_id,
#                 "type": "image",
#                 "metadata": metadata
#             })
#         return items

#     # Recursive Case: Group by the first column in the list
#     current_group_col = group_cols[0]
#     remaining_cols = group_cols[1:]
    
#     nested_items = []
#     for group_val, group_df in df.groupby(current_group_col):
#         # The ID for the sub-series is the value of the grouping column
#         sub_series_id = str(group_val)
        
#         nested_items.append({
#             "id": sub_series_id,
#             "type": "series",
#             "items": df_to_nested_series(group_df, remaining_cols, sub_series_id)
#         })
        
#     # If this is the top-level call, wrap it in a root series dict
#     if series_id == "root":
#         return {"id": "root", "type": "series", "items": nested_items}
    
#     return nested_items


class DataHandler:
    
    def __init__(
            self, 
            root: Union[str, Path]):

        self._root = Path(root) if isinstance(root, str) else root
        assert self._root.exists(), 'root path: {} does not exist'.format(self._root)

        self._raw_images = self._root / 'raw_images'
        assert self._raw_images.exists(), 'raw_images: {} does not exist.'.format(self._raw_images)

        self._bsgub_images = root / 'bgsub_images'
        assert self._bsgub_images.exists(), 'bgsub_images: {} does not exist.'.format(self._bgsub_images)

        self.image_metadata = self.load_image_metadata(self._bsgub_images / 'bgsub_images.csv')
        self.series_index_metadata = self.load_series_metadata(self._root / 'series_index.json')

        print("Loaded the following image sets:")
        series_mask = np.zeros(len(self.image_metadata), dtype=bool)
        for key in self.series_index_metadata:
            print(key)
            mask = self.image_metadata['image_path'].apply(str).str.contains(key)
            series_mask = series_mask + mask

        print("\nLoaded the following images:")
        for image in self.image_metadata[~series_mask]['image_identifier']:
            print(image)

    def load_image_metadata(self, image_metadata_path: Union[str, Path]):
        image_metadata =  pd.read_csv(image_metadata_path)
        image_metadata['image_path'] = image_metadata['image_path'].apply(lambda f: self._bsgub_images / Path(f))
        image_metadata['image_identifier'] = image_metadata['image_path'].apply(lambda f: f.with_suffix('').name)
        return image_metadata

    def load_series_metadata(self, series_metadata_path: Union[str, Path]):

        with open(series_metadata_path, 'r') as f:
            series_index = json.load(f)

        series_metadata = {}
        for series in series_index:
            identifier = series['identifier']
            df = series_to_dataframe(series)

            # TODO: make this less hacky
            # df['identifier'] = df['identifier'].apply(lambda f: self._bsgub_images / Path(f).with_suffix('.tif'))
            df['identifier'] = df['identifier'].apply(lambda f: self._bsgub_images / Path(*Path(f).parts[1:]).with_suffix('.tif'))

            series_metadata[identifier] = df
            
        return series_metadata

    def get_images(self, identifiers: Union[List[str], str]):

        if not isinstance(identifiers, list):
            identifiers = [identifiers]

        image_data = []

        for identifier in identifiers:
            if identifier in self.series_index_metadata.keys():
                image_data.append(self._get_images_by_series_id(identifier))

            elif identifier in self.image_metadata['image_identifier'].tolist():
                image_data.append(self._get_images_by_image_id(identifier))

            else:
                print('{} not found.'.format(identifier))

        return pd.concat(image_data)

    def _get_images_by_series_id(self, series_identifier: str):

        series_df = self.series_index_metadata[series_identifier].copy()
        series_df.sort_values(by='identifier', inplace=True)
        series_mask = self.image_metadata['image_path'].isin(series_df['identifier'])
        
        image_df = self.image_metadata[series_mask].copy()
        
        image_df.sort_values(by='image_path', inplace=True)

        merged = pd.merge(series_df, image_df, left_on='identifier', right_on='image_path')

        return merged

    def _get_images_by_image_id(self, image_identifier: str):
        mask = self.image_metadata['image_identifier'] == image_identifier
        image_df = self.image_metadata.copy()[mask]        
        return image_df
    