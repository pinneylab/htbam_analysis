import pandas as pd
import numpy as np
import warnings
import unittest

from htbam_analysis.processing import chip
from htbam_analysis.processing import experiment as exp
from htbam_analysis.processing import chipcollections as collections
chip.Stamp.chamberrad = 32


def make_reference_chip(
        reference_image: str, 
        channel: str,
        exposure: int,
        corners: tuple, 
        feature: str,
        coerce_center: bool,
        pinlist,
        setup_num: str, 
        device_num: str, 
        device_dimensions: tuple,
        save_summary: bool
        ):

    device = exp.Device(setup_num, device_num, device_dimensions, pinlist, corners)
    reference = collections.ChipQuant(device, 'Chamber_Ref')
    reference.load_file(reference_image, channel, exposure)

    if feature == 'chamber':
        reference.process(mapped_features = 'chamber', coerce_center = coerce_center)
        if save_summary:
            reference.save_summary_image(feature_type = 'chamber')
    else:
        reference.process(coerce_center = coerce_center)
        if save_summary:
            reference.save_summary_image()

    return device, reference


class TestIO(unittest.TestCase):

    def setUp(self) -> None:
        self.chamber_rfu_test = pd.read_csv('./test/test_data/chamber_quant/chamber_quant.csv')['sum_chamber'].to_numpy()
        self.button_rfu_test = pd.read_csv('./test/test_data/button_quant/button_quant.csv')['summed_button_BGsub'].to_numpy()

        root = './test/test_data/'
        e = exp.Experiment('test', root, 'operator')
        self.dummy_pinlist = e.read_pinlist('{}/dummy_pinlist.csv'.format(root))

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*`np\.bool` is a deprecated alias.*"
        )

    def test_button_finding(self):

        button_quant_corners = ((447, 383)), ((6724, 467)), ((358, 6760)), ((6632, 6840))
        _, button_quant_chip = make_reference_chip(
            './test/test_data/button_quant/20250207-011718-d3_0_4_ADPR_100AF_PostWash_Quant/3/StitchedImages/BGSubtracted_StitchedImg_5_3_0.tif',
            channel='3',
            exposure='5',
            corners=button_quant_corners,
            feature='button',
            coerce_center=False,
            pinlist=self.dummy_pinlist,
            setup_num='s1',
            device_num='d3',
            device_dimensions=(32, 56),
            save_summary=False
        )

        button_rfu = button_quant_chip.summarize()['summed_button_BGsub'].to_numpy()
        normalized_difference = np.abs(self.button_rfu_test - button_rfu) / self.button_rfu_test
        self.assertLess(
            (normalized_difference > 0.05).sum(),
            1
        )
    
    def test_chamber_finding(self):

        chamber_quant_corners = ((397, 476), (6667, 435), (445, 6853), (6711, 6794))
        _, chamber_quant_chip = make_reference_chip(
            './test/test_data/chamber_quant/20250428_133901_15uM_NSP89/2/StitchedImages/BGSubtracted_StitchedImg_10_2_832.tif',
            channel='2',
            exposure='10',
            corners=chamber_quant_corners,
            feature='chamber',
            coerce_center=False,
            pinlist=self.dummy_pinlist,
            setup_num='s1',
            device_num='d3',
            device_dimensions=(32, 56),
            save_summary=False
        )

        chamber_rfu = chamber_quant_chip.summarize()['sum_chamber'].to_numpy()
        normalized_difference = np.abs(self.chamber_rfu_test - chamber_rfu) / self.chamber_rfu_test
        self.assertLess(
            (normalized_difference > 0.05).sum(),
            1
        )


if __name__ == '__main__':
    unittest.main()