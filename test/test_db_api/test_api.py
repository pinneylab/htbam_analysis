import unittest
from htbam_analysis.db_api import htbam_db_api
from htbam_analysis.db_api.units import units


class TestIO(unittest.TestCase):

    def setUp(self) -> None:
        self.db_api = htbam_db_api.LocalHtbamDBAPI(
            standard_curve_data_path="./test/test_db_api/test_data/mpro_standard_test.csv",
            standard_name="Mpro_std",
            standard_substrate="IDK",
            standard_units=units.uM,
            kinetic_data_path="./test/test_db_api/test_data/mpro_kinetic_test.csv",
            kinetic_name="Mpro_kin",
            kinetic_substrate="N4L",
            kinetic_units=units.uM,
            time_units=units.s,
            )
    
    def test_init(self):
        self.assertListEqual(['chamber_metadata', 'runs', 'button_quant'], list(self.db_api._json_dict.keys()))
        self.assertListEqual(['standard_0', 'kinetic_0'], list(self.db_api._json_dict['runs'].keys()))
   
    def test_get_run_assay_data(self):
        run_data = self.db_api.get_run_assay_data('standard_0')
        self.assertEqual(len(run_data), 4)
        chamber_idxs, luminance_data, conc_data, time_data = run_data
        self.assertListEqual(chamber_idxs.tolist(),["1,1", "1,2", "1,3"])
        self.assertEqual(conc_data.shape, (7,))
        self.assertEqual(time_data.shape, (1,7))
        self.assertEqual(luminance_data.shape, (1, len(chamber_idxs), len(conc_data)))

    def test_chamber_name_dicts(self):
        self.assertDictEqual({'1,1': 1224, '1,2': 1217, '1,3': 1329}, self.db_api.get_chamber_name_dict())
        self.assertDictEqual({1224: ['1,1'], 1217: ['1,2'], 1329: ['1,3']}, self.db_api.get_chamber_name_to_id_dict())
        
    

if __name__ == '__main__':
    unittest.main()