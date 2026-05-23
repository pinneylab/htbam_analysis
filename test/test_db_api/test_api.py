import unittest
import numpy as np
from htbam_analysis.db_api import htbam_db_api
from htbam_analysis.db_api.units import units
from htbam_analysis.db_api.data import Data4D, Data2D


class TestLocalHtbamDBAPI(unittest.TestCase):

    def setUp(self) -> None:
        self.db_api = htbam_db_api.LocalHtbamDBAPI(
            standard_curve_data_path="./test/test_db_api/test_data/mpro_standard_test.csv",
            standard_name="Mpro_std",
            standard_substrate="IDK",
            standard_units=units.uM,
            standard_concentration_col="concentration_uM",
            kinetic_data_path="./test/test_db_api/test_data/mpro_kinetic_test.csv",
            kinetic_name="Mpro_kin",
            kinetic_substrate="N4L",
            kinetic_units=units.uM,
            kinetic_concentration_col="series_index",
            time_units=units.s,
            button_quant_data_path="./test/test_data/button_quant/button_quant.csv",
            )
    
    def test_get_run_names(self):
        run_names = self.db_api.get_run_names()
        self.assertListEqual(['Mpro_std', 'Mpro_kin', 'button_quant'], list(run_names))

    def test_get_run_standard(self):
        std_run = self.db_api.get_run('Mpro_std')
        self.assertIsInstance(std_run, Data4D)
        self.assertListEqual(['luminance'], std_run.dep_var_type)
        
        # Check chambers
        chambers = std_run.indep_vars.chamber_IDs
        self.assertIn("1,1", chambers)
        self.assertIn("1,2", chambers)
        self.assertIn("1,3", chambers)

        # Check concentrations
        concs = std_run.indep_vars.concentration.magnitude
        np.testing.assert_allclose(concs, [0.0, 0.76, 1.53, 3.06, 6.13, 12.5, 25.0])

    def test_get_run_kinetics(self):
        kin_run = self.db_api.get_run('Mpro_kin')
        self.assertIsInstance(kin_run, Data4D)
        self.assertListEqual(['luminance'], kin_run.dep_var_type)
        
        chambers = kin_run.indep_vars.chamber_IDs
        self.assertIn("1,1", chambers)
        
        # Check that we parsed raw concentration strings into floats: e.g. 0.01, 0.81, 3.27 etc.
        concs = kin_run.indep_vars.concentration.magnitude
        self.assertGreater(len(concs), 0)
        self.assertIn(1.01, np.round(concs, 2))

    def test_get_run_button_quant(self):
        bq_run = self.db_api.get_run('button_quant')
        self.assertIsInstance(bq_run, Data2D)
        self.assertListEqual(['luminance'], bq_run.dep_var_type)


if __name__ == '__main__':
    unittest.main()