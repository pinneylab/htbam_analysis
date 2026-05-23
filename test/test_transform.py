import unittest
import numpy as np
from htbam_analysis.analysis.transform import transform_data
from htbam_analysis.db_api.data import Data2D, IndepVars, Meta
from htbam_analysis.db_api.units.units import units


class TestTransformData(unittest.TestCase):

    def setUp(self):
        # IndepVars for a 3-chamber system
        self.chambers = np.array(["1,1", "1,2", "1,3"])
        # sample_1 has two replicates, sample_2 has one
        self.samples = np.array(["sample_1", "sample_1", "sample_2"])
        
        self.indep_vars = IndepVars(
            concentration=np.array([]) * units.uM,
            chamber_IDs=self.chambers,
            sample_IDs=self.samples,
            button_quant_sum=np.zeros(3) * units.RFU,
            time=np.zeros((0, 0)) * units.s
        )

    def test_basic_subtraction(self):
        a = Data2D(
            indep_vars=self.indep_vars,
            dep_var=np.array([[100.0], [200.0], [300.0]]),
            dep_var_type=["luminance"],
            dep_var_units=[units.RFU],
            meta=Meta()
        )
        b = Data2D(
            indep_vars=self.indep_vars,
            dep_var=np.array([[10.0], [30.0], [50.0]]),
            dep_var_type=["luminance"],
            dep_var_units=[units.RFU],
            meta=Meta()
        )
        
        # Test basic subtraction
        res = transform_data([a, b], "(a_luminance - b_luminance)", "diff")
        
        self.assertIsInstance(res, Data2D)
        self.assertListEqual(["diff"], res.dep_var_type)
        self.assertEqual(res.dep_var_units[0], units.RFU)
        np.testing.assert_allclose(res.dep_var[..., 0], [90.0, 170.0, 250.0])

    def test_unit_simplification(self):
        a = Data2D(
            indep_vars=self.indep_vars,
            dep_var=np.array([[100.0], [200.0], [300.0]]),
            dep_var_type=["luminance"],
            dep_var_units=[units.RFU],
            meta=Meta()
        )
        
        # slope unit is RFU / uM
        slope = 10.0 * (units.RFU / units.uM)
        
        res = transform_data(
            [a],
            "a_luminance / slope",
            "concentration",
            expression_vars={"slope": slope}
        )
        
        self.assertListEqual(["concentration"], res.dep_var_type)
        # Unit should simplify to uM
        self.assertEqual(res.dep_var_units[0], units.uM)
        np.testing.assert_allclose(res.dep_var[..., 0], [10.0, 20.0, 30.0])

    def test_proxy_dot_notation(self):
        a = Data2D(
            indep_vars=self.indep_vars,
            dep_var=np.array([[10.0], [20.0], [100.0]]),
            dep_var_type=["luminance"],
            dep_var_units=[units.RFU],
            meta=Meta()
        )
        
        # "a.luminance" should be equivalent to "a_luminance"
        res = transform_data([a], "a.luminance * 2", "double_lum")
        np.testing.assert_allclose(res.dep_var[..., 0], [20.0, 40.0, 200.0])

    def test_sample_grouped_reductions(self):
        a = Data2D(
            indep_vars=self.indep_vars,
            dep_var=np.array([[10.0], [20.0], [100.0]]),
            dep_var_type=["luminance"],
            dep_var_units=[units.RFU],
            meta=Meta()
        )
        
        # Test mean of sample-level replicates
        # sample_1 has values 10.0 and 20.0 -> mean is 15.0
        # sample_2 has value 100.0 -> mean is 100.0
        res = transform_data([a], "np.mean(a.sample.luminance)", "mean_lum")
        
        np.testing.assert_allclose(res.dep_var[..., 0], [15.0, 15.0, 100.0])

    def test_device_reductions(self):
        a = Data2D(
            indep_vars=self.indep_vars,
            dep_var=np.array([[10.0], [20.0], [30.0]]),
            dep_var_type=["luminance"],
            dep_var_units=[units.RFU],
            meta=Meta()
        )
        
        # np.mean(a.device.luminance) should calculate the mean across the whole device:
        # (10 + 20 + 30) / 3 = 20.0, and map it back to all 3 chambers
        res = transform_data([a], "np.mean(a.device.luminance)", "device_mean")
        
        np.testing.assert_allclose(res.dep_var[..., 0], [20.0, 20.0, 20.0])

    def test_keep_existing(self):
        a = Data2D(
            indep_vars=self.indep_vars,
            dep_var=np.array([[10.0], [20.0], [30.0]]),
            dep_var_type=["luminance"],
            dep_var_units=[units.RFU],
            meta=Meta()
        )
        
        res = transform_data(
            [a],
            "a_luminance * 10",
            "luminance_x10",
            keep_existing=True
        )
        
        self.assertListEqual(["luminance", "luminance_x10"], res.dep_var_type)
        self.assertEqual(res.dep_var_units[0], units.RFU)
        self.assertEqual(res.dep_var_units[1], units.RFU)
        
        np.testing.assert_allclose(res.dep_var[..., 0], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(res.dep_var[..., 1], [100.0, 200.0, 300.0])


if __name__ == '__main__':
    unittest.main()
