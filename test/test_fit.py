import unittest
import numpy as np
from htbam_analysis.analysis.fit import (
    mm_model,
    inhibition_model,
    fit_concentration_vs_time,
    fit_luminance_vs_concentration,
    fit_initial_rates_vs_concentration_with_function
)
from htbam_analysis.db_api.data import Data4D, Data3D, Data2D, IndepVars, Meta
from htbam_analysis.db_api.units.units import units


class TestFitModels(unittest.TestCase):

    def test_mm_model(self):
        # mm_model(x, v_max, K_m) = v_max * x / (K_m + x)
        self.assertEqual(mm_model(0.0, 10.0, 2.0), 0.0)
        self.assertAlmostEqual(mm_model(2.0, 10.0, 2.0), 5.0)
        self.assertAlmostEqual(mm_model(1e6, 10.0, 2.0), 10.0, places=3)

    def test_inhibition_model(self):
        # inhibition_model(x, r_max, r_min, ic50) = r_min + (r_max - r_min) / (1 + (x / ic50))
        self.assertAlmostEqual(inhibition_model(0.0, 1.0, 0.1, 5.0), 1.0)
        self.assertAlmostEqual(inhibition_model(5.0, 1.0, 0.1, 5.0), 0.1 + 0.9 / 2)
        self.assertAlmostEqual(inhibition_model(1e6, 1.0, 0.1, 5.0), 0.1, places=3)


class TestFitFunctions(unittest.TestCase):

    def setUp(self):
        self.n_conc = 5
        self.n_time = 4
        self.n_chamb = 3
        self.chambers = np.array([f"1,{i}" for i in range(1, self.n_chamb + 1)])
        self.samples = np.array([f"sample_{i}" for i in range(1, self.n_chamb + 1)])
        self.concs = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    def test_fit_concentration_vs_time(self):
        # linear relation along time axis: y = 2.0 * t + 1.0
        time_vals = np.array([[0.0, 10.0, 20.0, 30.0]] * self.n_conc) # (n_conc, n_time)
        
        # dep_var shape (n_conc, n_time, n_chamb, n_vars)
        dep_var = np.zeros((self.n_conc, self.n_time, self.n_chamb, 1))
        for c in range(self.n_conc):
            for t in range(self.n_time):
                dep_var[c, t, :, 0] = 2.0 * time_vals[c, t] + 1.0
        
        indep_vars = IndepVars(
            concentration=self.concs * units.uM,
            chamber_IDs=self.chambers,
            sample_IDs=self.samples,
            button_quant_sum=np.ones(self.n_chamb) * 100.0 * units.RFU,
            time=time_vals * units.s
        )
        
        data_4d = Data4D(
            indep_vars=indep_vars,
            dep_var=dep_var,
            dep_var_type=["concentration"],
            dep_var_units=[units.uM],
            meta=Meta()
        )
        
        # Run fit
        fit_results, fit_mask = fit_concentration_vs_time(data_4d, max_reaction_percent=100)
        
        self.assertIsInstance(fit_results, Data3D)
        self.assertIsInstance(fit_mask, Data4D)
        self.assertListEqual(["slope", "intercept", "r_squared"], fit_results.dep_var_type)
        
        # Slopes should be 2.0
        np.testing.assert_allclose(fit_results.dep_var[..., 0], 2.0, rtol=1e-5)
        # Intercepts should be 1.0
        np.testing.assert_allclose(fit_results.dep_var[..., 1], 1.0, rtol=1e-5)
        # R2 should be 1.0
        np.testing.assert_allclose(fit_results.dep_var[..., 2], 1.0, rtol=1e-5)

    def test_fit_luminance_vs_concentration(self):
        # linear relation along conc axis: y = 5.0 * conc + 10.0
        time_vals = np.array([[0.0, 10.0, 20.0, 30.0]] * self.n_conc)
        dep_var = np.zeros((self.n_conc, self.n_time, self.n_chamb, 1))
        
        for c in range(self.n_conc):
            dep_var[c, :, :, 0] = 5.0 * self.concs[c] + 10.0
            
        indep_vars = IndepVars(
            concentration=self.concs * units.uM,
            chamber_IDs=self.chambers,
            sample_IDs=self.samples,
            button_quant_sum=np.ones(self.n_chamb) * 100.0 * units.RFU,
            time=time_vals * units.s
        )
        
        data_4d = Data4D(
            indep_vars=indep_vars,
            dep_var=dep_var,
            dep_var_type=["luminance"],
            dep_var_units=[units.RFU],
            meta=Meta()
        )
        
        fit_results = fit_luminance_vs_concentration(data_4d, timepoint=-1)
        
        self.assertIsInstance(fit_results, Data2D)
        self.assertListEqual(["slope", "intercept", "r_squared"], fit_results.dep_var_type)
        
        np.testing.assert_allclose(fit_results.dep_var[..., 0], 5.0, rtol=1e-5)
        np.testing.assert_allclose(fit_results.dep_var[..., 1], 10.0, rtol=1e-5)
        np.testing.assert_allclose(fit_results.dep_var[..., 2], 1.0, rtol=1e-5)

    def test_fit_initial_rates_vs_concentration_with_function(self):
        # Generate initial rates according to MM model: v_max = 10.0, K_m = 4.0
        mm_rates = mm_model(self.concs, 10.0, 4.0) # shape (5,)
        
        # dep_var shape (n_conc, n_chamb, n_vars)
        dep_var = np.zeros((self.n_conc, self.n_chamb, 1))
        for i in range(self.n_chamb):
            dep_var[:, i, 0] = mm_rates
            
        indep_vars = IndepVars(
            concentration=self.concs * units.uM,
            chamber_IDs=self.chambers,
            sample_IDs=self.samples,
            button_quant_sum=np.ones(self.n_chamb) * 100.0 * units.RFU,
            time=np.zeros((self.n_conc, self.n_time)) * units.s
        )
        
        data_3d = Data3D(
            indep_vars=indep_vars,
            dep_var=dep_var,
            dep_var_type=["slope"],
            dep_var_units=[units.uM / units.s],
            meta=Meta()
        )
        
        params_data, ypred_data = fit_initial_rates_vs_concentration_with_function(
            data_3d,
            mm_model,
            p0=[1.0, 1.0]
        )
        
        self.assertIsInstance(params_data, Data2D)
        self.assertIsInstance(ypred_data, Data3D)
        
        # Fitted parameters should be very close to v_max=10.0 and K_m=4.0
        np.testing.assert_allclose(params_data.dep_var[..., 0], 10.0, rtol=1e-2)
        np.testing.assert_allclose(params_data.dep_var[..., 1], 4.0, rtol=1e-2)
        # R2 should be very close to 1.0
        np.testing.assert_allclose(params_data.dep_var[..., 2], 1.0, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
