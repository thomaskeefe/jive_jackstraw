from unittest import TestCase
import numpy as np
import jive_jackstraw.jive_jackstraw

datablock = np.loadtxt("test_datablock.csv", delimiter=',')
cns = np.loadtxt("test_cns.csv", delimiter=',')

class TestJIVEJackstraw(TestCase):
    def test_OLS_F_stat(self):
        "Test that the custom OLS function matches statsmodels"
        test_f_stat = jive_jackstraw.jive_jackstraw.OLS_F_stat(datablock[:,0], datablock[:,1])
        reference_value = 1.1170073514191354 # computed with statsmodels
        self.assertAlmostEqual(test_f_stat, reference_value)
