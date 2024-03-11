# -*- coding: utf-8 -*-
"""
Example showing how to simulate a Simulink model (called the_model) with different
parameter and external input signal values using the MATLAB Engine API for Python.

The example requires:
    1. MATLAB and Simulink products installed and licensed
    2. MATLAB Engine API installed as a Python package
       https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

"""
import numpy as np
import matlab.engine

mle = matlab.engine.start_matlab()  # start the matlab engine

# Allocate res list to hold the results from 2 calls to sim_the_model
res = [0]*2
# 1st sim: with default parameter values: Mb = 1200 Kg
res[0] = mle.sim_the_model('StopTime', 30)

# 2nd sim: with new values for tunable parameters
tunableParams = {
    # use a new parameter for body mass: Mb = 5000 Kg
    'Mb': 5000.0
}
res[1] = mle.sim_the_model('StopTime', 30, 'TunableParameters', tunableParams)

# callback into MATLAB to plot the results
mle.plot_results(res, "Results from sim_the_model using MATLAB Engine")

input("Press enter to close the MATLAB figure and exit ...")
mle.quit()  # stop the matlab engine
