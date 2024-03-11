# -*- coding: utf-8 -*-
"""
Example showing how call a python package to simulate a Simulink
model (called the_model) with different tunable parameter and
external input signal values.

To run this script, you need the following prerequisites:
    1. Install the MATLAB Runtime (R2021b or later) 
       https://www.mathworks.com/products/compiler/matlab-runtime.html

    2. Install "sim_the_model" python package. See instructions below.
    
Notes:
1. Run build_python_package_around_sim_the_model.m script in MATLAB
   (R2021b or later) to create sim_the_model_python_package. This
   script requires the following products:
    - MATLAB
    - Simulink
    - MATLAB Compiler
    - Simulink Compiler
    - MATLAB Compiler SDK

   After your run the script, follow the instructions displayed on the
   MATLAB command window to install the sim_the_model_python_package

2. sim_the_model_python_package is a wrapper to run sim_the_model.m
   MATLAB function using deployed version of the Simulink model
   (the_model) and the MATLAB Runtime

3. Both MATLAB Runtime, and sim_the_model_python_package (once it is
   built) can be distributed freely and do not require licenses.
"""

import numpy as np
import matlab
import sim_the_model2
import matplotlib.pyplot as plt

# Specify the path to sim_the_model_python_package.
import sys
sys.path.append(".\\sim_the_model2_python_package\\Lib\\site-packages")

# initialize sim_the_model package
mlr = sim_the_model2.initialize()

# Allocate res list to hold the results from 2 calls to sim_the_model
res = [0]*2
# 1st sim: with default parameter values: Mb = 1200 Kg
res[0] = mlr.sim_the_model2('StopTime', 30)

# 2nd sim: with new values for tunable parameters
tunableParams = {
    'Mb': 5000.0   # use a new parameter for body mass Kg
}
res[1] = mlr.sim_the_model2('StopTime', 30, 'TunableParameters', tunableParams)


# Plot the results
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(res[0]['vertical_disp']['Time'], res[0]['vertical_disp']['Data'], color=cols[0],
        label="vertical displacement: 1st sim with default body mass Mb")
ax.plot(res[1]['vertical_disp']['Time'], res[1]['vertical_disp']['Data'],
        color=cols[1], label="vertical displacement: 2nd sim with updated body mass Mb ")


ax.grid()
lg = ax.legend(fontsize='x-small')
lg.set_draggable(True)
ax.set_title("Results from sim_the_model using MATLAB Runtime")
plt.show()

mlr.terminate()  # stop the MATLAB Runtime
