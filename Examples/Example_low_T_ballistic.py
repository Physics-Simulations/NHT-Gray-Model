import os
from GrayModelLibrary import *
import time

##########################################################################################################
												#PARAMETERS										 		 #
##########################################################################################################

Lx = 10e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 0.5e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 11.88
Tf = 3
Ti = 5

t_MAX = 1e-9
dt = 0.1e-12
#Total frames -> t_MAX/dt = 10000

W = 0.05
every_flux = 5

#Optional: Default are 100 and 'OUTPUTS'
every_restart = 1000
folder_outputs = 'OUTPUTS'


##########################################################################################################
##########################################################################################################

#RUN NEW SIMULATION

gray_model = GrayModel('low', Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)
gray_model.simulation(every_restart=every_restart)

