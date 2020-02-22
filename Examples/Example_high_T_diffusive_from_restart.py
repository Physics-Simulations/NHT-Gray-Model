import os
from GrayModelLibrary import *
import time

##########################################################################################################
												#PARAMETERS										 		 #
##########################################################################################################

Lx = 500e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 10e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 150
Tf = 100
Ti = 125

t_MAX = 50e-9
dt = 4e-12
#Total frames -> t_MAX/dt = 10000

W = 5000
every_flux = 5

#Optional: Default are 100 and 'OUTPUTS'
every_restart = 1000
folder_outputs = 'OUTPUTS'


##########################################################################################################
##########################################################################################################

#RUN NEW SIMULATION

gray_model = GrayModel('high', t_MAX=40e-9, init_restart=True, folder_restart='restart_example')
gray_model.simulation_from_restart(every_restart=every_restart)

