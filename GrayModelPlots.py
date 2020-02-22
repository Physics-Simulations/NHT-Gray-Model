import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

from scipy import stats

import os

current_dir  = os.getcwd()

def diffussive_T(T, T0, Tf, xf):
	'''
		Computes the steady state temperature for the diffussive regime
		from the Fourier Law (Boundaries at x0=0 and xf=xf)
	'''

	k = (Tf - T0) / xf
	return k * T + T0

def balistic_T(T0, Tf):
	'''
		Computes the steady state temperature for the balistic regime
		from the Boltzmann Law
	'''
	return ((T0**4 + Tf**4)/2)**(0.25)

def get_parameters(filename):
	f = open(filename, 'r')

	i = 0

	for line in f:

		try:

			cols = line.split()

			if len(cols) > 0:
				value = float(cols[1])

			if i == 0:
				Lx = value

			elif i == 1:
				Ly = value

			elif i == 2:
				Lz = value

			elif i == 4:
				Lx_subcell = value

			elif i == 5:
				Ly_subcell = value

			elif i == 6:
				Lz_subcell = value

			elif i == 8:
				T0 = value

			elif i == 9:
				Tf = value

			elif i == 10:
				Ti = value

			elif i == 12:
				t_MAX = value

			elif i == 13:
				dt = value

			elif i == 15:
				W = value

			elif i == 16:
				every_flux = value

			i += 1

		except:
			pass

	return Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux


def Ballistic_regime_1D(folder):
	os.chdir(current_dir + '/' + folder)

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

	print('Parameters used')
	print('---------------------------------------\n')

	print('Lx: ' + str(Lx))
	print('Ly: ' + str(Ly))
	print('Lz: ' + str(Lz) + '\n')

	print('Lx_subcell: ' + str(Lx_subcell))
	print('Ly_subcell: ' + str(Ly_subcell))
	print('Lz_subcell: ' + str(Lz_subcell) + '\n')

	print('T0: ' + str(T0))
	print('Tf: ' + str(Tf))
	print('Ti: ' + str(Ti) + '\n')

	print('t_MAX: ' + str(t_MAX))
	print('dt: ' + str(dt) + '\n')

	print('W: ' + str(W))

	E = np.load('Energy.npy')
	N = np.load('Phonons.npy')
	T_cells = np.load('Subcell_Ts.npy')
	scattering_events = np.load('Scattering_events.npy')
	temperatures = np.load('Temperatures.npy')
	elapsed_time = np.load('Elapsed_time.npy')

	print('N avg:', np.mean(N))
	print('SE avg:', np.mean(scattering_events))

	time = np.linspace(0, t_MAX*1e12, int(t_MAX/dt))

	#Subplots
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(time, E)

	plt.title('Energy evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Energy [J]')

	plt.subplot(2, 2, 2)
	plt.plot(time, N)

	plt.title('Number of phonons evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Phonons [#]')

	plt.subplot(2, 2, 3)
	plt.plot(time, temperatures)

	plt.title('Average temperature evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Temperature [K]')

	plt.subplot(2, 2, 4)
	plt.plot(time, scattering_events)

	plt.title('Scattering events in time')
	plt.xlabel('Time [ps]')
	plt.ylabel('Scattering events [#]')

	plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94, wspace=0.2, hspace=0.34)
	plt.show()

	#T plot
	plt.figure(figsize=(10, 6))
	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	#Average for each subcell in equilibrium

	average_T_cells = []
	desv_std = []
	desv_std_neg = []
	equil = 7500

	for i in range(int(round(Lx/Lx_subcell, 0))):
		average_T_cells.append(np.nanmean(T_cells[equil : , i]))
		desv_std.append(np.nanstd(T_cells[equil : , i, 0, 0]))
		desv_std_neg.append(-np.nanstd(T_cells[equil : , i, 0, 0]))

	max_errror = np.max(desv_std)

	y_balistic = np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x))

	print('Max error:', max_errror)

	total_avg_T = np.nanmean(T_cells[equil:])

	plt.plot(x, average_T_cells, ls='', color='r', marker='x', label='Average cell temperature')

	#plt.plot(x, T_cells[0][:, int(Ly / 2), int(Lz/2)], ls='-', color='c', marker='*', label='T at %.2f ps' % float(time[1]))
	#plt.plot(x, T_cells[1000][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^', label='T at %.2f ns' % float(time[1000] * 1e-3))
	#plt.plot(x, T_cells[5000][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='o', label='T at %.2f ns' % float(time[5000]* 1e-3))
	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='g', marker='s', label='T at %.2f ns' % float(time[-1] * 1e-3))

	plt.plot(x, desv_std + y_balistic, ls='--', color='b', label='Standard Deviation')
	plt.plot(x, desv_std_neg + y_balistic, ls='--', color='b')

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Domain temperature evolution')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()

def Diffusive_regime_1D(folder_filename):
	os.chdir(current_dir + '/' + folder_filename)

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

	print('Parameters used')
	print('---------------------------------------\n')

	print('Lx: ' + str(Lx))
	print('Ly: ' + str(Ly))
	print('Lz: ' + str(Lz) + '\n')

	print('Lx_subcell: ' + str(Lx_subcell))
	print('Ly_subcell: ' + str(Ly_subcell))
	print('Lz_subcell: ' + str(Lz_subcell) + '\n')

	print('T0: ' + str(T0))
	print('Tf: ' + str(Tf))
	print('Ti: ' + str(Ti) + '\n')

	print('t_MAX: ' + str(t_MAX))
	print('dt: ' + str(dt) + '\n')

	print('W: ' + str(W))

	time = np.linspace(0, t_MAX*1e12, int(t_MAX/dt))

	E = np.load('Energy.npy')
	N = np.load('Phonons.npy')
	T_cells = np.load('Subcell_Ts.npy')
	scattering_events = np.load('Scattering_events.npy')
	temperatures = np.load('Temperatures.npy')

	print('N avg:', np.mean(N))
	print('SE avg:', np.mean(scattering_events))

	#Subplots
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(time, E)

	plt.title('Energy evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Energy [J]')

	plt.subplot(2, 2, 2)
	plt.plot(time, N)

	plt.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2))

	plt.title('Number of phonons evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Phonons [#]')

	plt.subplot(2, 2, 3)
	plt.plot(time, temperatures)

	plt.title('Average temperature evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Temperature [K]')

	plt.subplot(2, 2, 4)
	plt.plot(time, scattering_events)

	plt.ticklabel_format(axis='y', style='scientific', scilimits=(-2,2))

	plt.title('Scattering events in time')
	plt.xlabel('Time [ps]')
	plt.ylabel('Scattering events [#]')

	plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94, wspace=0.2, hspace=0.34)
	plt.show()

	#T plot
	plt.figure(figsize=(10, 6))
	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	#Average for each subcell in equilibrium

	average_T_cells = []
	desv_std = []
	desv_std_neg = []
	equil = 2000

	for i in range(int(round(Lx/Lx_subcell, 0))):
		average_T_cells.append(np.nanmean(T_cells[equil : , i, 0, 0]))
		desv_std.append(np.nanstd(T_cells[equil : , i, 0, 0]))
		desv_std_neg.append(-np.nanstd(T_cells[equil : , i, 0, 0]))

	max_errror = np.max(desv_std)

	print('Max error:', max_errror)

	#markerfacecolor='None'
	plt.plot(x, average_T_cells, ls='', color='r', marker='x', label='Average cell temperature')
	plt.plot(x, desv_std + diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), ls='-', color='b', label='Standard Deviation')
	plt.plot(x, desv_std_neg + diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), ls='-', color='b')
	#plt.plot(x, T_cells[1000][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^', label='T at %.2f ns' % float(time[1000]*1e-3))
	#plt.plot(x, T_cells[2000][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='o', label='T at %.2f ns' % float(time[2000]*1e-3))
	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='g', marker='s', label='T at %.2f ns' % float(time[-1]*1e-3))

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Domain temperature evolution')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()

def all_plots(folder):
	os.chdir(current_dir + '/' + folder)

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

	print('Parameters used')
	print('---------------------------------------\n')

	print('Lx: ' + str(Lx))
	print('Ly: ' + str(Ly))
	print('Lz: ' + str(Lz) + '\n')

	print('Lx_subcell: ' + str(Lx_subcell))
	print('Ly_subcell: ' + str(Ly_subcell))
	print('Lz_subcell: ' + str(Lz_subcell) + '\n')

	print('T0: ' + str(T0))
	print('Tf: ' + str(Tf))
	print('Ti: ' + str(Ti) + '\n')

	print('t_MAX: ' + str(t_MAX))
	print('dt: ' + str(dt) + '\n')

	print('W: ' + str(W))
	print('Every_flux: ' + str(every_flux))

	E = np.load('Energy.npy')
	N = np.load('Phonons.npy')
	T_cells = np.load('Subcell_Ts.npy')
	scattering_events = np.load('Scattering_events.npy')
	temperatures = np.load('Temperatures.npy')
	elapsed_time = np.load('Elapsed_time.npy')
	flux = np.load('Flux.npy')

	time = np.linspace(0, t_MAX*1e12, int(round(t_MAX/dt, 0)))
	time_flux = np.linspace(0, t_MAX*1e12, int(round(t_MAX/(dt * every_flux))))

	#Subplots
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(time, E)

	plt.title('Energy evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Energy [J]')

	plt.subplot(2, 2, 2)
	plt.plot(time, N)

	plt.title('Number of phonons evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Phonons [#]')

	plt.subplot(2, 2, 3)
	plt.plot(time, temperatures)

	plt.title('Average temperature evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Temperature [K]')

	plt.subplot(2, 2, 4)
	plt.plot(time, scattering_events)

	plt.title('Scattering events in time')
	plt.xlabel('Time [ps]')
	plt.ylabel('Scattering events [#]')

	plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94, wspace=0.2, hspace=0.34)
	plt.show()

	#T plot
	plt.figure(figsize=(10, 6))
	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	#Average for each subcell in equilibrium

	average_T_cells = []
	equil = 500

	for i in range(int(round(Lx/Lx_subcell, 0))):
		average_T_cells.append(np.nanmean(T_cells[equil : , i]))

	total_avg_T = np.nanmean(T_cells[equil:])

	plt.plot(x, average_T_cells, ls='-', color='k', marker='s', label='Average cell T %.2f' % total_avg_T)
	#plt.plot(x, T_cells[0][:, int(Ly / 2), int(Lz/2)], ls='-', color='c', marker='*', label='T at %.2f ps' % float(time[1]))
	#plt.plot(x, T_cells[1000][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^', label='T at %.2f ps' % float(time[1000]))
	#plt.plot(x, T_cells[5000][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='o', label='T at %.2f ps' % float(time[5000]))
	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='g', marker='s', label='T at %.2f ns' % float(time[-1] * 1e-3))

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Domain temperature evolution')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()

	plt.plot(time_flux, flux, ls='--', color='c', label='')
	plt.title('Flux')
	plt.show()

	plt.plot(time, elapsed_time, ls='', marker='x', color='g')

	slope, intercept, r_value, p_value, std_error = stats.linregress(time, elapsed_time)

	y_regre = np.array(time) * slope + intercept

	plt.plot(time, y_regre, color='r', label='y=%.2fx+%.2f' % (slope, intercept))

	plt.title('Elapsed time')
	plt.legend()
	plt.show()

def plots_restart(folder, t_max):
	os.chdir(current_dir + '/' + folder)

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

	print('Parameters used')
	print('---------------------------------------\n')

	print('Lx: ' + str(Lx))
	print('Ly: ' + str(Ly))
	print('Lz: ' + str(Lz) + '\n')

	print('Lx_subcell: ' + str(Lx_subcell))
	print('Ly_subcell: ' + str(Ly_subcell))
	print('Lz_subcell: ' + str(Lz_subcell) + '\n')

	print('T0: ' + str(T0))
	print('Tf: ' + str(Tf))
	print('Ti: ' + str(Ti) + '\n')

	print('t_MAX: ' + str(t_MAX))
	print('dt: ' + str(dt) + '\n')

	print('W: ' + str(W))
	print('Every_flux: ' + str(every_flux))

	E = np.load('Energy.npy')
	N = np.load('Phonons.npy')
	T_cells = np.load('Subcell_Ts.npy')
	scattering_events = np.load('Scattering_events.npy')
	temperatures = np.load('Temperatures.npy')
	elapsed_time = np.load('Elapsed_time.npy')
	flux = np.load('Flux.npy')

	time = np.linspace(0, len(E), len(E))
	time_flux = np.linspace(0, len(flux), len(flux))

	#Subplots
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(time, E)

	plt.title('Energy evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Energy [J]')

	plt.subplot(2, 2, 2)
	plt.plot(time, N)

	plt.title('Number of phonons evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Phonons [#]')

	plt.subplot(2, 2, 3)
	plt.plot(time, temperatures)

	plt.title('Average temperature evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Temperature [K]')

	plt.subplot(2, 2, 4)
	plt.plot(time, scattering_events)

	plt.title('Scattering events in time')
	plt.xlabel('Time [ps]')
	plt.ylabel('Scattering events [#]')

	plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94, wspace=0.2, hspace=0.34)
	plt.show()

	#T plot
	plt.figure(figsize=(10, 6))
	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	#Average for each subcell in equilibrium

	average_T_cells = []
	equil = 7500

	for i in range(int(round(Lx/Lx_subcell, 0))):
		average_T_cells.append(np.nanmean(T_cells[equil : , i]))

	total_avg_T = np.nanmean(T_cells[equil:])

	plt.plot(x, average_T_cells, ls='-', color='k', marker='s', label='Average cell T %.2f' % total_avg_T)
	plt.plot(x, T_cells[0][:, int(Ly / 2), int(Lz/2)], ls='-', color='c', marker='*', label='T at %.2f ps' % float(time[1]))
	#plt.plot(x, T_cells[1000][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^', label='T at %.2f ps' % float(time[1000]))
	#plt.plot(x, T_cells[5000][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='o', label='T at %.2f ps' % float(time[5000]))
	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='g', marker='s', label='T at %.2f ns' % float(time[-1] * 1e-3))

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Domain temperature evolution')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()

	print('\n\nAvg_flux:', np.nanmean(flux))

	plt.plot(time_flux, flux, ls='--', color='c', label='')
	plt.title('Flux')
	plt.show()

	plt.plot(time, elapsed_time, ls='', marker='x', color='g')

	slope, intercept, r_value, p_value, std_error = stats.linregress(time, elapsed_time)

	y_regre = np.array(time) * slope + intercept

	plt.plot(time, y_regre, color='r', label='y=%.2fx+%.2f' % (slope, intercept))

	plt.title('Elapsed time')
	plt.legend()
	plt.show()


########################
#	UNCOMMENT TO USE   #
########################

#Ballistic_regime_1D('OUTPUTS')
#Diffusive_regime_1D('OUTPUTS')

#all_plots('OUTPUTS')
#plots_restart('restart_1000' 1000)
