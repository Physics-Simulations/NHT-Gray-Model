import numpy as np
import os
import random
import math
from scipy import integrate, stats

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

import time

current_dir = os.getcwd()
array_folder = current_dir + '/Input_arrays'

if not os.path.exists(array_folder): os.mkdir(array_folder)

current_time = lambda: round(time.time(), 2)

##########################################################	
##########################################################
#														 #
#					PARAMETERS 							 #				
#														 #
##########################################################	
##########################################################

##########################################################
#														 #
#					Silicium parameters					 #
#														 #	
##########################################################

#Constants
v_si_LA = 9.01e3
v_si_TA = 5.23e3 
v_si_LO = 0
v_si_TO = -2.57e3

c_si_LA = -2e-7
c_si_TA = -2.26e-7
c_si_LO = -1.6e-7
c_si_TO = 1.12e-7

omega_0_si_LA = 0
omega_0_si_TA = 0
omega_0_si_LO = 9.88e13
omega_0_si_TO = 10.2e13

k_max_si = 1.157e10 / 2

#Maximum frequencies for silicon
w_max_si_LA = 7.747e13
w_max_si_TA = 3.026e13
w_max_si_LO = omega_0_si_LO
w_max_si_TO = omega_0_si_TO

#Minimum frequencies for silicon
w_min_si_LA = 0
w_min_si_TA = 0
w_min_si_LO = 7.738e13
w_min_si_TO = 8.726e13

k_bulk_si = 139

vs_si = [v_si_TA, v_si_LA, v_si_LO, v_si_TO]
cs_si = [c_si_TA, c_si_LA, c_si_LO, c_si_TO]
maximum_freqs_si = [w_max_si_TA, w_max_si_LA, w_max_si_LO, w_max_si_TO]
minimum_freqs_si = [w_min_si_TA, w_min_si_LA, w_min_si_LO, w_min_si_TO]
omegas_0_si = [omega_0_si_TA, omega_0_si_LA, omega_0_si_LO, omega_0_si_TO]

##########################################################
#														 #
#					Germanium parameters				 #
#														 #	
##########################################################

#Constants
v_ge_LA = 5.3e3
v_ge_TA = 2.26e3 
v_ge_LO = -0.99e3
v_ge_TO = -0.18e3

c_ge_LA = -1.2e-7
c_ge_TA = -0.82e-7
c_ge_LO = -0.48e-7
c_ge_TO = 0

omega_0_ge_LA = 0
omega_0_ge_TA = 0
omega_0_ge_LO = 5.7e13
omega_0_ge_TO = 5.5e13

k_max_ge = 1.1105e10 / 2

#Maximum frequencies for germanium
w_max_ge_LA = 4.406e13
w_max_ge_TA = 1.498e13
w_max_ge_LO = omega_0_ge_LO
w_max_ge_TO = omega_0_ge_TO

#Minimum frequencies for germanium
w_min_ge_LA = 0
w_min_ge_TA = 0
w_min_ge_LO = 4.009e13
w_min_ge_TO = 5.3e13

k_bulk_ge = 58

vs_ge = [v_ge_TA, v_ge_LA, v_ge_LO, v_ge_TO]
cs_ge = [c_ge_TA, c_ge_LA, c_ge_LO, c_ge_TO]
maximum_freqs_ge = [w_max_ge_TA, w_max_ge_LA, w_max_ge_LO, w_max_ge_TO]
minimum_freqs_ge = [w_min_ge_TA, w_min_ge_LA, w_min_ge_LO, w_min_ge_TO]
omegas_0_ge = [omega_0_ge_TA, omega_0_ge_LA, omega_0_ge_LO, omega_0_ge_TO]

##########################################################
#														 #
#					General parameters				 	 #
#														 #	
##########################################################

hbar = 1.05457e-34
k_B = 1.38064e-23

k_max_array = [k_max_si, k_max_ge]


############################################
#										   #
#			 General Functions			   #
#										   #
############################################

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

def save_arrays_germanium(init_T, final_T, n):
	os.chdir(array_folder)

	properties = ThermalProperties(init_T, final_T, n, maximum_freqs_ge, vs_ge, cs_ge, omegas_0_ge, k_bulk_ge, 'Germanium')
	N, E, w, v, CV, MFP, E_tot, T = properties.fill_arrays()

	np.save('N_input.npy', N)
	np.save('E_input.npy', E)
	np.save('w_input.npy', w)
	np.save('v_input.npy', v)
	np.save('CV_input.npy', CV)
	np.save('MFP_input.npy', MFP)
	np.save('Etot_input.npy', E_tot)
	np.save('T_input.npy', T)

def save_arrays_silicon(init_T, final_T, n):

	os.chdir(array_folder)

	properties = ThermalProperties(init_T, final_T, n, maximum_freqs_si, vs_si, cs_si, omegas_0_si, k_bulk_si, 'Silicon')
	N, E, w, v, CV, MFP, E_tot, T = properties.fill_arrays()
 
	np.save('N_input.npy', N)
	np.save('E_input.npy', E)
	np.save('w_input.npy', w)
	np.save('v_input.npy', v)
	np.save('CV_input.npy', CV)
	np.save('MFP_input.npy', MFP)
	np.save('Etot_input.npy', E_tot)
	np.save('T_input.npy', T)

############################################
#										   #
#				  Classes			       #
#										   #
############################################

class ThermalProperties(object):
	def __init__(self, T_0, T_max, n, w_max_array, v_array, c_array, omega_0_array, k_bulk, name):
		self.Ns = []
		self.Es = []
		self.ws = []
		self.vs = []
		self.CVs = []
		self.MFPs = []
		self.E_tot = []

		self.T_0 = T_0
		self.T_max = T_max
		self.n = n

		self.Ts = np.linspace(self.T_0, self.T_max, self.n)

		self.w_max_array = w_max_array
		self.v_array = v_array
		self.c_array = c_array
		self.omega_0_array = omega_0_array
		self.k_bulk = k_bulk

		self.name = name

	def N(self, T):

		def f_N(w, T, v, c, omega_0):

			num = (-v + np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))**2 
			denom = 4*c**2 * (np.exp(hbar*w / (k_B*T)) - 1)*(np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))

			return num / denom

		N = 0
		for i in range(2): #Sum for all considered polarizations
			w_max = self.w_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			N += integrate.quad(f_N, 1, w_max, args = (T, v_i, c_i, omega_0_i))[0]

			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				N *= 2

		return N  / (2*np.pi**2) 

	def E(self, T):

		def f_E(w, T, v, c, omega_0):

			num = hbar * w * (-v + np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))**2
			denom = 4*c**2 * (np.exp(hbar*w / (k_B*T)) - 1)*(np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))

			return num / denom

		E = 0
		for i in range(2): #Sum for all considered polarizations
			w_max = self.w_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			E += integrate.quad(f_E, 1, w_max, args = (T, v_i, c_i, omega_0_i))[0]

			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				E *= 2

		return E / (2*np.pi**2)

	def v_avg(self, T):

		def f_v(w, T, v, c, omega_0):

			num = (-v + np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))**2 
			denom = 4*c**2 * (np.exp(hbar*w / (k_B*T)) - 1)

			return num / denom

		x = 0
		for i in range(2): #Sum for all considered polarizations
			w_max = self.w_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			x += integrate.quad(f_v, 1, w_max, args = (T, v_i, c_i, omega_0_i))[0]

			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				x *= 2

		return x / (2*np.pi**2) 

	def C_V(self, T):

		def f_C_V(w, T, v, c, omega_0):

			num = (-v + np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))**2 * (hbar * w)**2 * np.exp(hbar * w / (k_B * T))
			denom = 4*c**2 * k_B * T**2 * (np.exp(hbar*w / (k_B*T)) - 1)**2 * (np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))

			return num / denom

		CV = 0
		for i in range(2): #Sum for all considered polarizations
			w_max = self.w_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			CV += integrate.quad(f_C_V, 1, w_max, args = (T, v_i, c_i, omega_0_i))[0]
			
			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				CV *= 2

		return CV / (2*np.pi**2)

	def fill_arrays(self):
		for T in self.Ts:
			N_T = self.N(T)
			E_T = self.E(T)
			v = self.v_avg(T)
			CV_T = self.C_V(T)

			self.Ns.append(N_T) #N per unit volume
			self.E_tot.append(E_T) #E per unit volume
			self.Es.append(E_T / N_T) #E per unit volume per phonon
			self.ws.append(E_T / (hbar * N_T)) #w_avg
			self.vs.append(v / N_T) #v_avg
			self.CVs.append(CV_T) #Cv per unit volume
			self.MFPs.append(3 * N_T * self.k_bulk / (v * CV_T)) #MFP

		return self.Ns, self.Es, self.ws, self.vs, self.CVs, self.MFPs, self.E_tot, self.Ts

	def plot_properties(self):
		#N(T)
		plt.subplot(3, 2, 1)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.Ns)
		#plt.title('Nº phonons vs temperature')
		plt.ylabel('Nº phonons')
		plt.xlabel('T (K)')
		
		#E(T)
		plt.subplot(3, 2, 2)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.Es)
		#plt.title('Energy vs temperature')
		plt.ylabel('E (J) per phonon')
		plt.xlabel('T (K)')

		#w_avg
		plt.subplot(3, 2, 3)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.ws)
		#plt.title('Average frequency vs Temperature')
		plt.xlabel('T(K)')
		plt.ylabel(r'$\omega_{avg} \, (rad/s)$')

		#v_avg
		plt.subplot(3, 2, 4)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.vs)
		#plt.title('Average group velocity vs Temperature')
		plt.xlabel('T(K)')
		plt.ylabel(r'$v_{avg} \, (m/s)$')

		#C_V
		plt.subplot(3, 2, 5)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.CVs)
		#plt.title('Heat capacity vs temperature')
		plt.ylabel(r'$C_V$ (J/K)')
		plt.xlabel('T (K)')

		#MFP
		plt.subplot(3, 2, 6)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.MFPs)
		#plt.title('Mean Free Path vs Temperature')
		plt.xlabel('T(K)')
		plt.ylabel(r'$\Lambda \, (m)$')

		plt.suptitle('Thermal transport properties for %s' % self.name)
		plt.show()

class GrayModel(object):
	def __init__(self, type_, Lx=0, Ly=0, Lz=0, Lx_subcell=0, Ly_subcell=0, Lz_subcell=0, T0=0, Tf=0, Ti=0, t_MAX=0, dt=0, 
		W=0, every_flux=0, init_restart = False, folder_restart = 'None'):

		if type_ != 'high' and type_ != 'low':
			raise ValueError('Invalid type argument')
			
		if init_restart :

			self.read_restart(current_dir + '/' + folder_restart)

			self.type = type_

			if t_MAX != 0: 
				self.t_MAX = t_MAX
				self.Nt = int(round(t_MAX / self.dt, 0))

		else:

			self.Lx = float(Lx) #x length of the box
			self.Ly = float(Ly) #y length of the box
			self.Lz = float(Lz) #z lenght of the box
			self.T0 = float(T0) #Temperature of the initial sub-cell (Boundary)
			self.Tf = float(Tf) #Temperature of the last sub-cell (Boundary)
			self.Ti = float(Ti) #Initial temperature of studied subcells
			self.t_MAX = float(t_MAX) #Maximum simulation time
			self.dt = float(dt) #Step size
			self.W = float(W) #Weighting factor

			self.Nt = int(round(self.t_MAX / self.dt, 0)) #Number of simulation steps/iterations

			self.Lx_subcell = float(Lx_subcell) #x length of each subcell
			self.Ly_subcell = float(Ly_subcell) #x length of each subcell
			self.Lz_subcell = float(Lz_subcell) #x length of each subcell

			self.N_subcells_x = int(round(self.Lx / self.Lx_subcell, 0)) #Number of subcells
			self.N_subcells_y = int(round(self.Ly / self.Ly_subcell, 0))
			self.N_subcells_z = int(round(self.Lz / self.Lz_subcell, 0)) 

			self.every_flux = every_flux

			self.type = type_

			self.r = [] #list of the positions of all the phonons
			self.v = [] #list of the velocities of all the phonons

			self.E = []
			self.N = []
			self.w_avg = []
			self.v_avg = []
			self.C_V = []
			self.MFP = []

			self.scattering_time = []

			self.subcell_Ts = np.zeros((self.N_subcells_x, self.N_subcells_y, self.N_subcells_z))

		self.V_subcell = self.Ly_subcell * self.Lz_subcell * self.Lx_subcell

		#Load arrays
		os.chdir(array_folder)

		#Germanium
		self.N_ge = np.load('N_input.npy')
		self.E_ge = np.load('E_input.npy')
		self.w_ge = np.load('w_input.npy')
		self.v_ge = np.load('v_input.npy')
		self.CV_ge = np.load('CV_input.npy')
		self.MFP_ge = np.load('MFP_input.npy')
		self.Etot_ge = np.load('Etot_input.npy')

		#Temperature array
		self.Ts = np.load('T_input.npy')

		#Account for the different volumes
		self.N_ge *= self.V_subcell 
		self.CV_ge *= self.V_subcell
		self.Etot_ge *= self.V_subcell

		#Maximum energies
		self.E_max_ge = 1.659e-21

	def find_T(self, value, T): 
		'''
		For a given value of temperature returns the position in the T array
		'''
		
		for i in range(len(T)):
			if T[i] >= value:
				return i

	def create_phonons(self, N, subcell_x, subcell_y, subcell_z, T):
		r = np.zeros((N, 3)) #Array of vector positions
		v = np.zeros((N, 3)) #Array of vector velocities

		rx = np.random.random((N,)) * self.Lx_subcell + subcell_x * self.Lx_subcell
		ry = np.random.random((N,)) * self.Ly_subcell + subcell_y * self.Ly_subcell
		rz = np.random.random((N,)) * self.Lz_subcell + subcell_z * self.Lz_subcell

		pos = self.find_T(T, self.Ts)

		for j in range(N):
			r[j][0] = rx[j]
			r[j][1] = ry[j]
			r[j][2] = rz[j]

			self.E.append(self.E_ge[pos])
			self.v_avg.append(self.v_ge[pos])
			self.w_avg.append(self.w_ge[pos])
			self.C_V.append(self.CV_ge[pos])
			self.MFP.append(self.MFP_ge[pos])
			self.scattering_time.append(0.)

		v_polar = np.random.random((N, 2))

		v[:,0] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.cos(v_polar[:,1] * 2 * np.pi)) 
		v[:,1] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.sin(v_polar[:,1] * 2 * np.pi)) 
		v[:,2] = np.cos(np.cos(2*v_polar[:,0]-1)**(-1))

		v *= self.v_ge[pos]

		self.r += list(r)
		self.v += list(v)

	def init_particles(self):

		for i in range(self.N_subcells_x):
			for j in range(self.N_subcells_y):
				for k in range(self.N_subcells_z):

					if i == 0:
						T_i = self.T0
						self.subcell_Ts[i][j][k] = self.T0

					elif i == self.N_subcells_x - 1:
						T_i = self.Tf
						self.subcell_Ts[i][j][k] = self.Tf

					else:
						T_i = self.Ti
						self.subcell_Ts[i][j][k] = self.Ti

					pos = self.find_T(T_i, self.Ts)

					N = int(self.N_ge[pos] / self.W)

					self.create_phonons(N, i, j, k, T_i)

		self.r = np.array(self.r)
		self.v = np.array(self.v)
		self.E = np.array(self.E)
		self.v_avg = np.array(self.v_avg)
		self.w_avg = np.array(self.w_avg)
		self.C_V = np.array(self.C_V)
		self.MFP = np.array(self.MFP)
		self.scattering_time = np.array(self.scattering_time)

		return self.r, self.v, self.E, self.v_avg, self.w_avg, self.C_V, self.MFP

	def check_boundaries(self, i):

		if self.r[i][0] >= self.Lx or self.r[i][0] < 0:
			self.v[i][0] *= -1.

			if self.r[i][0] > self.Lx:
				self.r[i][0] = self.Lx
			else:
				self.r[i][0] = 0

		if self.r[i][1] > self.Ly or self.r[i][1] < 0:
			self.v[i][1] *= -1.

			if self.r[i][1] > self.Ly:
				delta_y = self.r[i][1] - self.Ly
				self.r[i][1] = self.r[i][1] - 2*delta_y
			else:
				delta_y = -self.r[i][1] 
				self.r[i][1] = delta_y

		if self.r[i][2] > self.Lz or self.r[i][2] < 0:
			self.v[i][2] *= -1.

			if self.r[i][2] > self.Lz:
				delta_z = self.r[i][2] - self.Lz
				self.r[i][2] = self.r[i][2] - 2*delta_z
			else:
				delta_z = -self.r[i][2] 
				self.r[i][2] = delta_z
		
	def match_T(self, value, E, T):
		for i in range(len(E)):
			if E[i] == value:
				return T[i]

			elif E[i] > value: #If we exceed the value, use interpolation
				return T[i] * value /  E[i]

	def calculate_subcell_T(self):

		E_subcells = np.zeros((self.N_subcells_x, self.N_subcells_y, self.N_subcells_z))
		N_subcells = np.zeros((self.N_subcells_x, self.N_subcells_y, self.N_subcells_z))

		for i in range(len(self.r)):
			x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
			y = int(self.r[i][1] / self.Ly * self.N_subcells_y)
			z = int(self.r[i][2] / self.Lz * self.N_subcells_z)

			E_subcells[x][y][z] += self.W * self.E[i]
			N_subcells[x][y][z] += self.W

		if self.type == 'high':

			for i in range(self.N_subcells_x):
				for j in range(self.N_subcells_y):
					for k in range(self.N_subcells_z):

						E_N = E_subcells[i][j][k] / N_subcells[i][j][k]

						self.subcell_Ts[i][j][k] = self.match_T(E_N, self.E_ge, self.Ts)

		elif self.type == 'low':

			for i in range(self.N_subcells_x):
				for j in range(self.N_subcells_y):
					for k in range(self.N_subcells_z):

						self.subcell_Ts[i][j][k] = self.match_T(E_subcells[i][j][k], self.Etot_ge, self.Ts)


		return E_subcells, N_subcells

	def find_subcell(self, i):
		for j in range(1, self.N_subcells - 1):
			if self.r[i][0] >=  j * self.Lx_subcell and self.r[i][0] <= (j + 1) * self.Lx_subcell: #It will be in the j_th subcell
				return j	

	def scattering(self):
		scattering_events = 0

		for i in range(len(self.r)):
			x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
			y = int(self.r[i][1] / self.Ly * self.N_subcells_y)
			z = int(self.r[i][2] / self.Lz * self.N_subcells_z)

			if x < 1 or x > (self.N_subcells_x - 1):
				pass #Avoid scattering for phonons in hot and cold boundary cells

			else:
				prob = 1 - np.exp(-self.v_avg[i] * self.scattering_time[i] / self.MFP[i])
				
				dice = random.uniform(0, 1)

				if prob > dice :#Scattering process

					v_polar = np.random.random((1, 2))

					self.v[i][0] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.cos(v_polar[:,1] * 2 * np.pi)) 
					self.v[i][1] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.sin(v_polar[:,1] * 2 * np.pi)) 
					self.v[i][2] = np.cos(np.cos(2*v_polar[:,0]-1)**(-1))

					current_T = self.subcell_Ts[x][y][z]
					pos = self.find_T(current_T, self.Ts)

					self.v[i] *= self.v_ge[pos]

					self.v_avg[i] = self.v_ge[pos]
					self.w_avg[i] = self.w_ge[pos]
					self.E[i] = self.E_ge[pos]
					self.C_V[i] = self.CV_ge[pos]
					self.MFP[i] = self.MFP_ge[pos]

					self.scattering_time[i] = 0. #Re-init scattering time

					scattering_events += self.W

				else:
					self.scattering_time[i] += self.dt #Account for the scattering time

		return scattering_events

	def energy_conservation(self, delta_E):
		for i in range(1, self.N_subcells_x - 1):
			for j in range(self.N_subcells_y):
				for k in range(self.N_subcells_z):

					current_energy = delta_E[i][j][k]

					if current_energy > self.E_max_ge:

						while current_energy > self.E_max_ge: #Delete phonons

							for l in range(len(self.r)):

								x = int((self.r[l][0] / self.Lx) * self.N_subcells_x)
								y = int((self.r[l][1] / self.Lx) * self.N_subcells_y)
								z = int((self.r[l][2] / self.Lx) * self.N_subcells_z)

								if x == i and y == j and z == k : #is in the i_th subcell

									current_energy -= self.E[l] * self.W

									self.r = np.delete(self.r, l, 0)
									self.v = np.delete(self.v, l, 0)
									self.E = np.delete(self.E, l, 0)
									self.v_avg = np.delete(self.v_avg, l, 0)
									self.w_avg = np.delete(self.w_avg, l, 0)
									self.C_V = np.delete(self.C_V, l, 0)
									self.MFP = np.delete(self.MFP, l, 0)
									self.scattering_time = np.delete(self.scattering_time, l, 0)

									break


					if -delta_E[i][j][k] > self.E_max_ge: #Production of phonons

						while -current_energy > self.E_max_ge:

							T = self.subcell_Ts[i][j][k]
							pos_T = self.find_T(T, self.Ts)

							E_phonon_T = self.E_ge[pos_T] #Energy per phonon for this subcell T	

							self.r = list(self.r)
							self.v = list(self.v)
							self.v_avg = list(self.v_avg)
							self.w_avg = list(self.w_avg)
							self.E = list(self.E)
							self.C_V = list(self.C_V)
							self.MFP = list(self.MFP)
							self.scattering_time = list(self.scattering_time)

							self.create_phonons(1, i, j, k, T)

							self.r = np.array(self.r)
							self.v = np.array(self.v)
							self.v_avg = np.array(self.v_avg)
							self.w_avg = np.array(self.w_avg)
							self.E = np.array(self.E)
							self.C_V = np.array(self.C_V)
							self.MFP = np.array(self.MFP)
							self.scattering_time = np.array(self.scattering_time)

							current_energy += E_phonon_T * self.W

	def re_init_boundary(self): #Eliminar tots i posar tots nous
		pos_T0 = self.find_T(self.T0, self.Ts)
		pos_Tf = self.find_T(self.Tf, self.Ts)

		N_0 = int(round(self.N_ge[pos_T0] / self.W, 0))
		N_f = int(round(self.N_ge[pos_Tf] / self.W, 0))

		total_indexs = []

		#Delete all the phonons in boundary subcells
		for i in range(len(self.r)):
			if self.r[i][0] <= self.Lx_subcell: #Subcell with T0 boundary
				total_indexs.append(i)

			elif self.r[i][0] >= (self.N_subcells_x - 1) * self.Lx_subcell: #Subcell with Tf boundary
				total_indexs.append(i)

		self.r = np.delete(self.r, total_indexs, 0)
		self.v = np.delete(self.v, total_indexs, 0)
		self.E = np.delete(self.E, total_indexs, 0)
		self.v_avg = np.delete(self.v_avg, total_indexs, 0)
		self.w_avg = np.delete(self.w_avg, total_indexs, 0)
		self.C_V = np.delete(self.C_V, total_indexs, 0)
		self.MFP = np.delete(self.MFP, total_indexs, 0)
		self.scattering_time = np.delete(self.scattering_time, total_indexs, 0)

		#Create the new phonons
		self.r = list(self.r)
		self.v = list(self.v)
		self.E = list(self.E)
		self.v_avg = list(self.v_avg)
		self.w_avg = list(self.w_avg)
		self.C_V = list(self.C_V)
		self.MFP = list(self.MFP)
		self.scattering_time = list(self.scattering_time)

		for j in range(self.N_subcells_y):
			for k in range(self.N_subcells_z):

				self.create_phonons(N_0, 0, j, k, self.T0)
				self.create_phonons(N_f, self.N_subcells_x - 1, j, k, self.Tf)

		self.r = np.array(self.r)
		self.v = np.array(self.v)
		self.E = np.array(self.E)
		self.v_avg = np.array(self.v_avg)
		self.w_avg = np.array(self.w_avg)
		self.C_V = np.array(self.C_V)
		self.MFP = np.array(self.MFP)
		self.scattering_time = np.array(self.scattering_time)

	def calculate_flux(self, i, r_previous):
		'''
			Calculates the flux in the yz plane in the middle of Lx lenght
		'''

		if self.r[i][0] > self.Lx/2 and r_previous[i][0] < self.Lx/2:

			return self.E[i] * self.W

		elif self.r[i][0] < self.Lx/2 and r_previous[i][0] > self.Lx/2:
			return -self.E[i] * self.W

		else:
			return 0

	def save_restart(self, nt):

		os.chdir(current_dir)

		if not os.path.exists('restart_%i' % nt): os.mkdir('restart_%i' % nt)
		os.chdir('restart_%i' % nt)

		np.save('r.npy', self.r)
		np.save('v.npy', self.v)

		np.save('E.npy', self.E)
		np.save('N.npy', self.N)
		np.save('w_avg.npy', self.w_avg)
		np.save('v_avg.npy', self.v_avg)
		np.save('C_V.npy', self.C_V)
		np.save('MFP.npy', self.MFP)

		np.save('scattering_time.npy', self.scattering_time)

		np.save('subcell_Ts.npy', self.subcell_Ts)

		f = open('parameters_used.txt', 'w')

		f.write('Lx: ' + str(self.Lx) + '\n')
		f.write('Ly: ' + str(self.Ly) + '\n')
		f.write('Lz: ' + str(self.Lz) + '\n\n')

		f.write('Lx_subcell: ' + str(self.Lx_subcell) + '\n')
		f.write('Ly_subcell: ' + str(self.Ly_subcell) + '\n')
		f.write('Lz_subcell: ' + str(self.Lz_subcell) + '\n\n')

		f.write('T0: ' + str(self.T0) + '\n')
		f.write('Tf: ' + str(self.Tf) + '\n')
		f.write('Ti: ' + str(self.Ti) + '\n\n')

		f.write('t_MAX: ' + str(self.t_MAX) + '\n')
		f.write('dt: ' + str(self.dt) + '\n\n')

		f.write('W: ' + str(self.W) + '\n')
		f.write('Every_flux: ' + str(self.every_flux))

		f.close()

	def get_parameters(self):
		f = open('parameters_used.txt', 'r')

		i = 0

		for line in f:

			try:

				cols = line.split()

				if len(cols) > 0:
					value = float(cols[1])

				if i == 0:
					self.Lx = value

				elif i == 1:
					self.Ly = value

				elif i == 2:
					self.Lz = value

				elif i == 4:
					self.Lx_subcell = value

				elif i == 5:
					self.Ly_subcell = value

				elif i == 6:
					self.Lz_subcell = value

				elif i == 8:
					self.T0 = value

				elif i == 9:
					self.Tf = value

				elif i == 10:
					self.Ti = value

				elif i == 12:
					self.t_MAX = value

				elif i == 13:
					self.dt = value

				elif i == 15:
					self.W = value

				elif i == 16:
					self.every_flux = value

				i += 1

			except:
				pass

	def read_restart(self, folder):

		os.chdir(folder)

		self.r = np.load('r.npy')
		self.v = np.load('v.npy')

		self.E = np.load('E.npy')
		self.N = np.load('N.npy')
		self.w_avg = np.load('w_avg.npy')
		self.v_avg = np.load('v_avg.npy')
		self.C_V = np.load('C_V.npy')
		self.MFP = np.load('MFP.npy')

		self.scattering_time = np.load('scattering_time.npy')

		self.subcell_Ts = np.load('subcell_Ts.npy')

		self.get_parameters()

		self.Nt = int(round(self.t_MAX / self.dt, 0)) #Number of simulation steps/iterations

		self.N_subcells_x = int(round(self.Lx / self.Lx_subcell, 0)) #Number of subcells
		self.N_subcells_y = int(round(self.Ly / self.Ly_subcell, 0))
		self.N_subcells_z = int(round(self.Lz / self.Lz_subcell, 0)) 

		os.chdir(current_dir)

	def simulation(self, every_restart=100, folder_outputs='OUTPUTS'):
		os.chdir(current_dir)

		self.init_particles()

		Energy = []
		Phonons = []
		Temperatures = []
		delta_energy = []
		scattering_events = []
		cell_temperatures = []
		elapsed_time = []

		flux = []

		t0 = current_time()

		for k in range(self.Nt):
			print('Timestep:', k, 'of', self.Nt, '(%.2f)' % (100 * k/self.Nt), '%')

			if k % every_restart == 0:

				#Save configuration actual properties
				self.save_restart(k)

				#Save outputs untill this moment (Inside the restart folder)
				flux_save = np.array(flux) / (self.Ly * self.Lz * self.dt)

				np.save('Energy.npy', Energy)
				np.save('Phonons.npy', Phonons)
				np.save('Subcell_Ts.npy', cell_temperatures)
				np.save('Temperatures.npy', Temperatures)
				np.save('Scattering_events.npy', scattering_events)
				np.save('Elapsed_time.npy', elapsed_time)
				np.save('Flux.npy', flux_save)

				os.chdir(current_dir) #Go back to the principal directory

			if k % int(self.every_flux) == 0:

				previous_r = np.copy(self.r) #Save the previous positions to calculate the flux

				self.r += self.dt * self.v #Drift

				flux_k = 0

				for i in range(len(self.r)):
					self.check_boundaries(i)

					flux_k += self.calculate_flux(i, previous_r)

				flux.append(flux_k)

			else:
				self.r += self.dt * self.v #Drift

				for i in range(len(self.r)):
					self.check_boundaries(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T() #Calculate energy before scattering

			scattering_events.append(self.scattering())

			E_subcells_new , N_subcells_new = self.calculate_subcell_T() #Calculate energy after scattering

			delta_E = np.array(E_subcells_new) - np.array(E_subcells) #Account for loss or gain of energy

			self.energy_conservation(delta_E) #Impose energy conservation

			E_subcells_final, N_subcells_final = self.calculate_subcell_T() #Calculate final T

			delta_E_final = np.array(E_subcells_final) - np.array(E_subcells)

			delta_energy.append(np.mean(delta_E_final))
			Energy.append(np.sum(E_subcells_final))
			Phonons.append(np.sum(N_subcells_final))
			Temperatures.append(np.mean(self.subcell_Ts))

			copy_subcells = np.copy(self.subcell_Ts)

			cell_temperatures.append(copy_subcells)

			elapsed_time.append(current_time() - t0)

		#Save last restart
		self.save_restart(k+1)

		if not  os.path.exists(current_dir + '/' + folder_outputs): os.mkdir(current_dir + '/' + folder_outputs)
		os.chdir(current_dir + '/' + folder_outputs)

		flux = np.array(flux) / (self.Ly * self.Lz * self.dt)

		np.save('Energy.npy', Energy)
		np.save('Phonons.npy', Phonons)
		np.save('Subcell_Ts.npy', cell_temperatures)
		np.save('Temperatures.npy', Temperatures)
		np.save('Scattering_events.npy', scattering_events)
		np.save('Elapsed_time.npy', elapsed_time)
		np.save('Flux.npy', flux)

		f = open('parameters_used.txt', 'w')

		f.write('Lx: ' + str(self.Lx) + '\n')
		f.write('Ly: ' + str(self.Ly) + '\n')
		f.write('Lz: ' + str(self.Lz) + '\n\n')

		f.write('Lx_subcell: ' + str(self.Lx_subcell) + '\n')
		f.write('Ly_subcell: ' + str(self.Ly_subcell) + '\n')
		f.write('Lz_subcell: ' + str(self.Lz_subcell) + '\n\n')

		f.write('T0: ' + str(self.T0) + '\n')
		f.write('Tf: ' + str(self.Tf) + '\n')
		f.write('Ti: ' + str(self.Ti) + '\n\n')

		f.write('t_MAX: ' + str(self.t_MAX) + '\n')
		f.write('dt: ' + str(self.dt) + '\n\n')

		f.write('W: ' + str(self.W) + '\n')
		f.write('Every_flux: ' + str(self.every_flux))

		f.close()

		print('\nSimulation finished!')

	def simulation_from_restart(self, every_restart=100, folder_outputs='OUTPUTS'):

		Energy = []
		Phonons = []
		Temperatures = []
		delta_energy = []
		scattering_events = []
		cell_temperatures = []
		elapsed_time = []

		flux = []

		t0 = current_time()

		for k in range(self.Nt):
			print('Timestep:', k, 'of', self.Nt, '(%.2f)' % (100 * k/self.Nt), '%')

			if k % every_restart == 0:

				self.save_restart(k)

				#Save outputs untill this moment (Inside the restart folder)
				flux_save = np.array(flux) / (self.Ly * self.Lz * self.dt * self.every_flux)

				np.save('Energy.npy', Energy)
				np.save('Phonons.npy', Phonons)
				np.save('Subcell_Ts.npy', cell_temperatures)
				np.save('Temperatures.npy', Temperatures)
				np.save('Scattering_events.npy', scattering_events)
				np.save('Elapsed_time.npy', elapsed_time)
				np.save('Flux.npy', flux_save)

				os.chdir(current_dir) #Go back to the principal directory

			if k % int(self.every_flux) == 0:

				previous_r = np.copy(self.r) #Save the previous positions to calculate the flux

				self.r += self.dt * self.v #Drift

				flux_k = 0

				for i in range(len(self.r)):
					self.check_boundaries(i)
					#self.diffusive_boundary(i)

					flux_k += self.calculate_flux(i, previous_r)

				flux.append(flux_k)

			else:
				self.r += self.dt * self.v #Drift

				for i in range(len(self.r)):
					self.check_boundaries(i)
					#self.diffusive_boundary(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T() #Calculate energy before scattering

			scattering_events.append(self.scattering())

			E_subcells_new , N_subcells_new = self.calculate_subcell_T() #Calculate energy after scattering

			delta_E = np.array(E_subcells_new) - np.array(E_subcells) #Account for loss or gain of energy

			self.energy_conservation(delta_E) #Impose energy conservation

			E_subcells_final, N_subcells_final = self.calculate_subcell_T() #Calculate final T

			delta_E_final = np.array(E_subcells_final) - np.array(E_subcells)

			delta_energy.append(np.mean(delta_E_final))
			Energy.append(np.sum(E_subcells_final))
			Phonons.append(np.sum(N_subcells_final))
			Temperatures.append(np.mean(self.subcell_Ts))

			copy_subcells = np.copy(self.subcell_Ts)

			cell_temperatures.append(copy_subcells)

			elapsed_time.append(current_time() - t0)

		#Save last restart
		self.save_restart(k+1)

		if not  os.path.exists(current_dir + '/' + folder_outputs): os.mkdir(current_dir + '/' + folder_outputs)
		os.chdir(current_dir + '/' + folder_outputs)

		flux = np.array(flux) / (self.Ly * self.Lz * self.dt * self.every_flux)

		np.save('Energy.npy', Energy)
		np.save('Phonons.npy', Phonons)
		np.save('Subcell_Ts.npy', cell_temperatures)
		np.save('Temperatures.npy', Temperatures)
		np.save('Scattering_events.npy', scattering_events)
		np.save('Elapsed_time.npy', elapsed_time)
		np.save('Flux.npy', flux)

		f = open('parameters_used.txt', 'w')

		f.write('Lx: ' + str(self.Lx) + '\n')
		f.write('Ly: ' + str(self.Ly) + '\n')
		f.write('Lz: ' + str(self.Lz) + '\n\n')

		f.write('Lx_subcell: ' + str(self.Lx_subcell) + '\n')
		f.write('Ly_subcell: ' + str(self.Ly_subcell) + '\n')
		f.write('Lz_subcell: ' + str(self.Lz_subcell) + '\n\n')

		f.write('T0: ' + str(self.T0) + '\n')
		f.write('Tf: ' + str(self.Tf) + '\n')
		f.write('Ti: ' + str(self.Ti) + '\n\n')

		f.write('t_MAX: ' + str(self.t_MAX) + '\n')
		f.write('dt: ' + str(self.dt) + '\n\n')

		f.write('W: ' + str(self.W) + '\n')
		f.write('Every_flux: ' + str(self.every_flux))

		f.close()

		print('\nSimulation finished!')

	def animation(self):
		self.init_particles()

		def update(t, x, lines):
			k = int(t / self.dt)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.check_boundaries(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T()

			self.scattering()

			E_subcells_new , N_subcells_new = self.calculate_subcell_T()

			delta_E = np.array(E_subcells_new) - np.array(E_subcells)

			self.energy_conservation(delta_E)

			self.calculate_subcell_T()

			lines[0].set_data(x, self.subcell_Ts[:, int(self.Ly / 2), int(self.Lz/2)])
			lines[1].set_data(x, diffussive_T(x, self.T0, self.Tf, self.Lx))
			lines[2].set_data(x, np.linspace(balistic_T(self.T0, self.Tf), balistic_T(self.T0, self.Tf), len(x)))
			lines[3].set_text('Time step %i of %i' % (k, self.Nt))

			return lines

		# Attaching 3D axis to the figure
		fig, ax = plt.subplots()

		x = np.linspace(0, self.Lx, int(round(self.Lx/self.Lx_subcell, 0)))

		lines = [ax.plot(x, self.subcell_Ts[:, int(self.Ly / 2), int(self.Lz/2)], '-o', color='r', label='Temperature')[0], ax.plot(x, diffussive_T(x, self.T0, self.Tf, self.Lx), label='Diffusive')[0],
		ax.plot(x, np.linspace(balistic_T(self.T0, self.Tf), balistic_T(self.T0, self.Tf), len(x)), ls='--', color='k', label='Ballistic')[0], ax.text(0, self.Tf, '', color='k', fontsize=10)]

		ani = FuncAnimation(fig, update, fargs=(x, lines), frames=np.linspace(0, self.t_MAX-self.dt, self.Nt),
		                    blit=True, interval=1, repeat=False)
		#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
		plt.legend(loc = 'upper right')
		plt.show()

	def animation_from_restart(self):

		def update(t, x, lines):
			k = int(t / self.dt)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.check_boundaries(i)
				#self.diffusive_boundary(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T()

			self.scattering()

			E_subcells_new , N_subcells_new = self.calculate_subcell_T()

			delta_E = np.array(E_subcells_new) - np.array(E_subcells)

			self.energy_conservation(delta_E)

			self.calculate_subcell_T()

			lines[0].set_data(x, self.subcell_Ts[:, int(self.Ly / 2), int(self.Lz/2)])
			lines[1].set_data(x, diffussive_T(x, self.T0, self.Tf, self.Lx))
			lines[2].set_data(x, np.linspace(balistic_T(self.T0, self.Tf), balistic_T(self.T0, self.Tf), len(x)))
			lines[3].set_text('Time step %i of %i' % (k, self.Nt))

			return lines

		# Attaching 3D axis to the figure
		fig, ax = plt.subplots()

		x = np.linspace(0, self.Lx, int(round(self.Lx/self.Lx_subcell, 0)))

		lines = [ax.plot(x, self.subcell_Ts[:, int(self.Ly / 2), int(self.Lz/2)], '-o', color='r', label='Temperature')[0], ax.plot(x, diffussive_T(x, self.T0, self.Tf, self.Lx), label='Diffusive')[0],
		ax.plot(x, np.linspace(balistic_T(self.T0, self.Tf), balistic_T(self.T0, self.Tf), len(x)), ls='--', color='k', label='Ballistic')[0], ax.text(0, self.Tf, '', color='k', fontsize=10)]

		ani = FuncAnimation(fig, update, fargs=(x, lines), frames=np.linspace(0, self.t_MAX-self.dt, self.Nt),
		                    blit=True, interval=1, repeat=False)
		#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
		plt.legend(loc = 'upper right')
		plt.show()

		return self.r, self.subcell_Ts

class GrayModel_diffusive_walls(object):
	def __init__(self, type_, Lx=0, Ly=0, Lz=0, Lx_subcell=0, Ly_subcell=0, Lz_subcell=0, T0=0, Tf=0, Ti=0, t_MAX=0, dt=0, 
		W=0, every_flux=0, init_restart = False, folder_restart = 'None'):

		if type_ != 'high' and type_ != 'low':
			raise ValueError('Invalid type argument')

		if init_restart :

			self.read_restart(current_dir + '/' + folder_restart)

			self.type = type_

			if t_MAX != 0: 
				self.t_MAX = t_MAX
				self.Nt = int(round(t_MAX / self.dt, 0))

		else:

			self.Lx = float(Lx) #x length of the box
			self.Ly = float(Ly) #y length of the box
			self.Lz = float(Lz) #z lenght of the box
			self.T0 = float(T0) #Temperature of the initial sub-cell (Boundary)
			self.Tf = float(Tf) #Temperature of the last sub-cell (Boundary)
			self.Ti = float(Ti) #Initial temperature of studied subcells
			self.t_MAX = float(t_MAX) #Maximum simulation time
			self.dt = float(dt) #Step size
			self.W = float(W) #Weighting factor

			self.Nt = int(round(self.t_MAX / self.dt, 0)) #Number of simulation steps/iterations

			self.Lx_subcell = float(Lx_subcell) #x length of each subcell
			self.Ly_subcell = float(Ly_subcell) #x length of each subcell
			self.Lz_subcell = float(Lz_subcell) #x length of each subcell

			self.N_subcells_x = int(round(self.Lx / self.Lx_subcell, 0)) #Number of subcells
			self.N_subcells_y = int(round(self.Ly / self.Ly_subcell, 0))
			self.N_subcells_z = int(round(self.Lz / self.Lz_subcell, 0)) 

			self.every_flux = every_flux

			self.type = type_

			self.r = [] #list of the positions of all the phonons
			self.v = [] #list of the velocities of all the phonons

			self.E = []
			self.N = []
			self.w_avg = []
			self.v_avg = []
			self.C_V = []
			self.MFP = []

			self.scattering_time = []

			self.subcell_Ts = np.zeros((self.N_subcells_x, self.N_subcells_y, self.N_subcells_z))

		self.V_subcell = self.Ly_subcell * self.Lz_subcell * self.Lx_subcell

		#Load arrays
		os.chdir(array_folder)

		#Germanium
		self.N_ge = np.load('N_input.npy')
		self.E_ge = np.load('E_input.npy')
		self.w_ge = np.load('w_input.npy')
		self.v_ge = np.load('v_input.npy')
		self.CV_ge = np.load('CV_input.npy')
		self.MFP_ge = np.load('MFP_input.npy')
		self.Etot_ge = np.load('Etot_input.npy')

		#Temperature array
		self.Ts = np.load('T_input.npy')
		#Account for the different volumes
		self.N_ge *= self.V_subcell 
		self.CV_ge *= self.V_subcell
		self.Etot_ge *= self.V_subcell

		#Maximum energies
		self.E_max_ge = 1.659e-21

	def find_T(self, value, T): 
		'''
		For a given value of temperature returns the position in the T array
		'''
		
		for i in range(len(T)):
			if T[i] >= value:
				return i

	def create_phonons(self, N, subcell_x, subcell_y, subcell_z, T):
		r = np.zeros((N, 3)) #Array of vector positions
		v = np.zeros((N, 3)) #Array of vector velocities

		rx = np.random.random((N,)) * self.Lx_subcell + subcell_x * self.Lx_subcell
		ry = np.random.random((N,)) * self.Ly_subcell + subcell_y * self.Ly_subcell
		rz = np.random.random((N,)) * self.Lz_subcell + subcell_z * self.Lz_subcell

		pos = self.find_T(T, self.Ts)

		for j in range(N):
			r[j][0] = rx[j]
			r[j][1] = ry[j]
			r[j][2] = rz[j]

			self.E.append(self.E_ge[pos])
			self.v_avg.append(self.v_ge[pos])
			self.w_avg.append(self.w_ge[pos])
			self.C_V.append(self.CV_ge[pos])
			self.MFP.append(self.MFP_ge[pos])
			self.scattering_time.append(0.)

		v_polar = np.random.random((N, 2))

		v[:,0] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.cos(v_polar[:,1] * 2 * np.pi)) 
		v[:,1] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.sin(v_polar[:,1] * 2 * np.pi)) 
		v[:,2] = np.cos(np.cos(2*v_polar[:,0]-1)**(-1))

		v *= self.v_ge[pos]

		self.r += list(r)
		self.v += list(v)

	def init_particles(self):

		for i in range(self.N_subcells_x):
			for j in range(self.N_subcells_y):
				for k in range(self.N_subcells_z):

					if i == 0:
						T_i = self.T0
						self.subcell_Ts[i][j][k] = self.T0

					elif i == self.N_subcells_x - 1:
						T_i = self.Tf
						self.subcell_Ts[i][j][k] = self.Tf

					else:
						T_i = self.Ti
						self.subcell_Ts[i][j][k] = self.Ti

					pos = self.find_T(T_i, self.Ts)

					N = int(self.N_ge[pos] / self.W)

					self.create_phonons(N, i, j, k, T_i)

		self.r = np.array(self.r)
		self.v = np.array(self.v)
		self.E = np.array(self.E)
		self.v_avg = np.array(self.v_avg)
		self.w_avg = np.array(self.w_avg)
		self.C_V = np.array(self.C_V)
		self.MFP = np.array(self.MFP)
		self.scattering_time = np.array(self.scattering_time)

		return self.r, self.v, self.E, self.v_avg, self.w_avg, self.C_V, self.MFP

	def diffusive_boundary(self, i):

		y_wall = False
		z_wall = False

		if self.r[i][0] >= self.Lx or self.r[i][0] < 0:
			self.v[i][0] *= -1.

			if self.r[i][0] >= self.Lx:
				self.r[i][0] = self.Lx - 0.01*self.Lx
			else:
				self.r[i][0] = 0

		if self.r[i][1] >= self.Ly or self.r[i][1] < 0:
			y_wall = True

			if self.r[i][1] > self.Ly or self.r[i][1] < 0:
				self.v[i][1] *= -1.

			if self.r[i][1] > self.Ly:
				delta_y = self.r[i][1] - self.Ly
				self.r[i][1] = self.r[i][1] - 2*delta_y
			else:
				delta_y = -self.r[i][1] 
				self.r[i][1] = delta_y

		if self.r[i][2] >= self.Lz or self.r[i][2] < 0:
			z_wall = True

			if self.r[i][2] > self.Lz:
				delta_z = self.r[i][2] - self.Lz
				self.r[i][2] = self.r[i][2] - 2*delta_z
			else:
				delta_z = -self.r[i][2] 
				self.r[i][2] = delta_z

		if y_wall or z_wall :
			if y_wall and not z_wall:
				x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
				y = int(self.N_subcells_y - 1)
				z = int(self.r[i][2] / self.Lz * self.N_subcells_z)
			if not y_wall and z_wall:
				x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
				y = int(self.r[i][1] / self.Lz * self.N_subcells_z)
				z = int(self.Lz - 1)

			if y_wall and z_wall:
				x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
				y = int(self.Ly - 1)
				z = int(self.Lz - 1)

			v_polar = np.random.random((1, 2))

			self.v[i][0] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.cos(v_polar[:,1] * 2 * np.pi)) 
			self.v[i][1] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.sin(v_polar[:,1] * 2 * np.pi)) 
			self.v[i][2] = np.cos(np.cos(2*v_polar[:,0]-1)**(-1))

			current_T = self.subcell_Ts[x][y][z]
			pos = self.find_T(current_T, self.Ts)

			self.v[i] *= self.v_ge[pos]

			self.v_avg[i] = self.v_ge[pos]
			self.w_avg[i] = self.w_ge[pos]
			self.E[i] = self.E_ge[pos]
			self.C_V[i] = self.CV_ge[pos]
			self.MFP[i] = self.MFP_ge[pos]
		
	def match_T(self, value, E, T):
		for i in range(len(E)):
			if E[i] == value:
				return T[i]

			elif E[i] > value: #If we exceed the value, use interpolation
				return T[i] * value /  E[i]

	def calculate_subcell_T(self):

		E_subcells = np.zeros((self.N_subcells_x, self.N_subcells_y, self.N_subcells_z))
		N_subcells = np.zeros((self.N_subcells_x, self.N_subcells_y, self.N_subcells_z))

		for i in range(len(self.r)):
			x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
			y = int(self.r[i][1] / self.Ly * self.N_subcells_y)
			z = int(self.r[i][2] / self.Lz * self.N_subcells_z)

			E_subcells[x][y][z] += self.W * self.E[i]
			N_subcells[x][y][z] += self.W

		if self.type == 'high':

			for i in range(self.N_subcells_x):
				for j in range(self.N_subcells_y):
					for k in range(self.N_subcells_z):

						E_N = E_subcells[i][j][k] / N_subcells[i][j][k]

						self.subcell_Ts[i][j][k] = self.match_T(E_N, self.E_ge, self.Ts)

		elif self.type == 'low':

			for i in range(self.N_subcells_x):
				for j in range(self.N_subcells_y):
					for k in range(self.N_subcells_z):

						self.subcell_Ts[i][j][k] = self.match_T(E_subcells[i][j][k], self.Etot_ge, self.Ts)


		return E_subcells, N_subcells

	def find_subcell(self, i):
		for j in range(1, self.N_subcells - 1):
			if self.r[i][0] >=  j * self.Lx_subcell and self.r[i][0] <= (j + 1) * self.Lx_subcell: #It will be in the j_th subcell
				return j	

	def scattering(self):
		scattering_events = 0

		for i in range(len(self.r)):
			x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
			y = int(self.r[i][1] / self.Ly * self.N_subcells_y)
			z = int(self.r[i][2] / self.Lz * self.N_subcells_z)

			if x < 1 or x > (self.N_subcells_x - 1):
				pass #Avoid scattering for phonons in hot and cold boundary cells

			else:
				prob = 1 - np.exp(-self.v_avg[i] * self.scattering_time[i] / self.MFP[i])
				
				dice = random.uniform(0, 1)

				if prob > dice :#Scattering process

					v_polar = np.random.random((1, 2))

					self.v[i][0] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.cos(v_polar[:,1] * 2 * np.pi)) 
					self.v[i][1] = (np.sin(np.cos(2*v_polar[:,0]-1)**(-1)) * np.sin(v_polar[:,1] * 2 * np.pi)) 
					self.v[i][2] = np.cos(np.cos(2*v_polar[:,0]-1)**(-1))

					current_T = self.subcell_Ts[x][y][z]
					pos = self.find_T(current_T, self.Ts)

					self.v[i] *= self.v_ge[pos]

					self.v_avg[i] = self.v_ge[pos]
					self.w_avg[i] = self.w_ge[pos]
					self.E[i] = self.E_ge[pos]
					self.C_V[i] = self.CV_ge[pos]
					self.MFP[i] = self.MFP_ge[pos]

					self.scattering_time[i] = 0. #Re-init scattering time

					scattering_events += self.W

				else:
					self.scattering_time[i] += self.dt #Account for the scattering time

		return scattering_events

	def energy_conservation(self, delta_E):
		for i in range(1, self.N_subcells_x - 1):
			for j in range(self.N_subcells_y):
				for k in range(self.N_subcells_z):

					current_energy = delta_E[i][j][k]

					if current_energy > self.E_max_ge:

						while current_energy > self.E_max_ge: #Delete phonons

							for l in range(len(self.r)):

								x = int((self.r[l][0] / self.Lx) * self.N_subcells_x)
								y = int((self.r[l][1] / self.Lx) * self.N_subcells_y)
								z = int((self.r[l][2] / self.Lx) * self.N_subcells_z)

								if x == i and y == j and z == k : #is in the i_th subcell

									current_energy -= self.E[l] * self.W

									self.r = np.delete(self.r, l, 0)
									self.v = np.delete(self.v, l, 0)
									self.E = np.delete(self.E, l, 0)
									self.v_avg = np.delete(self.v_avg, l, 0)
									self.w_avg = np.delete(self.w_avg, l, 0)
									self.C_V = np.delete(self.C_V, l, 0)
									self.MFP = np.delete(self.MFP, l, 0)
									self.scattering_time = np.delete(self.scattering_time, l, 0)

									break


					if -delta_E[i][j][k] > self.E_max_ge: #Production of phonons

						while -current_energy > self.E_max_ge:

							T = self.subcell_Ts[i][j][k]
							pos_T = self.find_T(T, self.Ts)

							E_phonon_T = self.E_ge[pos_T] #Energy per phonon for this subcell T	

							self.r = list(self.r)
							self.v = list(self.v)
							self.v_avg = list(self.v_avg)
							self.w_avg = list(self.w_avg)
							self.E = list(self.E)
							self.C_V = list(self.C_V)
							self.MFP = list(self.MFP)
							self.scattering_time = list(self.scattering_time)

							self.create_phonons(1, i, j, k, T)

							self.r = np.array(self.r)
							self.v = np.array(self.v)
							self.v_avg = np.array(self.v_avg)
							self.w_avg = np.array(self.w_avg)
							self.E = np.array(self.E)
							self.C_V = np.array(self.C_V)
							self.MFP = np.array(self.MFP)
							self.scattering_time = np.array(self.scattering_time)

							current_energy += E_phonon_T * self.W

	def re_init_boundary(self): #Eliminar tots i posar tots nous
		pos_T0 = self.find_T(self.T0, self.Ts)
		pos_Tf = self.find_T(self.Tf, self.Ts)

		N_0 = int(round(self.N_ge[pos_T0] / self.W, 0))
		N_f = int(round(self.N_ge[pos_Tf] / self.W, 0))

		total_indexs = []

		#Delete all the phonons in boundary subcells
		for i in range(len(self.r)):
			if self.r[i][0] <= self.Lx_subcell: #Subcell with T0 boundary
				total_indexs.append(i)

			elif self.r[i][0] >= (self.N_subcells_x - 1) * self.Lx_subcell: #Subcell with Tf boundary
				total_indexs.append(i)

		self.r = np.delete(self.r, total_indexs, 0)
		self.v = np.delete(self.v, total_indexs, 0)
		self.E = np.delete(self.E, total_indexs, 0)
		self.v_avg = np.delete(self.v_avg, total_indexs, 0)
		self.w_avg = np.delete(self.w_avg, total_indexs, 0)
		self.C_V = np.delete(self.C_V, total_indexs, 0)
		self.MFP = np.delete(self.MFP, total_indexs, 0)
		self.scattering_time = np.delete(self.scattering_time, total_indexs, 0)

		#Create the new phonons
		self.r = list(self.r)
		self.v = list(self.v)
		self.E = list(self.E)
		self.v_avg = list(self.v_avg)
		self.w_avg = list(self.w_avg)
		self.C_V = list(self.C_V)
		self.MFP = list(self.MFP)
		self.scattering_time = list(self.scattering_time)

		for j in range(self.N_subcells_y):
			for k in range(self.N_subcells_z):

				self.create_phonons(N_0, 0, j, k, self.T0)
				self.create_phonons(N_f, self.N_subcells_x - 1, j, k, self.Tf)

		self.r = np.array(self.r)
		self.v = np.array(self.v)
		self.E = np.array(self.E)
		self.v_avg = np.array(self.v_avg)
		self.w_avg = np.array(self.w_avg)
		self.C_V = np.array(self.C_V)
		self.MFP = np.array(self.MFP)
		self.scattering_time = np.array(self.scattering_time)

	def calculate_flux(self, i, r_previous):
		'''
			Calculates the flux in the yz plane in the middle of Lx lenght
		'''

		if self.r[i][0] > self.Lx/2 and r_previous[i][0] < self.Lx/2:

			return self.E[i] * self.W

		elif self.r[i][0] < self.Lx/2 and r_previous[i][0] > self.Lx/2:
			return -self.E[i] * self.W

		else:
			return 0

	def save_restart(self, nt):

		os.chdir(current_dir)

		if not os.path.exists('restart_%i' % nt): os.mkdir('restart_%i' % nt)
		os.chdir('restart_%i' % nt)

		np.save('r.npy', self.r)
		np.save('v.npy', self.v)

		np.save('E.npy', self.E)
		np.save('N.npy', self.N)
		np.save('w_avg.npy', self.w_avg)
		np.save('v_avg.npy', self.v_avg)
		np.save('C_V.npy', self.C_V)
		np.save('MFP.npy', self.MFP)

		np.save('scattering_time.npy', self.scattering_time)

		np.save('subcell_Ts.npy', self.subcell_Ts)

		f = open('parameters_used.txt', 'w')

		f.write('Lx: ' + str(self.Lx) + '\n')
		f.write('Ly: ' + str(self.Ly) + '\n')
		f.write('Lz: ' + str(self.Lz) + '\n\n')

		f.write('Lx_subcell: ' + str(self.Lx_subcell) + '\n')
		f.write('Ly_subcell: ' + str(self.Ly_subcell) + '\n')
		f.write('Lz_subcell: ' + str(self.Lz_subcell) + '\n\n')

		f.write('T0: ' + str(self.T0) + '\n')
		f.write('Tf: ' + str(self.Tf) + '\n')
		f.write('Ti: ' + str(self.Ti) + '\n\n')

		f.write('t_MAX: ' + str(self.t_MAX) + '\n')
		f.write('dt: ' + str(self.dt) + '\n\n')

		f.write('W: ' + str(self.W) + '\n')
		f.write('Every_flux: ' + str(self.every_flux))

		f.close()

	def get_parameters(self):
		f = open('parameters_used.txt', 'r')

		i = 0

		for line in f:

			try:

				cols = line.split()

				if len(cols) > 0:
					value = float(cols[1])

				if i == 0:
					self.Lx = value

				elif i == 1:
					self.Ly = value

				elif i == 2:
					self.Lz = value

				elif i == 4:
					self.Lx_subcell = value

				elif i == 5:
					self.Ly_subcell = value

				elif i == 6:
					self.Lz_subcell = value

				elif i == 8:
					self.T0 = value

				elif i == 9:
					self.Tf = value

				elif i == 10:
					self.Ti = value

				elif i == 12:
					self.t_MAX = value

				elif i == 13:
					self.dt = value

				elif i == 15:
					self.W = value

				elif i == 16:
					self.every_flux = value

				i += 1

			except:
				pass

	def read_restart(self, folder):

		os.chdir(folder)

		self.r = np.load('r.npy')
		self.v = np.load('v.npy')

		self.E = np.load('E.npy')
		self.N = np.load('N.npy')
		self.w_avg = np.load('w_avg.npy')
		self.v_avg = np.load('v_avg.npy')
		self.C_V = np.load('C_V.npy')
		self.MFP = np.load('MFP.npy')

		self.scattering_time = np.load('scattering_time.npy')

		self.subcell_Ts = np.load('subcell_Ts.npy')

		self.get_parameters()

		self.Nt = int(round(self.t_MAX / self.dt, 0)) #Number of simulation steps/iterations

		self.N_subcells_x = int(round(self.Lx / self.Lx_subcell, 0)) #Number of subcells
		self.N_subcells_y = int(round(self.Ly / self.Ly_subcell, 0))
		self.N_subcells_z = int(round(self.Lz / self.Lz_subcell, 0)) 

		os.chdir(current_dir)

	def simulation(self, every_restart=100, folder_outputs='OUTPUTS'):
		os.chdir(current_dir)

		self.init_particles()

		Energy = []
		Phonons = []
		Temperatures = []
		delta_energy = []
		scattering_events = []
		cell_temperatures = []
		elapsed_time = []

		flux = []

		t0 = current_time()

		for k in range(self.Nt):
			print('Timestep:', k, 'of', self.Nt, '(%.2f)' % (100 * k/self.Nt), '%')

			if k % every_restart == 0:

				#Save configuration actual properties
				self.save_restart(k)

				#Save outputs untill this moment (Inside the restart folder)
				flux_save = np.array(flux) / (self.Ly * self.Lz * self.dt)

				np.save('Energy.npy', Energy)
				np.save('Phonons.npy', Phonons)
				np.save('Subcell_Ts.npy', cell_temperatures)
				np.save('Temperatures.npy', Temperatures)
				np.save('Scattering_events.npy', scattering_events)
				np.save('Elapsed_time.npy', elapsed_time)
				np.save('Flux.npy', flux_save)

				os.chdir(current_dir) #Go back to the principal directory

			if k % int(self.every_flux) == 0:

				previous_r = np.copy(self.r) #Save the previous positions to calculate the flux

				self.r += self.dt * self.v #Drift

				flux_k = 0

				for i in range(len(self.r)):
					self.diffusive_boundary(i)

					flux_k += self.calculate_flux(i, previous_r)

				flux.append(flux_k)

			else:
				self.r += self.dt * self.v #Drift

				for i in range(len(self.r)):
					self.diffusive_boundary(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T() #Calculate energy before scattering

			scattering_events.append(self.scattering())

			E_subcells_new , N_subcells_new = self.calculate_subcell_T() #Calculate energy after scattering

			delta_E = np.array(E_subcells_new) - np.array(E_subcells) #Account for loss or gain of energy

			self.energy_conservation(delta_E) #Impose energy conservation

			E_subcells_final, N_subcells_final = self.calculate_subcell_T() #Calculate final T

			delta_E_final = np.array(E_subcells_final) - np.array(E_subcells)

			delta_energy.append(np.mean(delta_E_final))
			Energy.append(np.sum(E_subcells_final))
			Phonons.append(np.sum(N_subcells_final))
			Temperatures.append(np.mean(self.subcell_Ts))

			copy_subcells = np.copy(self.subcell_Ts)

			cell_temperatures.append(copy_subcells)

			elapsed_time.append(current_time() - t0)

		#Save last restart
		self.save_restart(k+1)

		if not  os.path.exists(current_dir + '/' + folder_outputs): os.mkdir(current_dir + '/' + folder_outputs)
		os.chdir(current_dir + '/' + folder_outputs)

		flux = np.array(flux) / (self.Ly * self.Lz * self.dt)

		np.save('Energy.npy', Energy)
		np.save('Phonons.npy', Phonons)
		np.save('Subcell_Ts.npy', cell_temperatures)
		np.save('Temperatures.npy', Temperatures)
		np.save('Scattering_events.npy', scattering_events)
		np.save('Elapsed_time.npy', elapsed_time)
		np.save('Flux.npy', flux)

		f = open('parameters_used.txt', 'w')

		f.write('Lx: ' + str(self.Lx) + '\n')
		f.write('Ly: ' + str(self.Ly) + '\n')
		f.write('Lz: ' + str(self.Lz) + '\n\n')

		f.write('Lx_subcell: ' + str(self.Lx_subcell) + '\n')
		f.write('Ly_subcell: ' + str(self.Ly_subcell) + '\n')
		f.write('Lz_subcell: ' + str(self.Lz_subcell) + '\n\n')

		f.write('T0: ' + str(self.T0) + '\n')
		f.write('Tf: ' + str(self.Tf) + '\n')
		f.write('Ti: ' + str(self.Ti) + '\n\n')

		f.write('t_MAX: ' + str(self.t_MAX) + '\n')
		f.write('dt: ' + str(self.dt) + '\n\n')

		f.write('W: ' + str(self.W) + '\n')
		f.write('Every_flux: ' + str(self.every_flux))

		f.close()

		print('\nSimulation finished!')

	def simulation_from_restart(self, every_restart=100, folder_outputs='OUTPUTS'):

		Energy = []
		Phonons = []
		Temperatures = []
		delta_energy = []
		scattering_events = []
		cell_temperatures = []
		elapsed_time = []

		flux = []

		t0 = current_time()

		for k in range(self.Nt):
			print('Timestep:', k, 'of', self.Nt, '(%.2f)' % (100 * k/self.Nt), '%')

			if k % every_restart == 0:

				self.save_restart(k)

				#Save outputs untill this moment (Inside the restart folder)
				flux_save = np.array(flux) / (self.Ly * self.Lz * self.dt * self.every_flux)

				np.save('Energy.npy', Energy)
				np.save('Phonons.npy', Phonons)
				np.save('Subcell_Ts.npy', cell_temperatures)
				np.save('Temperatures.npy', Temperatures)
				np.save('Scattering_events.npy', scattering_events)
				np.save('Elapsed_time.npy', elapsed_time)
				np.save('Flux.npy', flux_save)

				os.chdir(current_dir) #Go back to the principal directory

			if k % int(self.every_flux) == 0:

				previous_r = np.copy(self.r) #Save the previous positions to calculate the flux

				self.r += self.dt * self.v #Drift

				flux_k = 0

				for i in range(len(self.r)):
					self.diffusive_boundary(i)

					flux_k += self.calculate_flux(i, previous_r)

				flux.append(flux_k)

			else:
				self.r += self.dt * self.v #Drift

				for i in range(len(self.r)):
					self.diffusive_boundary(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T() #Calculate energy before scattering

			scattering_events.append(self.scattering())

			E_subcells_new , N_subcells_new = self.calculate_subcell_T() #Calculate energy after scattering

			delta_E = np.array(E_subcells_new) - np.array(E_subcells) #Account for loss or gain of energy

			self.energy_conservation(delta_E) #Impose energy conservation

			E_subcells_final, N_subcells_final = self.calculate_subcell_T() #Calculate final T

			delta_E_final = np.array(E_subcells_final) - np.array(E_subcells)

			delta_energy.append(np.mean(delta_E_final))
			Energy.append(np.sum(E_subcells_final))
			Phonons.append(np.sum(N_subcells_final))
			Temperatures.append(np.mean(self.subcell_Ts))

			copy_subcells = np.copy(self.subcell_Ts)

			cell_temperatures.append(copy_subcells)

			elapsed_time.append(current_time() - t0)

		#Save last restart
		self.save_restart(k+1)

		if not  os.path.exists(current_dir + '/' + folder_outputs): os.mkdir(current_dir + '/' + folder_outputs)
		os.chdir(current_dir + '/' + folder_outputs)

		flux = np.array(flux) / (self.Ly * self.Lz * self.dt * self.every_flux)

		np.save('Energy.npy', Energy)
		np.save('Phonons.npy', Phonons)
		np.save('Subcell_Ts.npy', cell_temperatures)
		np.save('Temperatures.npy', Temperatures)
		np.save('Scattering_events.npy', scattering_events)
		np.save('Elapsed_time.npy', elapsed_time)
		np.save('Flux.npy', flux)

		f = open('parameters_used.txt', 'w')

		f.write('Lx: ' + str(self.Lx) + '\n')
		f.write('Ly: ' + str(self.Ly) + '\n')
		f.write('Lz: ' + str(self.Lz) + '\n\n')

		f.write('Lx_subcell: ' + str(self.Lx_subcell) + '\n')
		f.write('Ly_subcell: ' + str(self.Ly_subcell) + '\n')
		f.write('Lz_subcell: ' + str(self.Lz_subcell) + '\n\n')

		f.write('T0: ' + str(self.T0) + '\n')
		f.write('Tf: ' + str(self.Tf) + '\n')
		f.write('Ti: ' + str(self.Ti) + '\n\n')

		f.write('t_MAX: ' + str(self.t_MAX) + '\n')
		f.write('dt: ' + str(self.dt) + '\n\n')

		f.write('W: ' + str(self.W) + '\n')
		f.write('Every_flux: ' + str(self.every_flux))

		f.close()

		print('\nSimulation finished!')

	def animation(self):
		self.init_particles()

		def update(t, x, lines):
			k = int(t / self.dt)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.diffusive_boundary(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T()

			self.scattering()

			E_subcells_new , N_subcells_new = self.calculate_subcell_T()

			delta_E = np.array(E_subcells_new) - np.array(E_subcells)

			self.energy_conservation(delta_E)

			self.calculate_subcell_T()

			lines[0].set_data(x, self.subcell_Ts[:, int(self.Ly / 2), int(self.Lz/2)])
			lines[1].set_data(x, diffussive_T(x, self.T0, self.Tf, self.Lx))
			lines[2].set_data(x, np.linspace(balistic_T(self.T0, self.Tf), balistic_T(self.T0, self.Tf), len(x)))
			lines[3].set_text('Time step %i of %i' % (k, self.Nt))

			return lines

		# Attaching 3D axis to the figure
		fig, ax = plt.subplots()

		x = np.linspace(0, self.Lx, int(round(self.Lx/self.Lx_subcell, 0)))

		lines = [ax.plot(x, self.subcell_Ts[:, int(self.Ly / 2), int(self.Lz/2)], '-o', color='r', label='Temperature')[0], ax.plot(x, diffussive_T(x, self.T0, self.Tf, self.Lx), label='Diffusive')[0],
		ax.plot(x, np.linspace(balistic_T(self.T0, self.Tf), balistic_T(self.T0, self.Tf), len(x)), ls='--', color='k', label='Ballistic')[0], ax.text(0, self.Tf, '', color='k', fontsize=10)]

		ani = FuncAnimation(fig, update, fargs=(x, lines), frames=np.linspace(0, self.t_MAX-self.dt, self.Nt),
		                    blit=True, interval=1, repeat=False)
		#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
		plt.legend(loc = 'upper right')
		plt.show()

		return self.r, self.subcell_Ts

	def animation_from_restart(self):

		def update(t, x, lines):
			k = int(t / self.dt)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.diffusive_boundary(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T()

			self.scattering()

			E_subcells_new , N_subcells_new = self.calculate_subcell_T()

			delta_E = np.array(E_subcells_new) - np.array(E_subcells)

			self.energy_conservation(delta_E)

			self.calculate_subcell_T()

			lines[0].set_data(x, self.subcell_Ts[:, int(self.Ly / 2), int(self.Lz/2)])
			lines[1].set_data(x, diffussive_T(x, self.T0, self.Tf, self.Lx))
			lines[2].set_data(x, np.linspace(balistic_T(self.T0, self.Tf), balistic_T(self.T0, self.Tf), len(x)))
			lines[3].set_text('Time step %i of %i' % (k, self.Nt))

			return lines

		# Attaching 3D axis to the figure
		fig, ax = plt.subplots()

		x = np.linspace(0, self.Lx, int(round(self.Lx/self.Lx_subcell, 0)))

		lines = [ax.plot(x, self.subcell_Ts[:, int(self.Ly / 2), int(self.Lz/2)], '-o', color='r', label='Temperature')[0], ax.plot(x, diffussive_T(x, self.T0, self.Tf, self.Lx), label='Diffusive')[0],
		ax.plot(x, np.linspace(balistic_T(self.T0, self.Tf), balistic_T(self.T0, self.Tf), len(x)), ls='--', color='k', label='Ballistic')[0], ax.text(0, self.Tf, '', color='k', fontsize=10)]

		ani = FuncAnimation(fig, update, fargs=(x, lines), frames=np.linspace(0, self.t_MAX-self.dt, self.Nt),
		                    blit=True, interval=1, repeat=False)
		#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
		plt.legend(loc = 'upper right')
		plt.show()

		return self.r, self.subcell_Ts
