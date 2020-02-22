# Nanoscale Heat Transport - Gray Model
Python little library to simulate nanoscale hat transport from the Gray Model approach. It has been used to study nanoscale heat transport in the following final degree project: https://github.com/agimenezromero/NHT-Gray-Model/tree/master/Final%20Degree%20Project

# Overview
The aim of this program is to simulate thermal transport at nanoscale. The method used is known as the Gray Model, which considers that phonon properties (shuch as energy, average velocity, average frequency...) are based only on local sub-cell temperature. 

Two main simulation classes have been implemented:

1. `GrayModel` : Considers specular reflection with the domain walls.
2. `GrayModel_diffusive_walls` : Considers diffusive reflection with the domain walls.

More information about the model can be found in the [final degree project](https://github.com/agimenezromero/NHT-Gray-Model/tree/master/Final%20Degree%20Project). 

Table of contents
=================

<!--ts-->
   * [Overview](#overview)
   * [Table of contents](#table-of-contents)
   * [Requeriments](#requeriments)
   * [Documentation](#documentation)
       - [Creating input arrays](#creating-input-arrays)
       - [Simulation classes initialisation](#simulation-classes-initialisation)
       - [Running simulations](#running-simulations)
   * [Examples](#examples)
       - [Create input arrays](#create-input-arrays)
       - [New simulation](#new-simulation)
       - [Simulation from restart](#simulation-from-restart)
       - [Animation](#animation)
       - [Animation from restart](#animation-from-restart)
   * [Running example scripts](#running-example-scripts)
   * [Authors](#authors)
   * [License](#license)
<!--te-->

# Requeriments
Python 3 installed with the following libraries
- NumPy
- Matplotlib
- SciPy

# Documentation
## Creating input arrays
First of all the input arrays dependent on temperature need to be created. To do so the `ThermalProperties` class has been developed. For the study in the final degree project germanium has been simulated, so a simple function have been implemented to create and storage the corresponding arrays easily: 

* `save_arrays_germanium(init_T, final_T, n)` 

  - `ìnit_T` (float) - Initial temperature for the computed properties.
  - `final_T` (float) - Final temperature for the computed properties.
  - `n` (int) - Number of points between initial and final temperatures.
  
 Moreover a function to create the input arrays for silicon is also available: `save_arrays_silicon(init_T, final_T, n)`.
 
 Once the arrays are created, they are stored in an automatically created folder named `Input_arrays`. This step only needs to be made one time, unless we decide to change the material (Germanium for Silicon for example). So this program just support one kind of element at a time, which corresponds to the one storage in the `Input_arrays` folder.
 
## Simulation classes initialisation

To initialise both of the available simulation classes (`GrayModel`, `GrayModel_diffusive_walls`) the following parameters must be passed in:

- `type_` (string) - Temperature regime ('high' or 'low').

- `Lx` (float) - Domain length (x-direction).
- `Ly` (float) - Domain width (y-direction).
- `Lz` (float) - Domain height (z-direction).

- `Lx_subcell` (float) - Subcell length (x-direction).
- `Ly_subcell` (float) - Subcell width (y-direction).
- `Lz_subcell` (float) - Subcell height (z-direction).

- `T0` (float) - Hot boundary (first cell or array of cells).
- `Tf` (float) - Cold boundary (last cell or array of cells).
- `Ti` (float) - Other cells initial temperature.

- `t_MAX` (float) - Maximum simulation time.
- `dt` (float) - Integration time step.

- `W` (float) - Weighting factor.
- `every_flux` (int) - Flux calculation period in frame units.

- `init_restart` : (bool, optional) - Set to true to initialise a simulation from a restart. 
- `folder_restart` : (string, optional) - Specify the restart folder to start with.

## Running simulations

There are 4 ways to run simulations for each of the classes previously mentioned: `simulation(every_restart, folder_outputs)`, `simulation_from_restart(every_restart, folder_outputs)`, `animation()`, `animation_from_restart()`.

- `every_restart` (int, optional) - Restart writting period in frame units.
- `folder_outputs` : (string, optional) - Folder name to save output files.

## Output files

- `Energy.npy`: Binary file storing a numpy ndarray of the total energy of the system for each time step. So the array has size=(Nt,) where `Nt` is the total number of frames or iterations. 
- `Phonons.npy`: Binary file storing a numpy ndarray of the total number of phonons in the whole domain for each time step with size (Nt,).
- `Temperatures.npy`: Binary file storing a numpy ndarray of the total average temperature of the system for each time step with size (Nt,).
- `Scattering events.npy`: Binary file storing a numpy ndarray of the number of scattering events in the whole system every time step with size (Nt,).
- `Elapsed_time.npy`: Binary file storing a numpy ndarray of the time elapsed untill the start of the simulation for each time step with size (Nt,).

- `Subcell_Ts.npy`: Binary file storing a numpy ndarray of the temperature of each subcell for each time step. Thus, it will have size (Nt, (Nx, Ny, Nz)) where Nx, Ny and Nz are the number of cells in each direction.

This files will be stored in the outputs folder. Moreover, they will be also saved in every restart folder with the values computed untill the moment of writting each restart.

# Examples

## Create input arrays
To create and storage the input arrays needed by the simulation software the `ThermalProperties` class has been developed. Then, two functions have been built to use it to create the input arrays for germanium and silicon. For the final degree project only germanium has been studied, so the code below will create the arrays necessary to simulate bulk germanium. 

```python
from GrayModelLibrary import *

init_T = 1
final_T = 500
n = 10000

save_arrays_germanium(init_T, final_T, n)
```

With this simple code the input arrays for these materials will be created  and stored in the automatically created `Input_arrays` folder.

## New simulation

To perform a new simulation, all the system parameters must be initialised.

```python
from GrayModelLibrary import *

Lx = 10e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 0.5e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 11.88
Tf = 3
Ti = 5

t_MAX = 10e-9
dt = 0.1e-12

W = 0.05
every_flux = 5

#Optional: Default are 100 and 'OUTPUTS'
every_restart = 1000
folder_outputs = 'EXAMPLE_OUTPUTS'

gray_model = GrayModel('low', Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)
gray_model.simulation(every_restart, folder_outputs)
```

## Simulation from restart

To run a simulation from a restart only the `init_restart` and `restart_folder` are needed to initialise the class. If the class is initialized with some of the other arguments of the exemple above they will be simply ignored. (We set the regime temperature to 'high' here for consistency with the restart example script in the Example folder).

```python
from GrayModelLibrary import *

gray_model = GrayModel('high', init_restart=True, folder_restart='restart_example)
gray_model.simulation_from_restart()
```
However optional arguments can be passed to the `simulation_from_restart` function

```python
from GrayModelLibrary import *

gray_model = GrayModel('high', init_restart=True, folder_restart='restart_example')
gray_model.simulation_from_restart(every_restart=1000, folder='EXAMPLE_OUTPUTS')
```
## Animation
A real time animation of the sub-cell temperature evolution is also available, which makes all the calculation needed *on the fly*. Runing the animation for a new system is as simple as running a simulation.

```python
from GrayModelLibrary import *

Lx = 10e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 0.5e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 11.88
Tf = 3
Ti = 5

t_MAX = 10e-9
dt = 0.1e-12

W = 0.05
every_flux = 5

gray_model = GrayModel('low', Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)
gray_model.animation()
```

## Animation from restart
And one can also start the animation from an existing restart

```python
from GrayModelLibrary import *

gray_model = GrayModelLybrary('high', init_restart=True, folder_restart='restart_example')
gray_model.animation_from_restart()
```

## Diffusive boundary walls
To considere diffusive boundary walls just call the other implemented class named `GrayModel_diffusive_walls` which is used in the same way as the `GrayModel` class. Anyway a single example is presented

```python
from GrayModelLibrary import *

Lx = 10e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 0.5e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 11.88
Tf = 3
Ti = 5

t_MAX = 10e-9
dt = 0.1e-12

W = 0.05
every_flux = 5

#Optional: Default are 100 and 'OUTPUTS'
every_restart = 1000
folder_outputs = 'EXAMPLE_OUTPUTS'

gray_model = GrayModel_diffusive_walls('low', Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)
gray_model.simulation(every_restart, folder_outputs)
```

# Running example scripts
Some of the examples above reported have been included in the [Examples](https://github.com/agimenezromero/NHT-Gray-Model/tree/master/Examples) folder as python .py files. To run them follow the next steps (after dowloading the project)

- Go to the Example folder and open a console in the current folder.
- Type `python3 desired_file_name.py` to run the desired script. For example `python3 Example_low_T_ballistic.py`
- **Input files not need to be created as there have been already included** (see Input_arrays folder)

If you run one of the simulation scripts restart folders and, at the end of the simulation, an output folder will be created. The output folder contains compressed .npy arrays of all the relevant magnitudes calculated in the simulation. These arrays can be loaded using numpy and ploted with matplotlib easily.

A simple plot script (`GrayModelPlots.py`) has been included in the project, which implements some functions to plot the basic magnitudes computed during the simulation. 

# Authors
* **A. Giménez-Romero**

# License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/agimenezromero/NHT-Gray-Model/blob/master/LICENSE) file for details

