#!/usr/bin/env python
## Function: plot "compute" results in GPUMD for NEMD simulations
## Usage: python3 plot_compute_ave.py
## Edition: Peng-Hu Du, 2024.09.28

## Import relevant functions
from ase.io import read, write
from thermo.gpumd.data import load_compute, load_shc
from pylab import *
import numpy as np

## Figure setting
axe_width = 2.5
font_family = 'Arial'
font_size = 30

matplotlib.rc('font', family=font_family, size=font_size)
matplotlib.rc('axes', linewidth=axe_width)

def set_ticks(ax_list):
    tick_length_major = 6
    tick_width = 2.5
    tick_length_minor = 0
    
    for ax in ax_list:
        ax.tick_params(which='major', length=tick_length_major, width=tick_width)
        ax.tick_params(which='minor', length=tick_length_minor, width=tick_width)
        ax.tick_params(which='both', axis='both', direction='in', pad=5, right=True, top=True)

## Data process
# NEMD parameters
temp = 300  # averaged temperature
interval = 50  # temperature interval for heat source/sink
direction = '+x'

# size parameters
length = 59.028 + 0.1  # side length of ST
height = 51.120 + 0.1  # height of ST
prop = 70/100  # proportion of heat current

# load "compute.out" file
compute_dirs = ['../{}_{}_+x_'.format(temp, interval) + str(i) for i in range(0, 1) ]

compute_ave = {}
compute_ave['temp_ave'] = []
compute_ave['heat_current'] = []
compute_ave['kappa'] = []
for compute in compute_dirs:
    compute = load_compute(quantities=['T'], directory=compute, filename='compute.out')
    print(compute.keys())

    T = compute['T']
    Ein = compute['Ein']
    Eout = compute['Eout']

    # compute time
    ndata = T.shape[0]
    ts = 0.0005  # in ps
    Ns = 1000  # sample interval, unit is time step
    t = ts * Ns * np.arange(1, ndata+1) / 1000  # in ns

    # compute temperature gradient
    temp_ave = mean(T[int(ndata*1/2):, 1:], axis=0)
    print('Average temperature: ', temp_ave)
    compute_ave['temp_ave'].append(temp_ave)

    deltaT = temp_ave[0] - temp_ave[-1]  # in K
    print('Temperature difference:', deltaT)
    gradT = deltaT / (height * prop * 10**(-10)) # in K/m
    print('Temperature gradient:', gradT)

    # compute heat flux
    #Q1 = (Ein[int(ndata*4/5)] - Ein[-1]) / (ndata/5) / ts / Ns  # energy transfer rate on one side in eV/ps
    Q1 = np.polyfit( t[int(ndata*1/2):]*1000, Ein[int(ndata*1/2):], 1 )[0]
    #Q2 = (Eout[-1] - Eout[int(ndata*4/5)]) / (ndata/5) / ts / Ns
    Q2 = np.polyfit( t[int(ndata*1/2):]*1000, Eout[int(ndata*1/2):], 1 )[0]
    print('Energy transfer rate on both sides:', Q1, Q2)
    Q = mean( [abs(Q1), abs(Q2)] )  # averaged energy transfer rate in eV/ps
    print('Averaged energy transfer rate in eV/ps:', Q)
    compute_ave['heat_current'].append(Q)
    Q = mean( [abs(Q1), abs(Q2)] ) * 1.602177 * 10**(-19) / (10**(-12)) # averaged energy transfer rate in W=J/s
    print('Averaged energy transfer rate in J/s:', Q)

    area = length * 3.460 * (10**(-20))  # use the area in the middle on ST, in m^2
    J = Q / area
    print('Heat flux:', J)
 
    # compute kappa
    kappa = J / gradT  # in W/m/K
    print('Thermal conductivity:', kappa)
    compute_ave['kappa'].append(kappa)

compute_ave['temp_ave'] = np.array(compute_ave['temp_ave'])
compute_ave['heat_current'] = np.array(compute_ave['heat_current'])
compute_ave['kappa'] = np.array(compute_ave['kappa'])
print('Average temperature:\n', compute_ave['temp_ave'])
print(np.mean(compute_ave['temp_ave'], axis=0))
print('Heat current:\n', compute_ave['heat_current'])
print('Average heat current:\n', np.mean(compute_ave['heat_current']))
print('Kappa:\n', compute_ave['kappa'])
print('Average kappa:\n', np.mean(compute_ave['kappa']))


## Plot NEMD results
figure(figsize=(10.72, 8.205), dpi=100)  # unit in inch, 1 inch = 2.54 cm.
axes((0.1789, 0.1661, 0.6819, 0.7179))

# plot temperature profile
#subplot(1, 2, 1)
group_idx = np.arange(1, 4, 1)  # group 1~5
plot(group_idx, mean(compute_ave['temp_ave'][:, :], axis=0), linewidth=4, color='#0000CC', marker='o', markersize=17)
#legend(loc='best', frameon=False)
#title('(a)',  x=-0.1, y=1.05)

ylabel(r'Temperature (K)', labelpad=3)
xlabel(r'Position', labelpad=3)
xlim([min(group_idx), max(group_idx)])
ylim([250, 350])

set_ticks([gca()])   # set uniform axes.
gca().set_xticks(group_idx)
#gca().set_xticklabels(['K', r'$\Gamma$', 'L', 'W', 'X'])
gca().set_yticks(np.arange(250, 351, 25))
#gca().set_yticklabels([])

'''
# plot energies accumulated in the thermostats
subplot(1, 2, 2)
plot(t, Ein/1000, linewidth=4, color='#0000CC')
plot(t, Eout/1000, linewidth=4, linestyle='--', color='#CC0000')
#legend(loc='best', frameon=False)
title('(b)',  x=-0.1, y=1.05)

ylabel(r'Heat (keV)', labelpad=3)
xlabel(r'Time (ns)', labelpad=3)
xlim([0, max(t)])
#ylim([-0.10, 0.15])

set_ticks([gca()])   # set uniform axes.
gca().set_xticks(np.linspace(0, max(t), 6))
#gca().set_xticklabels(['K', r'$\Gamma$', 'L', 'W', 'X'])
#gca().set_yticks(np.linspace(-0.10, 0.15, 6))
#gca().set_yticklabels([])
'''

#tight_layout()
savefig('temp_current_nemd_{}K_{}_{}_ave.tif'.format(temp, interval, direction), dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
#show()
