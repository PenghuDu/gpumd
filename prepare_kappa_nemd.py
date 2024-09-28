#!/usr/bin/env python
## Function: prepare inputs to use NEMD method to calculate kappa.
## Usage: python3 prepare_kappa_nemd.py
## Edition: Peng-Hu Du, 2024.1.4

## Import relevant functions
# The inputs/outputs for gpumd executable are processed using the ASE and thermo package.

from ase.io import read, write
from thermo.gpumd.preproc import add_group_by_position, add_basis, repeat
import numpy as np


## Generate model.xyz file:
# read unit cell in POSCAR
unit_cell = read('ST4_gra_2_new.xyz', format='xyz')
length = 59.028+0.1
height = 51.120+0.1
unit_cell.cell = [ [height, 0, 0], [0, length, 0], [0, 0, 3.35] ]
add_basis(unit_cell)
print('Unit cell:', unit_cell)

# transform unit cell to supercell
super_cell = repeat(unit_cell, [1, 1, 1]) # construct simulation box
# Invert the supercell
#super_cell.euler_rotate( theta=180,  center='COU' )  # rotate atoms via Euler angles (phi, theta, psi) in degrees; center: the point to rotate about, 'COU' to select center of cell; thets: rotation around the x axis;
super_cell.center()  # centers the atoms in the unit cell, so there is the same amount of vacuum on all sides.
len_cell = super_cell.cell.lengths()    # get cell length
super_cell.pbc = [False, False, False]	# set boundary conditions
print('Supercell:', super_cell)

# add groups for NEMD
split = np.array( [len_cell[0]*2.0/100] 
                + [len_cell[0]*10/100] 
                + [len_cell[0]*70/100]  # 70/100
                + [len_cell[0]*10/100] )   
split = [0] + list(np.cumsum(split)) + [len_cell[0]]
print('Split list:', split)

ncounts = add_group_by_position(split, super_cell, direction='x')  # return a list
print('Atoms per group:', ncounts)
print('Total atoms:', sum(ncounts))


# write model.xyz file
write('model.xyz', super_cell, write_info=False)  # write an initial extxyz file
write('ST4_gra_2_new_box.vasp', super_cell, format='vasp', label='ST2_gra_2_box', direct=True, sort=['C'])


with open('model.xyz', 'r') as fin:  # revise the extxyz file
    lines = fin.readlines()
    fin.close()

lines[1] = lines[1].split("R:3")[0] + "R:3:group:I:1" + lines[1].split("R:3")[1]
for i in range(len(lines) - 2):
    group_id = super_cell.info[i]["groups"][0]
    if group_id  == 4:
        lines[i + 2] = lines[i + 2][:-2] + f"\t0\n"
    else:
        lines[i + 2] = lines[i + 2][:-2] + f"\t{group_id}\n"

with open("model.xyz", "w") as fout:
    fout.writelines(lines)
    fout.close()
