#!/usr/bin/python

from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels.custom import *
# import ocpmodels.custom.plot_and_filter import 

import numpy as np
import pandas as pd

import sys
import os
import re

import ase

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import reorient_z
from ase.io import read, write
from ase.calculators.vasp import Vasp


def calculate_surface_k_points(atoms):
    '''
    For surface calculations, it's a good practice to calculate the k-point
    mesh given the unit cell size. We do that on-the-spot here.
    Arg:
        atoms   `ase.Atoms` object of the structure we want to relax
    Returns:
        k_pts   A 3-tuple of integers indicating the k-point mesh to use
    '''
    cell = atoms.get_cell()
    order = np.inf
    a0 = np.linalg.norm(cell[0], ord=order)
    b0 = np.linalg.norm(cell[1], ord=order)
    multiplier = 40
    k_pts = (max(1, int(round(multiplier/a0))),
             max(1, int(round(multiplier/b0))),
             1)
    return k_pts

VASP_FLAGS = {'ibrion': 2,
              'nsw': 200,
              'isif': 0,
              'isym': 0,
              'lreal': 'Auto',
              'ediffg': -0.03,
              'symprec': 1e-10,
              'encut': 450.,
              'laechg': False,
              'lwave': False,
              'ncore': 4,
              'kpar': 4,
              'gga': 'PE',
              'pp': 'PBE',
              'prec': 'Normal',
              'ediff': 1e-5,
              'ivdw': 12,
              'amin': 0.01,
              'lcorr': True,
              'nsim': 8,
              'ismear': 1,
              'gga_compat': False
              # 'xc': 'PBE'
              }

datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/is2re/10k/train/full-binaryCu-relax-split.lmdb'
# datadir = '/home/ccprice/catalysis-data/ocp/data/is2re/100k/train/binaryCu-relax.lmdb'
dbloc = {"src": datadir }
traindb = SinglePointLmdbDataset(dbloc)

rlxatoms = init_atoms_from_lmdb(traindb, np.random.randint(len(traindb)))

saveconstraints = rlxatoms.constraints.copy()

p1 = AseAtomsAdaptor.get_structure(rlxatoms)

p1 = reorient_z(p1)

rotatoms = AseAtomsAdaptor.get_atoms(p1)
rotatoms.constraints = saveconstraints
print(rotatoms.constraints)
# sys.exit()

number_of_tensors = 5

# man_strains = generate_strain_tensors(number_of_tensors, man_override = np.array([[0.0, 0.1, 0.]]))
man_strains = generate_strain_tensors(number_of_tensors, max_mag = 0.03)

# sys.exit()

for ai, aa in enumerate(man_strains):

    vasp_flags = VASP_FLAGS.copy()
    
    os.makedirs(str(ai), exist_ok=True)
    os.chdir(str(ai))

    t1 = rotatoms.copy()
    satoms = strain_atoms(t1, aa)    
    print(satoms.cell)

    vasp_flags['kpts'] = calculate_surface_k_points(satoms)
    calc = Vasp(**vasp_flags)
    calc.write_input(satoms)

    os.system('ln -s ~/script-repo/v6vasp_submit.sh')
    os.system('sub')
    os.chdir('..')