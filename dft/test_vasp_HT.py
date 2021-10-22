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
import time
from contextlib import redirect_stdout

import ase
import ase.db
import ase.io

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import reorient_z
from ase.calculators.vasp import Vasp


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
          'gga': 'RP',
          'pp': 'PBE',
          'prec': 'Normal',
          'ediff': 1e-4,
          'ivdw': 12,
          'amin': 0.01,
          'lcorr': True,
          'nsim': 8,
          'ismear': 1,
          # 'gga_compat': False
          # 'xc': 'PBE'
          }

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

def run_slab_calc(basedir, atoms, data):

    os.chdir(basedir)
    cwd = os.getcwd()

    newdir = str(data['ads_sid']) + '.' + str(data['strain_id']) + '/slab'
    os.makedirs(newdir, exist_ok=True)
    os.chdir(newdir)

    ## check for running status.
    mod_time = sorted([os.path.getmtime(f) for f in os.listdir()])
    if len(mod_time) > 0:
        last_time = mod_time[-1]
        if round(time.time()) - last_time < 60*5: # if the most recent file has been updated within the last 5 minutes.
            print('too new')
            os.chdir(basedir)
            return False, data

    ## check for CONTCAR / continuation
    if os.path.exists('CONTCAR'):
        if os.path.getsize('CONTCAR') > 0:
            atoms = ase.io.read('CONTCAR', format='vasp')

    vasp_flags = VASP_FLAGS.copy()
    vasp_flags['kpts'] = calculate_surface_k_points(atoms)
    calc = Vasp(setups='recommended', **vasp_flags)
    atoms.calc = calc

    # with open('log.out', 'w') as f:
    #     with redirect_stdout(f):
    try:
        energy = atoms.get_potential_energy()
        data['ads_E'] = energy
    except Exception as err:
        data['error'] = str(err)
        print(str(err))

    # calc.write_input(atoms)
    
    os.chdir(basedir)

    return True, atoms, data


def run_cat_calc(basedir, atoms, data):

    os.chdir(basedir)
    cwd = os.getcwd()

    newdir = str(data['ads_sid']) + '.' + str(data['strain_id']) + '/ads'
    os.makedirs(newdir, exist_ok=True)
    os.chdir(newdir)

    ## check for running status.
    mod_time = sorted([os.path.getmtime(f) for f in os.listdir()])
    if len(mod_time) > 0:
        last_time = mod_time[-1]
        if round(time.time()) - last_time < 60*5: # if the most recent file has been updated within the last 5 minutes.
            print('too new')
            os.chdir(basedir)
            return False, data

    ## check for CONTCAR / continuation
    if os.path.exists('CONTCAR'):
        if os.path.getsize('CONTCAR') > 0:
            atoms = ase.io.read('CONTCAR', format='vasp')

    vasp_flags = VASP_FLAGS.copy()
    vasp_flags['kpts'] = calculate_surface_k_points(atoms)
    calc = Vasp(setups='recommended', **vasp_flags)
    atoms.calc = calc

    # with open('log.out', 'w') as f:
    #     with redirect_stdout(f):
    try:
        energy = atoms.get_potential_energy()
        data['ads_E'] = energy
    except Exception as err:
        data['error'] = str(err)
        print(str(err))

    # calc.write_input(atoms)
    
    os.chdir(basedir)

    return True, atoms, data



basedir = '/scratch/vshenoy1/chrispr/catalysis/test-dft/'
data = {
        'status': '',
        'occupied': False,
        'error' : '',
        'start_time': '',
        'calc_time': 999,
        'ads_sid': 0,
        'slab_sid': 1,
        'mol_sid': 2,
        'tags': np.arange(20),
        'strain': np.eye(3),
        'strain_id': 3,                    
        'ads_E': 999,
        'slab_E': 999,
        'mol_E': 999
        }

for dd in np.arange(5):
    atoms = ase.io.read(basedir + '/' + str(dd) + '/CONTCAR')
    run_cat_calc(basedir, atoms, data)