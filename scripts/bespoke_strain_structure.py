#!/usr/bin/python

from ocpmodels.preprocessing import AtomsToGraphs, StrainAtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels.custom import *
# import ocpmodels.custom.plot_and_filter import 

import numpy as np
import pandas as pd

import sys
import os
import re

import ase
import ase.db

from sklearn.utils.class_weight import compute_class_weight

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import reorient_z
from ase.io import read, write
from ase.calculators.vasp import Vasp

VASP_FLAGS = {'ibrion': 2,
          'nsw': 600,
          # 'nelm': 5,  # turn this off!
          'isif': 0,
          'isym': 0,
          'lreal': 'Auto',
          'ediffg': -0.03,
          'symprec': 1e-10,
          'encut': 400,
          # 'laechg': False,
          'lwave': False,
          'lcharg': False,
          'ncore': 8,
          'kpar': 2,
          'gga': 'RP',
          'pp': 'PBE',
          'prec': 'Normal',
          'ediff': 1e-4,
          'ivdw': 12,
          'amin': 0.01,
          'lcorr': True,
          'nsim': 16,
          'ismear': 1,
          # 'nblock': 1,
          # 'gga_compat': False
          # 'xc': 'PBE'
          }

# class DeltaTemplate(Template):
#     delimiter = "%"

# def strfdelta(tdelta, fmt):
#     d = {"D": tdelta.days}
#     hours, rem = divmod(tdelta.seconds, 3600)
#     hours += tdelta.days*24
#     minutes, seconds = divmod(rem, 60)
#     d["H"] = '{:02d}'.format(hours)
#     d["M"] = '{:02d}'.format(minutes)
#     d["S"] = '{:02d}'.format(seconds)
#     t = DeltaTemplate(fmt)
#     return t.substitute(**d)

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
            return False, data

    os.system('echo $SLURM_JOBID >> jobfile.out')
    ## delete adsorbate atoms
    atoms = atoms.copy()    
    keep = [0,1]  # 0 = subsurface atoms, 1 = surface atoms, 2 = adsorbate atoms
    icount = 0
    tags = data['tags'].copy()
    while not np.array_equal(np.unique(tags), np.unique(keep)):
        if tags[icount] in keep:
            icount += 1
        else:
            atoms.pop(icount)
            tags = np.delete(tags, icount)
    
    ## check for CONTCAR / continuation
    if os.path.exists('CONTCAR'):
        if os.path.getsize('CONTCAR') > 0:
            atoms = ase.io.read('CONTCAR', format='vasp')

    vasp_flags = VASP_FLAGS.copy()
    vasp_flags['kpts'] = calculate_surface_k_points(atoms)
    calc = Vasp(setups='recommended', **vasp_flags)
    atoms.calc = calc

    ## output from vasp goes to vasp.out
    try:
        energy = atoms.get_potential_energy()
        data['slab_E'] = energy
    except Exception as err:
        data['error'] = str(err) + '_slab'
        print(str(err))

    # calc.write_input(atoms)
    
    os.chdir(basedir)

    return True, data

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
            return False, atoms, data

    os.system('echo $SLURM_JOBID >> jobfile.out')
    ## check for CONTCAR / continuation
    if os.path.exists('CONTCAR'):
        if os.path.getsize('CONTCAR') > 0:
            atoms = ase.io.read('CONTCAR', format='vasp')

    vasp_flags = VASP_FLAGS.copy()
    vasp_flags['kpts'] = calculate_surface_k_points(atoms)
    calc = Vasp(setups='recommended', **vasp_flags)
    atoms.calc = calc

    try:
        energy = atoms.get_potential_energy()
        data['ads_E'] = energy
    except Exception as err:
        data['error'] = str(err)
        print(str(err))

    # calc.write_input(atoms)
    
    os.chdir(basedir)

    return True, atoms, data




adssid = 1436007

base_datadir = '/mnt/io2/scratch_vshenoy1/chrispr/catalysis/dft/'

existing_strain_ids = os.listdir(base_datadir)
existing_strain_ids = [i.split('.')[0] for i in existing_strain_ids if i.split('.')[0] == str(adssid)]
max_strain_ids = len(existing_strain_ids)

adsgroundstate = read(base_datadir + str(adssid) + '.0/ads/CONTCAR')
slabgroundstate = read(base_datadir + str(adssid) + '.0/slab/CONTCAR')


straintensor = np.eye(3) - np.array([[0.99565088, 1.02005077, 0.01211224, ]])

# straintensor = np.eye(3) - straintensor

strain_tensor_list = generate_strain_tensors(1, man_override=straintensor)

adsgroundstate = strain_atoms(adsgroundstate, strain_tensor_list[-1])
slabgroundstate = strain_atoms(slabgroundstate, strain_tensor_list[-1])

print(adsgroundstate.cell)
print(slabgroundstate.cell)

# os.makedirs(base_datadir + str(adssid) + '.' + str(max_strain_ids))
# os.chdir(base_datadir + str(adssid) + '.' + str(max_strain_ids))

# os.makedirs('ads')
# os.chdir('ads')


# write('CONTCAR', adsgroundstate, format='vasp')
# # vasp_flags = VASP_FLAGS.copy()
# # vasp_flags['kpts'] = calculate_surface_k_points(adsgroundstate)
# # calc = Vasp(setups='recommended', **vasp_flags)
# # adsgroundstate.calc = calc
# # energy = adsgroundstate.get_potential_energy()

# print(os.getcwd())

# os.chdir('..')
# os.makedirs('slab')
# os.chdir('slab')

# write('CONTCAR', slabgroundstate, format='vasp')


# print(os.getcwd())
# os.chdir('..')

# # vasp_flags = VASP_FLAGS.copy()
# # vasp_flags['kpts'] = calculate_surface_k_points(slabgroundstate)
# # calc = Vasp(setups='recommended', **vasp_flags)
# # slabgroundstate.calc = calc
# # energy = slabgroundstate.get_potential_energy()

# print(os.getcwd())

# os.chdir(base_datadir)