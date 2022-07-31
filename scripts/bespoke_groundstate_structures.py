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
from ase import Atoms

from sklearn.utils.class_weight import compute_class_weight

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import reorient_z
from ase.io import read, write
from ase.calculators.vasp import Vasp

from ocpmodels.custom.analyze import generate_new_sid, load_map_frame
from ase.io import read, write
from ase.build import add_adsorbate

import pickle

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


datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/'
dft_datadir = '/scratch/vshenoy1/chrispr/catalysis/dft/'

# new_ads = ['H', 'N', 'H2', 'NH2', 'NH3']
new_ads = [1, 77, 82, 71, 72]
new_sids = [1266822, 216803, 1908636, 671160, 514753]

base_slab = dft_datadir + "2207859.0/slab/CONTCAR"
adsdb = datadir+"adsorbates.db"
slabatoms = read(base_slab)

cwd = os.getcwd()

with ase.db.connect(datadir + 'adsorbates.db') as db:
    rows = list(db.select())
    
for ni, nn in enumerate(new_ads):

    print(nn)
    adsorb = Atoms(rows[nn].toatoms())
    new_atoms = slabatoms.copy()

    px = np.random.random()*np.linalg.norm(slabatoms.cell[0,:])
    py = np.random.random()*np.linalg.norm(slabatoms.cell[1,:])
    h = new_atoms.positions[:,2].max() + 0.85
    add_adsorbate(new_atoms, adsorb, .85, (px, py))

    vasp_flags = VASP_FLAGS.copy()
    vasp_flags['kpts'] = calculate_surface_k_points(new_atoms)
    calc = Vasp(setups='recommended', **vasp_flags)
    new_atoms.calc = calc

    new_dir = str(new_sids[ni])+'.0/ads'
    # new_slab_dir = str(new_sids[ni]+'.0/slab')

    os.makedirs(dft_datadir + new_dir, exist_ok=True)
    os.chdir(dft_datadir + new_dir)

    new_atoms.get_calculator().initialize(new_atoms)
    new_atoms.get_calculator().write_input(new_atoms)
    write('POSCAR', new_atoms, format='vasp', vasp5=True)
    
    os.chdir(cwd)

    mapdict = np.load(datadir + 'oc20_data_mapping.pkl', allow_pickle=True)
    d1 = {'bulk_id': 3204, 'ads_id': nn, 'bulk_mpid': 'mp-1225835', 'bulk_symbols': 'Cu4S2', 'ads_symbols': rows[nn].data['SMILE'], 'miller_index': (1, 1, 0), 'shift': 0.217, 'top': False, 'adsorption_site': ((px, py, h),)}
    mapdict['random' + str(new_sids[ni])] = d1
    with open(datadir + 'oc20_data_mapping.pkl', 'wb') as handle:
        pickle.dump(mapdict, handle, protocol=4)


    slabdict = np.load(datadir + 'mapping_adslab_slab.pkl', allow_pickle=True)
    slabdict['random' + str(new_sids[ni])] = 'random397927'
    with open(datadir + 'mapping_adslab_slab.pkl', 'wb') as handle:
        pickle.dump(slabdict, handle, protocol=4)

    print('done')