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
import ase.db

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import reorient_z
from ase.io import read, write
from ase.calculators.vasp import Vasp


cwd = os.getcwd()
ads_id_keep_to_start = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,62,63,65,69,70,71,72,73,74,75,76,77,78,81]

VASP_FLAGS = {'ibrion': 2,
          'nsw': 300,
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

if 'scratch' in cwd:
    base_datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/'
    # datadirs = [
    #         base_datadir + 'is2re/all/train/data.lmdb',
    #         base_datadir + 'is2re/all/val_id/data.lmdb',
    #         base_datadir + 'is2re/all/val_ood_ads/data.lmdb',
    #         base_datadir + 'is2re/all/val_ood_both/data.lmdb',
    #         base_datadir + 'is2re/all/val_ood_cat/data.lmdb',
    #         base_datadir + 'is2re/all/test_id/data.lmdb',
    #         base_datadir + 'is2re/all/test_ood_ads/data.lmdb',
    #         base_datadir + 'is2re/all/test_ood_cat/data.lmdb',
    #         base_datadir + 'is2re/all/test_ood_both/data.lmdb',
    #         ]
else:
    base_datadir = '/home/ccprice/catalysis-data/ocp/data/'
    # datadirs = [base_datadir + 'is2re/10k/train/data.lmdb',]


# map_dict = np.load(base_datadir + 'oc20_data_mapping.pkl', allow_pickle=True)
# cat_map = np.load(base_datadir + 'mapping_adslab_slab.pkl', allow_pickle=True)
# regexp = re.compile(r'Cu')
# binary_coppers = regex_symbol_filter(map_dict, regexp, nelements=2, molecules=ads_id_keep_to_start)

database = base_datadir + 'adsorbates.db'
box_size = 12

for mi, mm in enumerate(ads_id_keep_to_start):
    
    with ase.db.connect(database) as db:
        
        atoms = db.get_atoms(mm+1)
        row = list(db.select(mm+1))[0]
        data = row.data
        print(atoms)
    
    atoms.cell = np.array([[box_size,0,0],[0,box_size,0],[0,0,box_size]])
    atoms.positions += box_size/2

    os.mkdir(str(mm), exist_ok=True)
    os.chdir(str(mm))

    vasp_flags = VASP_FLAGS.copy()
    vasp_flags['kpts'] = (1,1,1)
    calc = Vasp(setups='recommended', **vasp_flags)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    data['energy'] = energy

    with ase.db.connect(database) as db:            
        db.update(row.id, atoms=atoms, data=data)

    del atoms, row, data

    os.chdir(cwd)