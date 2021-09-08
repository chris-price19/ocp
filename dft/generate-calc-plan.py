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


# number of strains per structure
db = ase.db.connect('total-calc-plan.db')

number_of_tensors = 5
max_magnitude = 0.03

cwd = os.getcwd()

if 'scratch' in cwd:
    base_datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/'
else:
    base_datadir = '/home/ccprice/catalysis-data/ocp/data/'
map_dict = np.load(base_datadir + 'oc20_data_mapping.pkl', allow_pickle=True)
cat_map = np.load(base_datadir + 'mapping_adslab_slab.pkl', allow_pickle=True)
regexp = re.compile(r'Cu')
binary_coppers = regex_symbol_filter(map_dict, regexp, 2)
### write sids, strains to an ase db

datadirs = [base_datadir + 'is2re/10k/train/data.lmdb',]
            # base_datadir + 'is2re/all/val_id/data.lmdb']

for di, dd in enumerate(datadirs):

    dbloc = {"src": dd}
    traindb = SinglePointLmdbDataset(dbloc)
    split = dd.split('/')[-2]
    print(split)

    binary_inds = sids2inds(traindb, list(binary_coppers.keys()))
    print(len(binary_inds) * (1+number_of_tensors))

    for fi, ff in enumerate(binary_inds):

        # print('fi = %d' % fi)

        if 'pos_relaxed' in traindb[ff].keys:
            rlxatoms = relaxed_atoms_from_lmdb(traindb, ff) # these are relaxed structures
        else:
            rlxatoms = init_atoms_from_lmdb(traindb, ff)

        saveconstraints = rlxatoms.constraints.copy()
        p1 = AseAtomsAdaptor.get_structure(rlxatoms)
        p1 = reorient_z(p1)
        rotatoms = AseAtomsAdaptor.get_atoms(p1)
        rotatoms.constraints = saveconstraints

        # man_strains = generate_strain_tensors(number_of_tensors, man_override = np.array([[0.0, 0.1, 0.]]))
        man_strains = generate_strain_tensors(number_of_tensors, max_mag = max_magnitude)

        for ai, aa in enumerate(man_strains):

            # print('ai = %d' % ai)
            # if ai % 5 == 0:
            #     stat = 1
            # else:
            #     stat = np.NaN

            data = {
                    'status': '',
                    'occupied': False,
                    'error' : '',
                    'start_time': '',
                    'calc_time': 999,
                    'ads_sid': rlxatoms.info['sid'],
                    'slab_sid': cat_map['random'+str(rlxatoms.info['sid'])],
                    'mol_sid': map_dict['random'+str(rlxatoms.info['sid'])]['ads_id'],
                    'tags': rlxatoms.info['tags'],
                    'strain': aa.eps,
                    'strain_id': aa.eid,                    
                    'ads_E': 999,
                    'slab_E': 999,
                    'mol_E': 999
                    }

            # print([type(value) for (key, value) in data.items()])
            satoms = rotatoms.copy()
            satoms = strain_atoms(rotatoms, aa)
            
            db.write(satoms, data=data)

            data = {}
            del satoms

            # sys.exit()


    ### 

def check_for_sid(cat_map):

    pass

    return slabsid

