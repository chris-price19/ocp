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

number_of_tensors = 6
max_magnitude = 0.03
ads_id_keep_to_start = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,62,63,65,69,70,71,72,73,74,75,76,77,78,81]

db = ase.db.connect('total-calc-plan.db')

cwd = os.getcwd()

if 'scratch' in cwd:
    base_datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/'
    datadirs = [
            base_datadir + 'is2re/all/train/data.lmdb',
            base_datadir + 'is2re/all/val_id/data.lmdb',
            base_datadir + 'is2re/all/val_ood_ads/data.lmdb',
            base_datadir + 'is2re/all/val_ood_both/data.lmdb',
            base_datadir + 'is2re/all/val_ood_cat/data.lmdb',
            base_datadir + 'is2re/all/test_id/data.lmdb',
            base_datadir + 'is2re/all/test_ood_ads/data.lmdb',
            base_datadir + 'is2re/all/test_ood_cat/data.lmdb',
            base_datadir + 'is2re/all/test_ood_both/data.lmdb',
            ]
else:
    base_datadir = '/home/ccprice/catalysis-data/ocp/data/'
    datadirs = [base_datadir + 'is2re/10k/train/data.lmdb',]

map_dict = np.load(base_datadir + 'oc20_data_mapping.pkl', allow_pickle=True)
cat_map = np.load(base_datadir + 'mapping_adslab_slab.pkl', allow_pickle=True)
regexp = re.compile(r'Cu')
binary_coppers = regex_symbol_filter(map_dict, regexp, nelements=2, molecules=ads_id_keep_to_start)
### write sids, strains to an ase db

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
                    'slab_sid': int(cat_map['random'+str(rlxatoms.info['sid'])].split('random')[-1]),
                    'mol_sid': map_dict['random'+str(rlxatoms.info['sid'])]['ads_id'],
                    'original_split': split,
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

