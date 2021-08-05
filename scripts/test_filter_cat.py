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

from pymatgen.analysis.adsorption import reorient_z
from pymatgen.io.ase import AseAtomsAdaptor

import lmdb
from tqdm import tqdm
import pickle

def get_cat_sids(cat_map, sids):

    print('local') 
    s = set(cat_map.keys())
    cids_int = [cat_map[key] for key in sids if key in s]
    # cids_int = [cat_map[key] for key in sids if key in set(cat_map.keys())]

    return np.unique(cids_int)

# datadir = '/home/ccprice/catalysis-data/ocp/data/'
datadir = '/backup/chris/catalysis-staging/slab_trajectories/'
cat_map = np.load(datadir + 'mapping_adslab_slab.pkl', allow_pickle=True)
map_dict = np.load(datadir + 'oc20_data_mapping.pkl', allow_pickle=True)
rewrite=True

if rewrite or not os.path.isfile('intermediate_cids.txt'):
    regexp = re.compile(r'Cu')
    binary_coppers = regex_symbol_filter(map_dict, regexp, 2)
    binary_cids = get_cat_sids(cat_map, list(binary_coppers.keys()))
    with open('intermediate_cids.txt', 'w') as f:
        for item in binary_cids:
            f.write("%s\n" % item)
else:
    with open('intermediate_cids.txt', 'r') as f:
        binary_cids = f.readlines()

print(binary_cids)
print(len(binary_cids))

missing = []
for ci, cc in enumerate(binary_cids):

    try:
        file = 'slab_trajectories/' + cc + '.extxyz.xz'
        atoms = read_lzma_to_atoms(datadir + file)
    except:
        missing.append(cc)
        continue

print(missing)
print(len(missing))