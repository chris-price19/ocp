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


# datadir = '/home/ccprice/catalysis-data/ocp/data/'
datadir = '/backup/chris/catalysis-staging/slab_trajectories/'
cat_map = np.load(datadir + 'mapping_adslab_slab.pkl', allow_pickle=True)
map_dict = np.load(datadir + 'oc20_data_mapping.pkl', allow_pickle=True)

regexp = re.compile(r'Cu')
binary_coppers = regex_symbol_filter(map_dict, regexp, 2)

binary_cids = get_cat_sids(cat_map, list(binary_coppers.keys()))

# binary_cids = binary_cids[0:10]
print(binary_cids)

filename = 'binaryCu-cat-relax.lmdb'

db = lmdb.open(
    datadir + filename,
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

## energy pass_through for now, find out if extrapolated or free energy was used

a2g_rlx = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy='pass_through',
    r_forces=False,
    r_distances=True,
    r_fixed=True,
)

missing = []

for ci, cc in enumerate(binary_cids):

    file = 'slab_trajectories/' + cc + '.extxyz.xz'
    try:
        file = 'slab_trajectories/' + cc + '.extxyz.xz'
        atoms = read_lzma_to_atoms(datadir + file)
    except:
        missing.append(cc)
        continue
    tags = [i.tag for i in atoms]
    atoms.info['tags'] = tags
    atoms.info['sid'] = int(cc.split('random')[-1])

    rcell, Q = atoms.cell.standard_form()
    atoms.cell = atoms.cell @ Q.T
    atoms.cell[np.abs(atoms.cell) < 1e-8] = 0.
    atoms.positions = atoms.positions @ Q.T

    p1 = AseAtomsAdaptor.get_structure(atoms)
    p1 = reorient_z(p1)
    compareatoms = AseAtomsAdaptor.get_atoms(p1)

    if np.amax(np.abs(compareatoms.cell - atoms.cell)) > 1e-4:
        print('atoms 1')
        print(atoms)
        print('atoms 2')
        print(compareatoms)
        break

    data = a2g_rlx.convert(atoms)
       
    data.fid = ci # becomes ind

    # no neighbor edge case check
    if data.edge_index.shape[1] == 0:
        print("no neighbors", traj_path)
        continue

    txn = db.begin(write=True)
    txn.put(f"{ci}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()

    db.sync()

db.close()

print(missing)