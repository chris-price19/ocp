#!/usr/bin/python

from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels.custom import *
# import ocpmodels.custom.plot_and_filter import 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



import sys
import os
import re

import networkx as nx
import torch_geometric

import ase
import matplotlib
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

from pymatgen.analysis.adsorption import reorient_z
from pymatgen.io.ase import AseAtomsAdaptor

import random

import lmdb
from tqdm import tqdm
import pickle


def filter_lmdbs_and_graphs(datadir, map_dict, sids, graph_builder, outdir, outfile):

    ## train lmdb
    dbloc = {"src": datadir }

    traindb = SinglePointLmdbDataset(dbloc)

    binary_inds = sids2inds(traindb, list(sids.keys()))

    print(len(binary_inds))

    # sys.exit()
    glist = []
    for bi, bb in enumerate(binary_inds):
        
        rlxatoms = relaxed_atoms_from_lmdb(traindb, bb)

        rcell, Q = rlxatoms.cell.standard_form()
        rlxatoms.cell = rlxatoms.cell @ Q.T
        rlxatoms.positions = rlxatoms.positions @ Q.T

        p1 = AseAtomsAdaptor.get_structure(rlxatoms)
        p1 = reorient_z(p1)
        compareatoms = AseAtomsAdaptor.get_atoms(p1)
l
        if np.amax(np.abs(compareatoms.cell - rlxatoms.cell)) > 1e-4:
            print('atoms 1')
            print(rlxatoms)
            print('atoms 2')
            print(compareatoms)
            return

        surfatoms = filter_atoms_by_tag(rlxatoms, keep=np.array([1,2]))

        glist.append(surfatoms)

    glist = graph_builder.convert_all(glist)    

    target_col = "y_relaxed"
    mean, std = write_lmbd(glist, target_col, outdir, outfile)
    print(mean)
    print(std)

    return mean, std

# params
print(os.getcwd())
datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/' # 'data/is2re/10k/'
map_dict = np.load(datadir + 'oc20_data_mapping.pkl', allow_pickle=True)

regexp = re.compile(r'Cu')
binary_coppers = regex_symbol_filter(map_dict, regexp, 2)

a2g_rlx = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy='pass_through',
    r_forces=False,
    r_distances=True,
    r_fixed=True,
)


## train test split, optional
train_frac = 0.9
test_frac = 0.1

s = pd.Series(binary_coppers)
train_sids, test_sids  = [i.to_dict() for i in train_test_split(s, train_size=train_frac)]
# cutoff = np.ceil(len(binary_coppers) * test_frac)

# np.random.shuffle(binary_coppers)

# test_sids = binary_coppers[0:cutoff]
# train_sids = binary_coppers[cutoff:]

## all train
# filter_lmdbs_and_graphs(datadir + 'is2re/all/train/data.lmdb', map_dict, binary_coppers, a2g_rlx, datadir + 'is2re/all/train/', binaryCu-relax.lmdb')

## train/test split in-domain
filter_lmdbs_and_graphs(datadir + 'is2re/all/train/data.lmdb', map_dict, train_sids, a2g_rlx, datadir + 'is2re/all/train/', 'binaryCu-relax-split.lmdb')
filter_lmdbs_and_graphs(datadir + 'is2re/all/train/data.lmdb', map_dict, test_sids, a2g_rlx, datadir + 'is2re/all/test_id/', 'binaryCu-relax-split.lmdb')


filter_lmdbs_and_graphs(datadir + 'is2re/all/val_id/data.lmdb', map_dict, binary_coppers, a2g_rlx, datadir + 'is2re/all/val_id/', 'binaryCu-relax.lmdb')


## validation lmdb
# dbloc = {"src": datadir + 'is2re/all/val_id/data.lmdb'}

# valdb = SinglePointLmdbDataset(dbloc)

# binary_inds = sids2inds(valdb, list(binary_coppers.keys()))

# print(valdb[binary_inds[0]])
# print(len(binary_inds))

# # sys.exit()
# glist = []
# for bi, bb in enumerate(binary_inds):
    
#     rlxatoms = relaxed_atoms_from_lmdb(valdb, bb)
#     surfatoms = filter_atoms_by_tag(rlxatoms, keep=np.array([1,2]))
#     surfgraph = a2g_rlx.convert(surfatoms)
    
# #     fig, ax = plt.subplots(1, 2, figsize=(16,8))
# #     plot_atoms(rlxatoms, ax[0], radii=0.8, rotation=("0x, 90y, 0z"))
# #     plot_atoms(surfatoms, ax[1], radii=0.8, rotation=("0x, 90y, 0z"))
# #     break

#     glist.append(surfgraph)
    
# print(len(glist))
# print(glist[0])
# # sys.exit()
# target_col = "y_relaxed"
# mean, std = write_lmbd(glist, target_col, datadir + 'is2re/all/val_id', 'binaryCu-relax.lmdb')
# print(mean)
# print(std)