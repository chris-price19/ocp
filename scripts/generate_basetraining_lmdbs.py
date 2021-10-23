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

a2g_rlx = AtomsToGraphs(
    max_neigh=60,
    radius=7,
    r_energy='pass_through',
    r_forces=False,
    r_distances=True,
    r_fixed=True,
    r_tags=True
)


full_list = []
reduced_list = []

for di, dd in enumerate(datadirs):

    dbloc = {"src": dd}
    traindb = SinglePointLmdbDataset(dbloc)

    binary_inds = sids2inds(traindb, list(binary_coppers.keys()))
    print(len(binary_inds) * (1+number_of_tensors))

    if 'pos_relaxed' in traindb[0].keys:
        pass
    else:
        break

    glist_full, glist_reduced = filter_lmdbs_and_graphs(traindb, binary_inds, a2g_rlx, filteratoms=False)

    full_list = full_list + glist_full
    reduced_list = reduced_list + glist_reduced

outdir = base_datadir + 'full_structures'
outfile = 'binaryCu-relax-moleculesubset.lmdb'
target_col = "y_relaxed"
mean, std = write_lmbd(full_list, target_col, outdir, outfile)
with open(outdir + '/target.stats', 'w') as file:
    file.write('mean    std\n %f %f' % (mean, std))
print(mean, std)
fulldb = SinglePointLmdbDataset({"src": outdir + '/' + outfile})

outdir = base_datadir + 'reduced_structures'
outfile = 'binaryCu-relax-moleculesubset.lmdb'
target_col = "y_relaxed"
mean, std = write_lmbd(reduced_list, target_col, outdir, outfile)
with open(outdir + '/target.stats', 'w') as file:
    file.write('mean    std\n %.8f \t %.8f' % (mean, std))
print(mean, std)
reduceddb = SinglePointLmdbDataset({"src": outdir + '/' + outfile})
