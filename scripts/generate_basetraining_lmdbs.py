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

from sklearn.utils.class_weight import compute_class_weight

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import reorient_z
from ase.io import read, write
from ase.calculators.vasp import Vasp


ads_id_keep_to_start = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,62,63,65,69,70,71,72,73,74,75,76,77,78,81]

# db = ase.db.connect('total-calc-plan.db')

cwd = os.getcwd()

if 'scratch' in cwd:
    base_datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/'
    datadirs = [
            base_datadir + 'is2re/all/train/data.lmdb',
            base_datadir + 'is2re/all/val_id/data.lmdb',
            base_datadir + 'is2re/all/val_ood_ads/data.lmdb',
            base_datadir + 'is2re/all/val_ood_both/data.lmdb',
            base_datadir + 'is2re/all/val_ood_cat/data.lmdb',
            # base_datadir + 'is2re/all/test_id/data.lmdb',
            # base_datadir + 'is2re/all/test_ood_ads/data.lmdb',
            # base_datadir + 'is2re/all/test_ood_cat/data.lmdb',
            # base_datadir + 'is2re/all/test_ood_both/data.lmdb',
            ]
else:
    base_datadir = '/home/ccprice/catalysis-data/ocp/data/'
    datadirs = [base_datadir + 'is2re/10k/train/data.lmdb',]

map_dict = np.load(base_datadir + 'oc20_data_mapping.pkl', allow_pickle=True)
cat_map = np.load(base_datadir + 'mapping_adslab_slab.pkl', allow_pickle=True)
# corrections = pd.read_csv('../dft/baseline_y_relaxed_corrections_by_molsid.csv')
regexp = re.compile(r'Cu')
# binary_coppers = regex_symbol_filter(map_dict, regexp, nelements=3, molecules=ads_id_keep_to_start)
binary_coppers = regex_symbol_filter(map_dict, regexp, nelements=3) #, molecules=ads_id_keep_to_start)

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
full_targets = []
reduced_targets = []

for di, dd in enumerate(datadirs):

    print(dd)

    dbloc = {"src": dd}
    traindb = SinglePointLmdbDataset(dbloc)

    binary_inds = sids2inds(traindb, list(binary_coppers.keys()))

    if 'pos_relaxed' in traindb[0].keys:
        pass
    else:
        continue

    glist_full, glist_reduced, full_atom_targets, reduced_atom_targets = filter_lmdbs_and_graphs(traindb, binary_inds, a2g_rlx, filteratoms=False, corrections=False, mapping=map_dict) # pd.DataFrame.from_dict(map_dict, orient='index')

    full_list = full_list + glist_full
    reduced_list = reduced_list + glist_reduced

    full_targets = full_targets + full_atom_targets
    reduced_targets = reduced_targets + reduced_atom_targets

full_weights = compute_class_weight('balanced', classes=np.unique(full_targets), y=full_targets)
reduced_weights = compute_class_weight('balanced', classes=np.unique(reduced_targets), y=reduced_targets)

outdir = base_datadir + 'hybrid_full_structures'
outfile = 'binaryCu-relax-moleculesubset.lmdb'
target_col = "y_relaxed"
mean, std = write_lmbd(full_list, target_col, outdir, outfile)
data_norms = pd.DataFrame([[mean, std]], index=['target'], columns=['mean', 'std'])
data_norms.loc['global_min_target'] = [0, 0]
data_norms.loc['num_targets'] = 4
for ti in np.arange(data_norms.loc['num_targets'][0]-1):
    data_norms.loc['classweight_'+str(int(ti))] = full_weights[int(ti)]
data_norms.to_csv(outdir + '/data.stats')

reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=False)
reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=True)

fulldb = SinglePointLmdbDataset({"src": outdir + '/' + outfile})


############

outdir = base_datadir + 'hybrid_reduced_structures'
outfile = 'binaryCu-relax-moleculesubset.lmdb'
target_col = "y_relaxed"
mean, std = write_lmbd(reduced_list, target_col, outdir, outfile)

data_norms = pd.DataFrame([[mean, std]], index=['target'], columns=['mean', 'std'])
data_norms.loc['target'] = [mean, std]
data_norms.loc['global_min_target'] = [1, 1]
data_norms.loc['num_targets'] = 3
for ti in np.arange(data_norms.loc['num_targets'][0]-1):
    data_norms.loc['classweight_'+str(int(ti))] = full_weights[int(ti)]
data_norms.to_csv(outdir + '/data.stats')

reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=False)
reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=True)

reduceddb = SinglePointLmdbDataset({"src": outdir + '/' + outfile})
