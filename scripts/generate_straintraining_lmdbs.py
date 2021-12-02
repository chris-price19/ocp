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


cwd = os.getcwd()
# ads_id_keep_to_start = [0,1,2,3,4, 5, 6,7,8,9,10,11,12,13,14,15,62,63,65,69,70,71,72,73,74,75,76,77,78,81]

# a2g_strain_rlx = StrainAtomsToGraphs(
#     max_neigh=60,
#     radius=7,
#     r_energy='pass_through',
#     r_forces=False,
#     r_distances=True,
#     r_fixed=True,
#     r_tags=True,
#     r_strain = True,
#     r_energy_delta = True,
# )

a2g_strain_rlx = StrainAtomsToGraphs(
    max_neigh=60,
    radius=7,
    r_energy='energy_delta',
    r_forces=False,
    r_distances=True,
    r_fixed=True,
    r_tags=True,
    r_strain = True,
    r_energy_delta = True,
)

datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/'
dft_datadir = '/scratch/vshenoy1/chrispr/catalysis/dft/'

database = dft_datadir + 'total-calc-plan.db'
with ase.db.connect(database) as db:
    selection = db.select(filter=lambda x: (x.data.status=='full' and x.data.ads_E != 999 and x.data.slab_E != 999 and x.data.mol_E != 999))

with ase.db.connect(database) as db:
    ground_states = db.select(filter=lambda x: (x.data.status=='full' and x.data.ads_E != 999 and x.data.strain_id == 0))

ground_states = list(ground_states)
    
rows = []
glist_full = []
glist_reduced = []
strains = []
full_natoms = []
reduced_natoms = []
full_atom_targets = []
reduced_atom_targets = []

for fi, ff in enumerate(selection):
    
    rlxatoms = [i.toatoms() for i in ground_states if i.data.ads_sid == ff.data.ads_sid]
    if len(rlxatoms) > 0:
        rlxatoms = rlxatoms[0]
    else:
        continue
    # rlxatoms = ff.toatoms()
    rlxatoms.info['tags'] = ff.data.tags
    rlxatoms.info['sid'] = ff.data.ads_sid
    rlxatoms.info['energy'] = ff.data['ads_E'] - ff.data['slab_E'] - ff.data['mol_E']
    rlxatoms.info['strain_id'] = ff.data['strain_id']
    rlxatoms.info['og_strain'] = ff.data['strain']
    rlxatoms.info['strain'] = np.expand_dims((ff.data['strain'] - np.eye(3))[0:2,0:2].flatten(),0)*100  ## xx, xy, yx, yy
    rlxatoms.info['hand'] = 'R'
    augatoms = reflect_atoms(rlxatoms)

    strains.append((ff.data['strain'] - np.eye(3))[0:2,0:2]*100)
    strains.append(augatoms.info['strain'].squeeze().reshape((2,2)))

#     print(strains)
    full_natoms.append(len(rlxatoms.get_chemical_symbols()))
    full_natoms.append(len(augatoms.get_chemical_symbols()))
    
    glist_full.append(rlxatoms)
    glist_full.append(augatoms)
    
    reduced_atoms = filter_atoms_by_tag(rlxatoms, keep=np.array([1,2]))
    reduced_aug_atoms = filter_atoms_by_tag(augatoms, keep=np.array([1,2]))
    glist_reduced.append(reduced_atoms)
    glist_reduced.append(reduced_aug_atoms)
    
    reduced_natoms.append(len(reduced_atoms.get_chemical_symbols()))
    reduced_natoms.append(len(reduced_aug_atoms.get_chemical_symbols()))
    
#     print(type(rlxatoms.info['tags']))
    full_atom_targets = full_atom_targets + (rlxatoms.info['tags'].tolist()) + (augatoms.info['tags'].tolist())
    reduced_atom_targets = reduced_atom_targets + (reduced_atoms.info['tags'].tolist()) + (reduced_aug_atoms.info['tags'].tolist())
    
    shear_ratio = (np.sum(np.abs(ff.data.strain)) - np.trace(np.abs(ff.data.strain)) + 3) / np.trace(np.abs(ff.data.strain))
    # magnitude of off diagaonal elements / magnitude of diagonal elements. < 1 means more uniaxial, > 1 means more shear
    strain_norm = np.linalg.norm(ff.data.strain[0:2,0:2])
    strain_anisotropy = np.abs(np.diff(np.diag(ff.data.strain[0:2,0:2])))[0]

    rows.append([ff.data.ads_sid, ff.data.slab_sid, ff.data.mol_sid, ff.data.strain_id, ff.natoms, rlxatoms.info['hand'], ff.data.ads_E, ff.data.slab_E, ff.data.mol_E, strain_norm, shear_ratio, strain_anisotropy])
    rows.append([ff.data.ads_sid, ff.data.slab_sid, ff.data.mol_sid, ff.data.strain_id, ff.natoms, augatoms.info['hand'], ff.data.ads_E, ff.data.slab_E, ff.data.mol_E, strain_norm, shear_ratio, strain_anisotropy])

# sys.exit()
df = pd.DataFrame(rows, columns=["ads_sid", "slab_sid", "mol_sid", "strain_id", "total_natoms", "hand", "ads_E", "slab_E", "mol_E", "strain_norm", "shear_ratio", "strain_anisotropy"])
print(len(df))
df['ads_energy'] = df['ads_E'] - df['slab_E'] - df['mol_E']
df = pd.merge(df, df.loc[(df['strain_id'] == 0) & (df['hand'] == 'R'), ['ads_sid','ads_energy']], on='ads_sid', how='left')
df = pd.merge(df, df.loc[(df['strain_id'] == 0) & (df['hand'] == 'R'), ['ads_sid','slab_E']], on='ads_sid', how='left')
# df = pd.merge(df, moldf, on='mol_sid', how='left')
df['strain_delta'] = df['ads_energy_x'] - df['ads_energy_y']
df.rename(columns={'ads_energy_x':'ads_energy', 'ads_energy_y':'ground_state_energy', 'slab_E_x':'slab_E', 'slab_E_y':'slab_ground_state_energy'}, inplace=True)

# sys.exit()

print(len(rows))
print(len(df))
print(len(glist_full))


for di, dd in enumerate(df['strain_delta'].values):
    
#     print(df['ads_sid'][di])
#     print(df['strain_id'][di])
#     print(glist_full[di].info['sid'])
#     print(glist_reduced[di].info['sid'])
#     print(glist_reduced[di].info['strain_id'])
#     break
    glist_full[di].info['energy_delta'] = dd
    glist_reduced[di].info['energy_delta'] = dd

full_list = a2g_strain_rlx.convert_all(glist_full)
reduced_list = a2g_strain_rlx.convert_all(glist_reduced)

data_norms = pd.DataFrame(np.vstack([np.mean(strains, axis=0).flatten(), np.std(strains, axis=0).flatten()]).T, index=['strain_xx', 'strain_xy', 'strain_yx', 'strain_yy'],columns=['mean', 'std'])

full_weights = compute_class_weight('balanced', classes=np.unique(full_atom_targets), y=full_atom_targets)
reduced_weights = compute_class_weight('balanced', classes=np.unique(reduced_atom_targets), y=reduced_atom_targets)

outdir = datadir + 'aug_strained_full_structures'
outfile = 'binaryCu-relax-moleculesubset.lmdb'
target_col = "y_relaxed"
mean, std = write_lmbd(full_list, target_col, outdir, outfile)

print(mean, std)
data_norms.loc['target'] = [mean, std]
data_norms.loc['max_atoms'] = [int(np.amax(full_natoms)), int(np.amax(full_natoms))]
data_norms.loc['global_min_target'] = [0, 0]
data_norms.loc['num_targets'] = 4
for ti in np.arange(data_norms.loc['num_targets'][0]-1):
    data_norms.loc['classweight_'+str(int(ti))] = full_weights[int(ti)]
data_norms.to_csv(outdir + '/data.stats')
fulldb = SinglePointLmdbDataset({"src": outdir + '/' + outfile})
reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=False)
reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=True)

##############
data_norms = pd.DataFrame(np.vstack([np.mean(strains, axis=0).flatten(), np.std(strains, axis=0).flatten()]).T, index=['strain_xx', 'strain_xy', 'strain_yx', 'strain_yy'],columns=['mean', 'std'])
outdir = datadir + 'aug_strained_reduced_structures'
outfile = 'binaryCu-relax-moleculesubset.lmdb'
target_col = "y_relaxed"
mean, std = write_lmbd(reduced_list, target_col, outdir, outfile)

data_norms.loc['target'] = [mean, std]
data_norms.loc['max_atoms'] = [int(np.amax(reduced_natoms)), int(np.amax(reduced_natoms))]
data_norms.loc['global_min_target'] = [1, 1]
data_norms.loc['num_targets'] = 3
for ti in np.arange(data_norms.loc['num_targets'][0]-1):
    data_norms.loc['classweight_'+str(int(ti))] = reduced_weights[int(ti)]
data_norms.to_csv(outdir + '/data.stats')

reduceddb = SinglePointLmdbDataset({"src": outdir + '/' + outfile})
reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=False)
reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=True)