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

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import reorient_z
from ase.io import read, write
from ase.calculators.vasp import Vasp


cwd = os.getcwd()
# ads_id_keep_to_start = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,62,63,65,69,70,71,72,73,74,75,76,77,78,81]

a2g_strain_rlx = StrainAtomsToGraphs(
    max_neigh=60,
    radius=7,
    r_energy='pass_through',
    r_forces=False,
    r_distances=True,
    r_fixed=True,
    r_tags=True,
    r_strain = True,
    r_energy_delta = True,
)

base_datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/'
dft_datadir = '/scratch/vshenoy1/chrispr/catalysis/dft/'
database = dft_datadir + 'total-calc-plan.db'
with ase.db.connect(database) as db:
    selection = db.select(filter=lambda x: (x.data.status=='full' and x.data.ads_E != 999 and x.data.slab_E != 999 and x.data.mol_E != 999))
    
rows = []
glist_full = []
glist_reduced = []

for fi, ff in enumerate(selection):
    
    rlxatoms = ff.toatoms()
    rlxatoms.info['tags'] = ff.data.tags
    rlxatoms.info['sid'] = ff.data.ads_sid
    rlxatoms.info['energy'] = ff.data['ads_E'] - ff.data['slab_E'] - ff.data['mol_E']
    rlxatoms.info['strain_id'] = ff.data['strain_id']
    rlxatoms.info['strain'] = ff.data['strain']

    glist_full.append(rlxatoms)
    glist_reduced.append(filter_atoms_by_tag(rlxatoms, keep=np.array([1,2])))
    
    shear_ratio = (np.sum(np.abs(ff.data.strain)) - np.trace(np.abs(ff.data.strain)) + 3) / np.trace(np.abs(ff.data.strain))
    # magnitude of off diagaonal elements / magnitude of diagonal elements. < 1 means more uniaxial, > 1 means more shear
    strain_norm = np.linalg.norm(ff.data.strain[0:2,0:2])
    strain_anisotropy = np.abs(np.diff(np.diag(ff.data.strain[0:2,0:2])))[0]

    rows.append([ff.data.ads_sid, ff.data.slab_sid, ff.data.mol_sid, ff.data.strain_id, ff.natoms, ff.data.ads_E, ff.data.slab_E, ff.data.mol_E, strain_norm, shear_ratio, strain_anisotropy])
    
df = pd.DataFrame(rows, columns=["ads_sid", "slab_sid", "mol_sid", "strain_id", "total_natoms", "ads_E", "slab_E", "mol_E", "strain_norm", "shear_ratio", "strain_anisotropy"])
df['ads_energy'] = df['ads_E'] - df['slab_E'] - df['mol_E']
df = pd.merge(df, df.loc[df['strain_id'] == 0, ['ads_sid','ads_energy']], on='ads_sid')
df = pd.merge(df, df.loc[df['strain_id'] == 0, ['ads_sid','slab_E']], on='ads_sid')
df = pd.merge(df, moldf, on='mol_sid', how='left')
df['strain_delta'] = df['ads_energy_x'] - df['ads_energy_y']
df.rename(columns={'ads_energy_x':'ads_energy', 'ads_energy_y':'ground_state_energy', 'slab_E_x':'slab_E', 'slab_E_y':'slab_ground_state_energy'}, inplace=True)

for di, dd in enumerate(df['strain_delta'].values):

    glist_full[di].info['energy_delta'] = dd
    glist_reduced[di].info['energy_delta'] = dd

full_list = a2g_strain_rlx.convert_all(glist_full)
reduced_list = a2g_strain_rlx.convert_all(glist_reduced)

# os.makedirs()
outdir = base_datadir + 'strained_full_structures'
outfile = 'binaryCu-relax-moleculesubset.lmdb'
target_col = "y_relaxed"
mean, std = write_lmbd(full_list, target_col, outdir, outfile)
with open(outdir + '/target.stats', 'w') as file:
    file.write('mean\tstd\n%.8f\t%.8f' % (mean, std))
print(mean, std)
fulldb = SinglePointLmdbDataset({"src": outdir + '/' + outfile})

outdir = base_datadir + 'strained_reduced_structures'
outfile = 'binaryCu-relax-moleculesubset.lmdb'
target_col = "y_relaxed"
mean, std = write_lmbd(reduced_list, target_col, outdir, outfile)
with open(outdir + '/target.stats', 'w') as file:
    file.write('mean\tstd\n%.8f\t%.8f' % (mean, std))
print(mean, std)
reduceddb = SinglePointLmdbDataset({"src": outdir + '/' + outfile})
