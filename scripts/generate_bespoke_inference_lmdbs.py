#!/usr/bin/python

from ocpmodels.preprocessing import AtomsToGraphs, StrainAtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels.custom import *
from ocpmodels.custom.analyze import load_map_frame, generate_tags
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

def outcarparse(filepath):
    
    if os.path.isfile(filepath):
        with open(filepath,'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                # Free energy
                if line.lower().startswith('  free  energy   toten'):
                    energy_free = float(line.split()[-2])
                # Extrapolated zero point energy
                if line.startswith('  energy  without entropy'):
                    energy_zero = float(line.split()[-1])
                    
    return energy_zero


cwd = os.getcwd()
# ads_id_keep_to_start = [0,1,2,3,4, 5, 6,7,8,9,10,11,12,13,14,15,62,63,65,69,70,71,72,73,74,75,76,77,78,81]

a2g_strain_rlx = StrainAtomsToGraphs(
    max_neigh=60,
    radius=7,
    # r_energy='energy_delta_thresh',
    r_forces=False,
    r_distances=True,
    r_fixed=True,
    r_tags=True,
    r_strain = True,
    # r_energy_delta = True,
    r_hand = True,
)

datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/'
dft_datadir = '/scratch/vshenoy1/chrispr/catalysis/dft/'

map_frame = load_map_frame(datadir, filtering = False)

database = dft_datadir + 'temp.db'

with ase.db.connect(datadir + 'adsorbates.db') as db:
    ads_rows = list(db.select())
    
# print(ads_rows[69].data)

rows = []
glist_full = []
# strains = []
# full_natoms = []
# full_atom_targets = []

number_of_tensors = 500
max_magnitude = 0.03

base_ads_sids = [2238472, 2345755]

for fi, ff in enumerate(base_ads_sids):

    man_strains = generate_strain_tensors(number_of_tensors, max_mag = max_magnitude)

    submapframe = map_frame.loc[map_frame['ads_sid'] == ff].copy()
    submapframe['slab_sid'] = submapframe['slab'].apply(lambda x: int(x.split('random')[-1]))

    for ai, aa in enumerate(man_strains):

        if aa.eid == 0:
            # skip the ground state for inference since we already know the answer
            continue
        
        # rlxatoms = ff.toatoms()
        rlxatoms = read(dft_datadir + str(ff) + '.0/ads/CONTCAR')
        ads_eng = outcarparse(dft_datadir + str(ff) + '.0/ads/OUTCAR')
        slab_eng = outcarparse(dft_datadir + str(ff) + '.0/slab/OUTCAR')
        tags = generate_tags(rlxatoms)
        natoms = len(rlxatoms)
        
        strainatoms = strain_atoms(rlxatoms, aa)
        # rlxatoms.info['tags'] = ff.data.tags
        rlxatoms.info['tags'] = tags

        # rlxatoms.info['sid'] = ff.data.ads_sid
        rlxatoms.info['sid'] = ff

        ### 7/27 only thing left is to get energy I think for ads_E and slab_E. get from directory directly.
        # rlxatoms.info['gs_energy'] = ff.data['ads_E'] - ff.data['slab_E'] - ff.data['mol_E']
        rlxatoms.info['gs_energy'] = ads_eng - slab_eng - ads_rows[submapframe['ads_id'].values[0]].data['energy']
        #####

        rlxatoms.info['strain_id'] = aa.eid
        rlxatoms.info['og_strain'] = aa.eps
        rlxatoms.info['strain'] = np.expand_dims((aa.eps - np.eye(3))[0:2,0:2].flatten(),0)*100  ## xx, xy, yx, yy
    
        rlxatoms.info['hand'] = 'R'
        augatoms = reflect_atoms(rlxatoms)

        # strains.append((ff.data['strain'] - np.eye(3))[0:2,0:2]*100)
        # strains.append(augatoms.info['strain'].squeeze().reshape((2,2)))

        # full_natoms.append(len(rlxatoms.get_chemical_symbols()))
        # full_natoms.append(len(augatoms.get_chemical_symbols()))
        
        glist_full.append(rlxatoms)
        glist_full.append(augatoms)

        # full_atom_targets = full_atom_targets + (rlxatoms.info['tags'].tolist()) + (augatoms.info['tags'].tolist())
            
        # shear_ratio = (np.sum(np.abs(aa.eps)) - np.trace(np.abs(aa.eps)) + 3) / np.trace(np.abs(aa.eps))
        # # magnitude of off diagaonal elements / magnitude of diagonal elements. < 1 means more uniaxial, > 1 means more shear
        # strain_norm = np.linalg.norm(aa.eps[0:2,0:2])
        # strain_anisotropy = np.abs(np.diff(np.diag(aa.eps[0:2,0:2])))[0]

        shear_norm = np.linalg.norm(rlxatoms.info['og_strain']) - np.linalg.norm(np.diag(rlxatoms.info['og_strain']))
        uniaxial_norm = np.linalg.norm(np.diag(rlxatoms.info['og_strain'])) # - np.linalg.norm(np.eye(3))
        
    #     sv1 = np.trace(ff.data.strain) - 3
    #     sv2 = (np.trace(ff.data.strain)**2 - np.trace(ff.data.strain**2))/2
    #     sv3 = np.linalg.det(ff.data.strain)
        
        strain_norm = np.linalg.norm(rlxatoms.info['og_strain']) - np.linalg.norm(np.eye(3))
        area_strain = (strainatoms.cell.area(2) - rlxatoms.cell.area(2)) / rlxatoms.cell.area(2)
        strain_anisotropy = np.abs(np.linalg.norm(rlxatoms.info['og_strain'][0:]) / np.linalg.norm(rlxatoms.info['og_strain'][1:]))

        rows.append([submapframe['ads_sid'].values[0], submapframe['slab_sid'].values[0], submapframe['ads_id'].values[0], aa.eid, natoms, rlxatoms.info['hand'], rlxatoms.info['gs_energy'], area_strain, strain_norm, shear_norm, uniaxial_norm, strain_anisotropy, rlxatoms.info['strain'].squeeze()[0], rlxatoms.info['strain'].squeeze()[1], rlxatoms.info['strain'].squeeze()[-1]])
        rows.append([submapframe['ads_sid'].values[0], submapframe['slab_sid'].values[0], submapframe['ads_id'].values[0], aa.eid, natoms, augatoms.info['hand'], rlxatoms.info['gs_energy'], area_strain, strain_norm, shear_norm, uniaxial_norm, strain_anisotropy, augatoms.info['strain'].squeeze()[0], augatoms.info['strain'].squeeze()[1], augatoms.info['strain'].squeeze()[-1]])

df = pd.DataFrame(rows, columns=["ads_sid", "slab_sid", "mol_sid", "strain_id", "total_natoms", "hand", "gs_energy", "area_strain", "strain_norm", "shear_norm", "uniaxial_norm", "strain_anisotropy", "strain_xx", "strain_xy", "strain_yy"])
# df['ads_energy'] = df['ads_E'] - df['slab_E'] - df['mol_E']
# df = pd.merge(df, df.loc[(df['strain_id'] == 0) & (df['hand'] == 'R'), ['ads_sid','ads_energy']], on='ads_sid', how='left')
# df = pd.merge(df, df.loc[(df['strain_id'] == 0) & (df['hand'] == 'R'), ['ads_sid','slab_E']], on='ads_sid', how='left')
# df['strain_delta'] = df['ads_energy_x'] - df['ads_energy_y']
# df.rename(columns={'ads_energy_x':'ads_energy', 'ads_energy_y':'ground_state_energy', 'slab_E_x':'slab_E', 'slab_E_y':'slab_ground_state_energy'}, inplace=True)

# sys.exit()

print(len(rows))
print(len(df))
print(len(glist_full))

full_list = a2g_strain_rlx.convert_all(glist_full)

# data_norms = pd.DataFrame(np.vstack([np.mean(strains, axis=0).flatten(), np.std(strains, axis=0).flatten()]).T, index=['strain_xx', 'strain_xy', 'strain_yx', 'strain_yy'],columns=['mean', 'std'])

# full_weights = compute_class_weight('balanced', classes=np.unique(full_atom_targets), y=full_atom_targets)
# energy_class_weights = compute_class_weight('balanced', classes=np.unique(energy_targets), y=energy_targets)

outdir = datadir + 'inference_aug_strained_full_structures'
outfile = 'n2_traindomain_inference.lmdb'
target_col = None # "y_relaxed"
mean, std = write_lmbd(full_list, target_col, outdir, outfile, append=False)

df.to_csv(outdir + '/' + outfile.split('.')[0] + '.csv')


## data_norms.loc['target'] = [mean, std]
## data_norms.loc['target'] = [mean, std]
################
# data_norms.loc['max_atoms'] = [int(np.amax(full_natoms)), int(np.amax(full_natoms))]
# data_norms.loc['global_min_node_target'] = [0, 0]
# data_norms.loc['num_graph_targets'] = 3
# data_norms.loc['num_node_targets'] = 3

# for ti in np.arange(data_norms.loc['num_node_targets'][0]):
#     data_norms.loc['node_classweight_'+str(int(ti))] = full_weights[int(ti)]
    
# for ti in np.arange(data_norms.loc['num_graph_targets'][0]):
#     data_norms.loc['graph_classweight_'+str(int(ti))] = energy_class_weights[int(ti)]

# data_norms.to_csv(outdir + '/data.stats')
##################

# fulldb = SinglePointLmdbDataset({"src": outdir + '/' + outfile})
# reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=False)
# reshuffle_lmdb_splits(outdir + '/' + outfile, [0.8, 0.1, 0.1], outdir = outdir, ood=True)