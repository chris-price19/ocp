#!/usr/bin/python

import numpy as np
import pandas as pd

import sys
import os
import re

import networkx as nx
import torch_geometric

import ase
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as m2colors

from ase.visualize.plot import plot_atoms
from ase.constraints import FixAtoms

from pymatgen.analysis.adsorption import reorient_z
from pymatgen.io.ase import AseAtomsAdaptor

from ocpmodels.custom.structure_mod import reflect_atoms

mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)


def convert_asedb_to_dataframe(datadir, database, aug=False, kT=0.025):

    ## strained data

    # database = datadir + 'strain-snapshot.db'
    moleculesdb = datadir + 'adsorbates.db'
    with ase.db.connect(moleculesdb) as db:
        molselection = db.select()

    rows = []
    for fi, ff in enumerate(molselection):
    #     print(ff);    print(ff.id-1);    print(ff.toatoms());    print(ff.data);    print(ff.symbols);    print(ff.natoms);    print(ff.data.SMILE); break
    #     print(type(ff.symbols))
    #     sys.exit()
        rows.append([ff.id-1, ff.data.SMILE, ff.natoms, ff.symbols])

    moldf = pd.DataFrame(rows, columns=["mol_sid", "smile", "mol_natoms", "symbols"])

    with ase.db.connect(datadir + database) as db:
        selection = db.select(filter=lambda x: (x.data.status=='full' and x.data.ads_E != 999 and x.data.slab_E != 999 and x.data.mol_E != 999))

    with ase.db.connect(datadir + database) as db:
        ground_states = db.select(filter=lambda x: (x.data.status=='full' and x.data.ads_E != 999 and x.data.strain_id == 0))

    ground_states = list(ground_states)
    rows = []
    strains = []
    full_natoms = []

    for fi, ff in enumerate(selection):
        
    #     rlxatoms = ff.toatoms()
        rlxatoms = [i.toatoms() for i in ground_states if i.data.ads_sid == ff.data.ads_sid]
        if len(rlxatoms) > 0:
            rlxatoms = rlxatoms[0]
        else:
            continue
        rlxatoms.info['tags'] = ff.data.tags
        rlxatoms.info['sid'] = ff.data.ads_sid
        rlxatoms.info['energy'] = ff.data['ads_E'] - ff.data['slab_E'] - ff.data['mol_E']
        rlxatoms.info['strain_id'] = ff.data['strain_id']
        rlxatoms.info['og_strain'] = ff.data['strain']
        rlxatoms.info['strain'] = np.expand_dims((ff.data['strain'] - np.eye(3))[0:2,0:2].flatten(),0)*100  ## xx, xy, yx, yy
        rlxatoms.info['hand'] = 'R'
        strains.append((ff.data['strain'] - np.eye(3))[0:2,0:2]*100)
        full_natoms.append(len(rlxatoms.get_chemical_symbols()))
        
        slab_atomic_numbers = np.array(rlxatoms.get_chemical_symbols())[[True if i in [0,1] else False for i in rlxatoms.info['tags']]]
    #     slab_symbols = [i for i in rlxatoms.get_chemical_symbols() if slab_atomic_numbers]
    #     print(slab_atomic_numbers)
    #     sys.exit()
        cu_concentration = np.sum(slab_atomic_numbers == 'Cu') / len(slab_atomic_numbers)
        non_cu_atoms = np.unique(slab_atomic_numbers[slab_atomic_numbers != 'Cu'])
        if len(non_cu_atoms) == 0:
            non_cu_atoms = 'Cu'
        else:
            non_cu_atoms = non_cu_atoms[0]
            
       # magnitude of off diagaonal elements / magnitude of diagonal elements. < 1 means more uniaxial, > 1 means more shear
    #     shear_ratio = (np.sum(np.abs(ff.data.strain)) - np.trace(np.abs(ff.data.strain)) + 3) / np.trace(np.abs(ff.data.strain))
    #     shear_ratio = np.abs(ff.data.strain[0,1]) / (np.trace(np.abs(ff.data.strain)) - 3)
        shear_norm = np.linalg.norm(ff.data.strain) - np.linalg.norm(np.diag(ff.data.strain))
        uniaxial_norm = np.linalg.norm(np.diag(ff.data.strain)) # - np.linalg.norm(np.eye(3))
        
    #     sv1 = np.trace(ff.data.strain) - 3
    #     sv2 = (np.trace(ff.data.strain)**2 - np.trace(ff.data.strain**2))/2
    #     sv3 = np.linalg.det(ff.data.strain)
        
        strain_norm = np.linalg.norm(ff.data.strain) - np.linalg.norm(np.eye(3))
        area_strain = (ff.toatoms().cell.area(2) - rlxatoms.cell.area(2)) / rlxatoms.cell.area(2)
        strain_anisotropy = np.abs(np.linalg.norm(ff.data.strain[0:]) / np.linalg.norm(ff.data.strain[1:]))

        if aug:
            augatoms = reflect_atoms(rlxatoms)
            strains.append(augatoms.info['strain'].squeeze().reshape((2,2)))
            full_natoms.append(len(augatoms.get_chemical_symbols()))
            # glist_full.append(augatoms)
            # reduced_aug_atoms = filter_atoms_by_tag(augatoms, keep=np.array([1,2]))
            # glist_reduced.append(reduced_aug_atoms)
            # reduced_natoms.append(len(reduced_aug_atoms.get_chemical_symbols()))
            # full_atom_targets = full_atom_targets + (augatoms.info['tags'].tolist())
            # reduced_atom_targets = reduced_atom_targets + (reduced_atoms.info['tags'].tolist()) + (reduced_aug_atoms.info['tags'].tolist())

        rows.append([ff.data.ads_sid, ff.data.slab_sid, ff.data.mol_sid, ff.data.strain_id, ff.natoms, rlxatoms.info['hand'], ff.data.ads_E, ff.data.slab_E, ff.data.mol_E, area_strain, strain_norm, shear_norm, uniaxial_norm, strain_anisotropy, cu_concentration, non_cu_atoms])
        if aug:
            rows.append([ff.data.ads_sid, ff.data.slab_sid, ff.data.mol_sid, ff.data.strain_id, ff.natoms, augatoms.info['hand'], ff.data.ads_E, ff.data.slab_E, ff.data.mol_E, area_strain, strain_norm, shear_norm, uniaxial_norm, strain_anisotropy, cu_concentration, non_cu_atoms])
    #     rows.append([ff.data.ads_sid, ff.data.slab_sid, ff.data.mol_sid, ff.data.strain_id, ff.natoms, ff.data.ads_E, ff.data.slab_E, ff.data.mol_E, area_strain, sv1, sv2, sv3, strain_anisotropy, cu_concentration, non_cu_atoms])

    # sys.exit()
    df = pd.DataFrame(rows, columns=["ads_sid", "slab_sid", "mol_sid", "strain_id", "total_natoms", "hand", "ads_E", "slab_E", "mol_E", "area_strain", "strain_norm", "shear_norm", "uniaxial_norm", "strain_anisotropy", "cu_concentration", "alloy_element"])
    
    df['ads_energy'] = df['ads_E'] - df['slab_E'] - df['mol_E']

    df = pd.merge(df, df.loc[df['strain_id'] == 0, ['ads_sid','ads_energy']], on='ads_sid', how='left')
    df = pd.merge(df, df.loc[df['strain_id'] == 0, ['ads_sid','slab_E']], on='ads_sid', how='left')
    df = pd.merge(df, moldf, on='mol_sid', how='left')
    df['strain_delta'] = df['ads_energy_x'] - df['ads_energy_y']
    df.rename(columns={'ads_energy_x':'ads_energy', 'ads_energy_y':'ground_state_energy', 'slab_E_x':'slab_E', 'slab_E_y':'slab_ground_state_energy'}, inplace=True)

    df['real_ads_atoms'] = df['symbols'].apply(lambda x: ''.join(np.unique(x).tolist()))
    df['majority_Cu'] = np.where(df['cu_concentration'] > 0.5, True, False)

    df['strain_delta_bin'] = np.where(df['strain_delta'].abs() <= kT, 1, np.where(df['strain_delta'] > kT, 2, 0))

    data_norms = pd.DataFrame(np.vstack([np.mean(strains, axis=0).flatten(), np.std(strains, axis=0).flatten()]).T, index=['strain_xx', 'strain_xy', 'strain_yx', 'strain_yy'],columns=['mean', 'std'])

    print(len(df))
    print(df.columns)

    return df