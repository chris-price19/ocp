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
from ase.visualize.plot import plot_atoms
from ase.constraints import FixAtoms


def get_cat_sids(cat_map, sids):

    s = set(cat_map.keys())
    cids_int = [cat_map[key] for key in sids if key in s]
    # cids_int = [cat_map[key] for key in sids if key in set(cat_map.keys())]

    return np.unique(cids_int)


def regex_symbol_filter(map_dict, regexp, nelements = None):
    
    # mapdict_subset = {key: value for (key,value) in map_dict.items() if regexp.search(value['bulk_symbols'])}
    
    if nelements is not None:
        element_regex = '[A-Z][^A-Z]*'
        mapdict_subset = {key: value for (key, value) in map_dict.items() if regexp.search(value['bulk_symbols']) and len(re.findall(element_regex, value['bulk_symbols'])) <= nelements}
    else:
        mapdict_subset = {key: value for (key,value) in map_dict.items() if regexp.search(value['bulk_symbols'])}
    
    return mapdict_subset

def sids2inds(db, sids):
    
    if isinstance(sids[0], str):
        sids = [int(i.split('random')[-1]) for i in sids]
    
    sids = pd.DataFrame(sids, columns=['sid']).reset_index(drop=True)
    
    all_inds = pd.DataFrame([(i, db[i].sid) for i in np.arange(len(db))], columns=['ind', 'sid'])
    inds = all_inds.merge(sids, how='inner', on='sid')['ind'].values
    
    return inds

def init_atoms_from_lmdb(db, ind):

    """
    returns an atoms object with initial, non-relaxed positions from an lmdb file.
    """
    
    atoms = ase.Atoms(cell=db[ind].cell.numpy().squeeze(), 
                      positions=db[ind].pos.numpy(), 
                      numbers=db[ind].atomic_numbers.numpy(), pbc=True, 
                      info={'energy':db[ind].y_relaxed, 'sid':db[ind].sid, 'tags': db[ind].tags.numpy()})
    
    mask = [True if i == 0 else False for i in atoms.info['tags']]
    atoms.constraints += [FixAtoms(mask=mask)]

    return atoms

def relaxed_atoms_from_lmdb(db, ind):

    """
    returns an atoms object with relaxed positions from an lmdb file.
    """   
    
    atoms = ase.Atoms(cell=db[ind].cell.numpy().squeeze(), 
                      positions=db[ind].pos_relaxed.numpy(), 
                      numbers=db[ind].atomic_numbers.numpy(), pbc=True, 
                      info={'energy':db[ind].y_relaxed, 'sid':db[ind].sid, 'tags': db[ind].tags.numpy()})

    mask = [True if i == 0 else False for i in atoms.info['tags']]
    # print(mask)
    atoms.constraints += [FixAtoms(mask=mask)]
    # print(atoms.constraints)
    
    return atoms


def filter_atoms_by_tag(atoms, keep = None):

    """
    returns an atoms object with tags matching iterable keep
    """
    
    atoms = atoms.copy()
    # print(atoms)
    
    if keep is None:
        keep = np.unique(atoms.info['tags'])
    
    icount = 0
    # print(np.unique(atoms.info['tags']))
    # print(np.unique(atoms.info['tags']).shape)
    # print(type(np.unique(keep)))
        
    while not np.array_equal(np.unique(atoms.info['tags']), np.unique(keep)):
        if atoms.info['tags'][icount] in keep:
            icount += 1
        else:
            atoms.pop(icount)
            atoms.info['tags'] = np.delete(atoms.info['tags'], icount)
    
    return atoms


def plot_atom_graph(data_obj):
    
    """
    visualize atomic graph (useful for verifying type / number of elements, size of graph)
    """


    fig, ax = plt.subplots(1,1,figsize=(8,8))
    
    node_labels = [str(int(i)) for i in data_obj.atomic_numbers.numpy()]
    # print(node_labels)
    g = torch_geometric.utils.to_networkx(data_obj, to_undirected=True, node_attrs = ['atomic_numbers'])

    carac = pd.DataFrame({'ID':np.arange(len(node_labels)), 
                          'type': data_obj.atomic_numbers.numpy(),
                          'size': data_obj.atomic_numbers.numpy() / np.amax(data_obj.atomic_numbers.numpy())})

    carac = carac.set_index('ID')
    carac = carac.reindex(g.nodes())
    carac['type'] = pd.Categorical(carac['type'])
    # print(pd.unique(carac['type'].cat.codes))
    # print(carac['type'].cat.codes)
#     print(carac)
    # Specify colors
    cmap = matplotlib.colors.ListedColormap(['red', 'darkorange', 'green', 'blue', 'purple', 'yellow', 'gray'])

    # Draw graph
    nx.draw(g, with_labels=False, node_color=carac['type'].cat.codes, node_size = carac['size'] * 500, cmap=cmap)
    # can use edgecolors attribute to color border of nodes by tag
    
    return ax