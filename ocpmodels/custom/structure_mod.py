#!/usr/bin/python

import numpy as np
import pandas as pd

import sys
import os
import re

import ase
import pymatgen.analysis.elasticity.strain as pmgst
import pymatgen.transformations.standard_transformations as ptrans
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from copy import deepcopy

from numpy.random import default_rng

class StrainTensor:

    def __init__(self, eid, eps = np.eye(3)):

        self.eid = eid
        self.eps = eps


def generate_strain_tensors(number_of_tensors, max_mag = 0.01, man_override = None):

    strain_tensor_list = []
    rng = default_rng()

    strain_tensor_list.append(StrainTensor(0))
    # first is always the ground state

    if man_override is None:

        for ii in np.arange(number_of_tensors):

            ## xx, yy, xy
            ecoords = np.array([rng.uniform(-max_mag, max_mag), rng.uniform(-max_mag, max_mag), rng.uniform(-max_mag, max_mag)])

            eps = np.eye(3)
            eps[0,0] += ecoords[0]
            eps[1,1] += ecoords[1]
            eps[0,1] += ecoords[2]/2.
            eps[1,0] += ecoords[2]/2.

            # eps[1,0] += ecoords[2]
            # eps[1,0] += ecoords[2]
            # eps[1,0] += ecoords[2]
            # eps[1,0] += ecoords[2]

            e = StrainTensor(int(ii+1), eps)

            strain_tensor_list.append(e)

    else:
        for ii in np.arange(len(man_override)):

            ## xx, yy, xy
            ecoords = man_override[ii,:]

            eps = np.eye(3)
            eps[0,0] += ecoords[0]
            eps[1,1] += ecoords[1]
            eps[0,1] += ecoords[2]/2.
            eps[1,0] += ecoords[2]/2.

            e = StrainTensor(ii+1, eps)

            strain_tensor_list.append(e)

    return strain_tensor_list

# def bespoke_strain(strain_list)

def strain_atoms(atoms, eps):

    ## eps is a 3x3 deformation gradient matrix (symmetric)
    ## default = np.eye(3)

    epsilon = eps.eps
    # epsilon = eps

    # d = pmgst.convert_strain_to_deformation(epsilon, shape='symmetric')

    strainatoms = atoms.copy()
    save_constraints = atoms.constraints.copy()
    save_info = atoms.info.copy()
    save_info['y'] = np.NaN
    
    p1 = AseAtomsAdaptor.get_structure(strainatoms)

    d = pmgst.Deformation(epsilon)
    # print('***')
    # print(d.green_lagrange_strain)
    p2 = d.apply_to_structure(p1)

    strainatoms = AseAtomsAdaptor.get_atoms(p2)
    
    strainatoms.constraints = save_constraints
    strainatoms.info = save_info
    strainatoms.info['eid'] = int(eps.eid)

    return strainatoms


def reflect_atoms(atoms1):
    
    flippedatoms = atoms1.copy()
#     print(flippedatoms.info)
    
    savex = deepcopy(flippedatoms.positions[:,0])
    savey = deepcopy(flippedatoms.positions[:,1])
    savecell = deepcopy(flippedatoms.cell)

    flippedatoms.positions[:,0] = savey
    flippedatoms.positions[:,1] = savex
    flippedatoms.cell[0,0] = savecell[1,1]
    flippedatoms.cell[0,1] = savecell[1,0]
    flippedatoms.cell[1,0] = savecell[0,1]
    flippedatoms.cell[1,1] = savecell[0,0]

    savestrain = deepcopy(flippedatoms.info['og_strain'])
    newstrain = np.eye(3)
    
    newstrain[0,0] = savestrain[1,1]
    newstrain[1,0] = savestrain[0,1]
    newstrain[0,1] = savestrain[1,0]
    newstrain[1,1] = savestrain[0,0]
    
    flippedatoms.info['og_strain'] = newstrain
    flippedatoms.info['strain'] = np.expand_dims((flippedatoms.info['og_strain'] - np.eye(3))[0:2,0:2].flatten(),0)*100.
    flippedatoms.info['hand'] = 'L'
#     print(flippedatoms.info)
    
    return flippedatoms