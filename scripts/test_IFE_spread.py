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

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import reorient_z
from ase.io import read, write
from ase.calculators.vasp import Vasp

import matplotlib.pyplot as plt


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
    else:
        return np.NaN

    return energy_free

strains = []
energies = []

for ai, aa in enumerate(np.arange(6)):

    if ai == 0:
        gstatemolatoms = read(str(ai)+'/CONTCAR')
        gstateslabatoms = read(str(ai)+'s/CONTCAR')
        e_mol = outcarparse(str(ai)+'/OUTCAR')
        e_slab = outcarparse(str(ai)+'s/OUTCAR')

        gstatearea = gstatemolatoms.get_volume() / np.linalg.norm(gstatemolatoms.cell[-1,:])

        gstateife = e_slab - e_mol
        continue

    molatoms = read(str(ai)+'/CONTCAR')
    slabatoms = read(str(ai)+'s/CONTCAR')

    e_mol = outcarparse(str(ai)+'/OUTCAR')
    e_slab = outcarparse(str(ai)+'s/OUTCAR')

    ife = e_slab - e_mol

    print(np.abs((molatoms.cell - gstatemolatoms.cell) / gstatemolatoms.cell))
    strainmag = np.nansum(np.ma.masked_invalid(np.abs((molatoms.cell - gstatemolatoms.cell) / gstatemolatoms.cell)))
    print(strainmag)
    # sys.exit()

    strains.append(strainmag)
    energies.append(ife - gstateife)

print(strains)
print(energies)
print(gstatearea)
plt.scatter(strains, energies / gstatearea)
plt.show()