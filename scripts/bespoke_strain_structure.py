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


adssid = 1436007

base_datadir = '../../dft/'

adsgroundstate = read(base_datadir + str(adssid) + '.0/ads/CONTCAR')
slabgroundstate = read(base_datadir + str(adssid) + '.0/slab/CONTCAR') 

straintensor = np.array([0.99565088, 1.02005077, 0.01211224, ])

# straintensor = np.eye(3) - straintensor

strain_tensor_list = generate_strain_tensors(1, man_override=straintensor)