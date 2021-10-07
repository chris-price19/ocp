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
import time
from contextlib import redirect_stdout
from datetime import datetime

import ase
import ase.db
import ase.io

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import reorient_z
from ase.calculators.vasp import Vasp

database = 'total-calc-plan.db'
cwd = os.getcwd()

with ase.db.connect(database) as db:
    # this has been tested on command line, should work
    # use only numbers and strings, skip the NaNs... too much fuckery
    selection = db.select(filter=lambda x: (x.data.error!='')) #  or x.data.status=='half'

results = list(selection)
print(results)
print(len(results))


