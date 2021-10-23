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

import networkx as nx
import torch_geometric

import ase
import matplotlib
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

os.getcwd()
ads_id_keep_to_start = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,62,63,65,69,70,71,72,73,74,75,76,77,78,81]


ads_database = '../data/adsorbates.db'
total_database = '../../dft/total-calc-plan.db'

for mi, mm in enumerate(ads_id_keep_to_start):  

    with ase.db.connect(ads_database) as adb:
        row = list(adb.select(mm+1))[0]
        energy = row.data['energy']

    with ase.db.connect(total_database) as db:

        selection = db.select(filter=lambda x: (x.data.mol_sid == mm and x.data.status == 'full'))

        for fi, ff in enumerate(list(selection)):

            data = ff.data
            # print(data)
            data['mol_E'] = energy
            # print(data)
            db.update(ff.id, data=data)
            
            # sys.exit()
                