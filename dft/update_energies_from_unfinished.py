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


logfile = 'slurm-1040138.out'
total_database = '../../dft/total-calc-plan.db'


with open(logfile, 'r') as f:

    lines = f.readlines()[1:]

for li, ll in enumerate(lines):

    sid = ll.rstrip().split('/')[-2]
    strain_id = sid.split('.')[-1]
    ads_sid = sid.split('.')[0]
    calcstring = ll.rstrip().split('/')[-1]

    os.chdir(ads_sid + '.' + strain_id + '/' + calcstring)
    

    with ase.db.connect(total_database) as db:

        selection = db.select(filter=lambda x: (x.data.ads_sid == int(ads_sid) and x.data.strain_id == int(strain_id)))
        entry = next(selection)
        data = entry.data
        
        data[calcstring+'_E'] = 
        
        db.update(entry.id, data=data)
                