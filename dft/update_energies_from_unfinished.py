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


#use symbolic link to run this from the catalysis/dft directory
cwd = os.getcwd()
ads_id_keep_to_start = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,62,63,65,69,70,71,72,73,74,75,76,77,78,81]

# logfile = 'slurm-1040138.out'
logfile = 'snapshot-dirlist.out'
total_database = 'total-calc-plan.db'

with open(logfile, 'r') as f:

    lines = f.readlines()[1:]

for li, ll in enumerate(lines):

    print(li)
    sid = ll.rstrip().split('/')[-2]
    strain_id = sid.split('.')[-1]
    ads_sid = sid.split('.')[0]
    calcstring = ll.rstrip().split('/')[-1]

    os.chdir(ads_sid + '.' + strain_id + '/' + calcstring)
    
    print(ads_sid)
    print(strain_id)
    print(calcstring)
    ## pausing here - need to get the energy from outcar here and then change dirs back at the end
    updated_energy = outcarparse('OUTCAR')

    with ase.db.connect(cwd + '/' + total_database) as db:

        selection = db.select(filter=lambda x: (x.data.ads_sid == int(ads_sid) and x.data.strain_id == int(strain_id)))
        entry = next(selection)
        print(entry)
        data = entry.data
        
        data[calcstring+'_E'] = updated_energy
        
        db.update(entry.id, data=data)
    
    os.chdir(cwd)