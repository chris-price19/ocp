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


VASP_FLAGS = {'ibrion': 2,
          'nsw': 200,
          'isif': 0,
          'isym': 0,
          'lreal': 'Auto',
          'ediffg': -0.03,
          'symprec': 1e-10,
          'encut': 400,
          # 'laechg': False,
          'lwave': False,
          'lcharg': False,
          'ncore': 4,
          'kpar': 4,
          'gga': 'RP',
          'pp': 'PBE',
          'prec': 'Normal',
          'ediff': 1e-4,
          'ivdw': 12,
          # 'amin': 0.01,
          # 'lcorr': True,
          'nsim': 8,
          'ismear': 1,
          # 'nblock': 1,
          # 'gga_compat': False
          # 'xc': 'PBE'
          }

def calculate_surface_k_points(atoms):
    '''
    For surface calculations, it's a good practice to calculate the k-point
    mesh given the unit cell size. We do that on-the-spot here.
    Arg:
        atoms   `ase.Atoms` object of the structure we want to relax
    Returns:
        k_pts   A 3-tuple of integers indicating the k-point mesh to use
    '''
    cell = atoms.get_cell()
    order = np.inf
    a0 = np.linalg.norm(cell[0], ord=order)
    b0 = np.linalg.norm(cell[1], ord=order)
    multiplier = 40
    k_pts = (max(1, int(round(multiplier/a0))),
             max(1, int(round(multiplier/b0))),
             1)
    return k_pts


def run_slab_calc(basedir, atoms, data):

    os.chdir(basedir)
    cwd = os.getcwd()

    newdir = str(data['ads_sid']) + '.' + str(data['strain_id']) + '/slab'
    os.makedirs(newdir, exist_ok=True)
    os.chdir(newdir)

    ## check for running status.
    mod_time = sorted([os.path.getmtime(f) for f in os.listdir()])
    if len(mod_time) > 0:
        last_time = mod_time[-1]
        if round(time.time()) - last_time < 60*5: # if the most recent file has been updated within the last 5 minutes.
            return False, data

    ## delete adsorbate atoms
    atoms = atoms.copy()    
    keep = [0,1]  # 0 = subsurface atoms, 1 = surface atoms, 2 = adsorbate atoms
    icount = 0
    tags = data.tags.copy()
    while not np.array_equal(np.unique(tags), np.unique(keep)):
        if tags[icount] in keep:
            icount += 1
        else:
            atoms.pop(icount)
            tags = np.delete(tags, icount)
    
    ## check for CONTCAR / continuation
    if os.path.exists('CONTCAR'):
        if os.path.getsize('CONTCAR') > 0:
            atoms = ase.io.read('CONTCAR', format='vasp')

    vasp_flags = VASP_FLAGS.copy()
    vasp_flags['kpts'] = calculate_surface_k_points(atoms)
    calc = Vasp(setups='recommended', **vasp_flags)
    atoms.calc = calc

    ## output from vasp goes to vasp.out
    try:
        energy = atoms.get_potential_energy()
        data['slab_E'] = energy
    except Exception as err:
        data['error'] = str(err) + '_slab'
        print(str(err))

    # calc.write_input(atoms)
    
    os.chdir(basedir)

    return True, data

def run_cat_calc(basedir, atoms, data):

    os.chdir(basedir)
    cwd = os.getcwd()

    newdir = str(data['ads_sid']) + '.' + str(data['strain_id']) + '/ads'
    os.makedirs(newdir, exist_ok=True)
    os.chdir(newdir)

    ## check for running status.
    mod_time = sorted([os.path.getmtime(f) for f in os.listdir()])
    if len(mod_time) > 0:
        last_time = mod_time[-1]
        if round(time.time()) - last_time < 60*5: # if the most recent file has been updated within the last 5 minutes.
            return False, atoms, data

    ## check for CONTCAR / continuation
    if os.path.exists('CONTCAR'):
        if os.path.getsize('CONTCAR') > 0:
            atoms = ase.io.read('CONTCAR', format='vasp')

    vasp_flags = VASP_FLAGS.copy()
    vasp_flags['kpts'] = calculate_surface_k_points(atoms)
    calc = Vasp(setups='recommended', **vasp_flags)
    atoms.calc = calc

    try:
        energy = atoms.get_potential_energy()
        data['ads_E'] = energy
    except Exception as err:
        data['error'] = str(err)
        print(str(err))

    # calc.write_input(atoms)
    
    os.chdir(basedir)

    return True, atoms, data


def main():

    limit = 20
    basedir = '/scratch/vshenoy1/chrispr/catalysis/dft/'
    database = basedir + 'total-calc-plan.db'
    global_completion = False
    skip_ids = []

    while not global_completion:

        with ase.db.connect(database) as db:
            # this has been tested on command line, should work
            # use only numbers and strings, skip the NaNs... too much fuckery
            selection = list(db.select(filter=lambda x: (x.data.status=='' or x.data.status=='half') and x.id not in skip_ids, limit=limit))
        
        if not len(selection):
            global_completion = True
            break
        selection = selection[np.random.randint(limit)]
        data = selection.data.copy()

        atoms = selection.get_atoms()

        if data['status'] == '':
            
            begin = datetime.now()
            data['start_time'] = begin.strftime("%d/%m/%Y %H:%M:%S")

            success, rlxatoms, data = run_cat_calc(basedir, atoms, data)

            if not success:
                skip_ids.append(selection.id)
                continue

            halfend = datetime.now()
            data['calc_time'] = str((halfend - begin) // 1000000 * 1000000)

            with ase.db.connect(database) as db:
                data['status'] = 'half'
                db.update(selection.id, atoms=rlxatoms, data=data)
                skip_ids = []            

            success, data = run_slab_calc(basedir, atoms, data)

            if not success:
                skip_ids.append(selection.id)
                continue

            end = datetime.now()
            data['calc_time'] = str((end - begin) // 1000000 * 1000000)

            with ase.db.connect(database) as db:
                data['status'] = 'full'
                db.update(selection.id, data=data)
                skip_ids = []

        elif data['status'] == 'half':

            begin = datetime.now()

            success, data = run_slab_calc(basedir, atoms, data)

            if not success:
                skip_ids.append(selection.id)
                continue

            end = datetime.now()
            data['calc_time'] = str(datetime.strptime(data['calc_time'], "%H:%M:%S") + (end - begin) // 1000000 * 1000000)

            with ase.db.connect(database) as db:
                data['status'] = 'full'
                db.update(selection.id, data=data)
                skip_ids = []

    return

main()