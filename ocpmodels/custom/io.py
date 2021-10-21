#!/usr/bin/python

import numpy as np
import ase

import lmdb
from tqdm import tqdm
import pickle
import lzma

import os

def read_lzma(inpfile, outfile):
    with open(inpfile, "rb") as f:
        contents = lzma.decompress(f.read())
        with open(outfile, "wb") as op:
            op.write(contents)

def read_lzma_to_atoms(inpfile, ofile='temp.extxyz'):
    ipdir = '/'.join(inpfile.split('/')[:-1])
    ofile = ipdir + '/' + ofile
    with open(inpfile, "rb") as f:
        contents = lzma.decompress(f.read())
        with open(ofile, "wb") as op:
            op.write(contents)           
    atoms = ase.io.read(ofile, "-1")
    return atoms

def write_lmbd(data_objects, target_col, location, filename):
    
    ## data_objects must be iterable of Data()
    
    os.makedirs(location, exist_ok=True)
    db = lmdb.open(
        location + "/" + filename,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    target = []
    for fid, data in enumerate(data_objects):

        # Filter data if necessary
        # OCP filters adsorption energies > |10| eV and forces > |50| eV/A
        
        data.fid = fid # becomes ind

        # compute mean and std.
        target.append(data.y_relaxed)

        # no neighbor edge case check
        if data.edge_index.shape[1] == 0:
            print("no neighbors", traj_path)
            continue

        txn = db.begin(write=True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

        # txn = db.begin(write=True)
        # txn.put(f"length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
        # txn.commit()
        db.sync()

    db.close()
    
    mean = np.mean(target)
    std = np.std(target)

    return mean, std

def get_adsorbate_energy(database, index):
    
    with ase.db.connect(database) as db:
        
        row = list(db.select(index+1))[0]
        data = row.data
        
    return data['energy']