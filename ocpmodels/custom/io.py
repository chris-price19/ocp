#!/usr/bin/python

import lmdb
from tqdm import tqdm
import pickle

import os

def write_lmbd(data_objects, location, filename):
    
    ## data_objects must be iterable of Data()
    
    os.makedirs(location, exist_ok=True)
    db = lmdb.open(
        location + "/" + filename,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for fid, data in enumerate(data_objects):

        # Filter data if necessary
        # OCP filters adsorption energies > |10| eV and forces > |50| eV/A

        # no neighbor edge case check
        if data.edge_index.shape[1] == 0:
            print("no neighbors", traj_path)
            continue

        txn = db.begin(write=True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    txn = db.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
    txn.commit()


    db.sync()
    db.close()
    
    return