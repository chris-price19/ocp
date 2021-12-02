#!/usr/bin/python

import numpy as np
import pandas as pd
import ase

import lmdb
from tqdm import tqdm
import pickle
import lzma

import os

from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels.custom.plot_and_filter import sids2inds
from sklearn.model_selection import train_test_split

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
        target.append(data[target_col])

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
    
    """
    get energy from molecule database of plain molecules in a box. designed for adsorbates.db

    database: ase db file path (str)
    index: adsorbate index (int)
    """

    with ase.db.connect(database) as db:
        
        row = list(db.select(index+1))[0]
        data = row.data
        
    return data['energy']


def reshuffle_lmdb_splits(base_lmdb, splits, outdir = os.getcwd(), ood=False):

    def build_subset(basedb, ind):

        objs = []
        for ti, tt in enumerate(ind):

            objs.append(basedb[tt])

        return objs

    if np.sum(splits) != 1.:
        raise Exception('splits dont add to 1')

    basedb = SinglePointLmdbDataset({"src": base_lmdb})
    trainpct, valpct, testpct = splits

    if ood:

        domain_cols = ['ads_id', 'bulk_mpid']
        if 'scratch' in os.getcwd():
            base_datadir = '/scratch/vshenoy1/chrispr/catalysis/ocp/data/'
        else:
            base_datadir = '/home/ccprice/catalysis-data/ocp/data/'

        map_frame = pd.DataFrame.from_dict(np.load(base_datadir + 'oc20_data_mapping.pkl', allow_pickle=True), orient='index')
        cat_frame = pd.DataFrame.from_dict(np.load(base_datadir + 'mapping_adslab_slab.pkl', allow_pickle=True), orient='index', columns=['slab'])

        # set up dataframe with sid, ads_id, slab_id, bulk_mpid
        merged =  pd.merge(map_frame, cat_frame, left_index=True, right_index=True, how='inner')
        sids =  ['random'+str(aa.sid) for aa in basedb]
        merged = merged.loc[merged.index.isin(sids)]

        # get unique groups and shuffle them
        unique = merged.groupby(domain_cols).size().reset_index().rename(columns={0:'count'}).sample(frac=1).reset_index(drop=True)

        # perform the split using the unique groups
        traingroups = unique[0:int(np.floor(trainpct*len(unique)))]
        valgroups = unique[int(np.floor(trainpct*len(unique))):int(np.floor((trainpct+valpct)*len(unique)))]
        testgroups = unique[int(np.floor((trainpct+valpct)*len(unique))):]

        # remerge with original DF to get all sids corresponding to subgroups
        trainsids = pd.merge(merged.reset_index(), traingroups, on=domain_cols, how='inner')
        valsids = pd.merge(merged.reset_index(), valgroups, on=domain_cols, how='inner')
        testsids = pd.merge(merged.reset_index(), testgroups, on=domain_cols, how='inner')

        # convert to indices in base database
        ind_train = sids2inds(basedb, trainsids['index'].tolist())
        ind_valid = sids2inds(basedb, valsids['index'].tolist())
        ind_test = sids2inds(basedb, testsids['index'].tolist())

        adder = '_ood'

    else:

        inds = np.arange(len(basedb))

        ind_train, ind_rem = train_test_split(inds, train_size=trainpct)
        ind_valid, ind_test = train_test_split(ind_rem, test_size=valpct / (valpct+testpct))
        adder = ''

    print(len(basedb))
    print(len(ind_train))
    print(len(ind_valid))
    print(len(ind_test))
    # print(ind_train[0:50])

    trainobjs = build_subset(basedb, ind_train)
    m, s = write_lmbd(trainobjs, "y_relaxed", outdir, base_lmdb.split('/')[-1].split('.')[0] + '_train' + adder + '.lmdb')
    print(m, s)

    validobjs = build_subset(basedb, ind_valid)
    m, s = write_lmbd(validobjs, "y_relaxed", outdir, base_lmdb.split('/')[-1].split('.')[0] + '_valid' + adder + '.lmdb')
    print(m, s)

    testobjs = build_subset(basedb, ind_test)
    m, s = write_lmbd(testobjs, "y_relaxed", outdir, base_lmdb.split('/')[-1].split('.')[0] + '_test' + adder + '.lmdb')
    print(m, s)

    return