from concurrent.futures import ThreadPoolExecutor, as_completed
import ase.io 
from ase import Atoms, cell, constraints
import torch
import tqdm
from ocpmodels.datasets import LmdbDataset
import pandas as pd
import os, glob, tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subset_num", type=int, default=None)
parser.add_argument("--lmdb_path", type=str, default="/home/jovyan/shared-scratch/jhoon/ocp2023/lmdb/oc20dense-val/data.lmdb")
parser.add_argument("--save_dir_path", type=str, default="/home/jovyan/shared-scratch/jhoon/latent/cifs/val/dense-raw")
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

subset_num = args.subset_num
lmdb_path = args.lmdb_path
save_dir_path = args.save_dir_path
num_workers = args.num_workers
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)


lmdb_data = LmdbDataset({"src": lmdb_path})
# breakpoint()
print("----------------------------------------------")
print("Total number of traj files: ", len(lmdb_data))
print("Save cif files to: ", save_dir_path)
print("----------------------------------------------")

def pyg2atoms(data):
    '''Convert a pytorch geometric data object to an ASE atoms object.'''
    atoms = Atoms(
        numbers=data.atomic_numbers,
        positions=data.pos,
        cell=cell.Cell(data.cell.squeeze(0).numpy()),
        pbc=True,
        tags=data.tags)
    
    fixed_atom_indices = torch.nonzero(data.fixed == 1).squeeze().tolist()
    fix_atoms = constraints.FixAtoms(indices=fixed_atom_indices)
    atoms.set_constraint(fix_atoms)
    return atoms

for data in tqdm.tqdm(lmdb_data):
    sid = data.sid
    cif_path = os.path.join(save_dir_path, f"{sid}.cif")
    relaxed_adslab = pyg2atoms(data)
    ase.io.write(cif_path, relaxed_adslab, format='cif')
print("Processing complete.")