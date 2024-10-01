import pandas as pd
import lmdb
from ocpmodels.datasets import LmdbDataset
from ocpmodels.preprocessing import AtomsToGraphs
import torch
import ase.io
import os
import pickle
import tqdm



def invert_mapping(mapping_file_path):
    mapping = pd.read_pickle(mapping_file_path)
    inv_mapping = {}
    for key, value in mapping.items():
        system_id = value['system_id']
        config_id = value['config_id']
        file_name = f'{system_id}_{config_id}'
        inv_mapping[file_name] = key
    return inv_mapping


def collect_traj_paths(path):
    system_paths = []

    for dir in os.listdir(path):
        if os.path.isdir(path+dir):
            for file in os.listdir(path+dir):
                if file.endswith(".traj") and not file.endswith("surface.traj"):
                    system_paths.append(path+dir+"/"+file)
    return system_paths


def read_trajectory_extract_features(a2g, traj_path):
    traj = ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    #data_objects[1].tags = torch.LongTensor(tags)
    return data_objects


def write_lmdb(save_path, traj_paths, mapping_file_path, refE_path, tag_path):

    inv_mapping = invert_mapping(mapping_file_path)
    system_paths = collect_traj_paths(traj_paths)
    refE = pd.read_pickle(refE_path)
    tags = pd.read_pickle(tag_path)
    
    a2g = AtomsToGraphs(
                        max_neigh=50,
                        radius=6,
                        r_energy=True,    # False for test data
                        r_forces=True,    # False for test data
                        r_distances=False,
                        r_fixed=True,
                    )
    db = lmdb.open(
        save_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    idx = 0
    for system in tqdm.tqdm(system_paths):
        file_name = system.split("/")[-1].split(".")[0]
        if "_rand" in file_name:
            adslab_id = file_name.split("_rand")[0]
        elif "_heur" in file_name:
            adslab_id = file_name.split("_heu")[0]

        ref_energy = refE[adslab_id]

        sid = inv_mapping[file_name]
        # Extract Data object
        data_objects = read_trajectory_extract_features(a2g, system)
        relaxed_struc = data_objects[0]
        relaxed_struc.y = relaxed_struc.y - ref_energy
        # Filter data if necessary
        # OCP filters adsorption energies > |10| eV

        relaxed_struc.sid = sid  # arbitrary unique identifier
        # add tags
        parts = file_name.split("_")
        code = "_".join(parts[:3])
        relaxed_struc.tags = torch.tensor(tags[code])

        # no neighbor edge case check
        if relaxed_struc.edge_index.shape[1] == 0:
            print("no neighbors", system)
            continue
        # breakpoint()
        # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(relaxed_struc, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1

    db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file_path', type=str, help='Path to the mapping file', required=True)
    parser.add_argument('--traj_path', type=str, help='Path to the trajectory files', required=True)
    parser.add_argument('--save_path', type=str, help='Path to save the lmdb file', required=True)
    parser.add_argument('--refE_path', type=str, help='Path to the reference energies', required=True)
    parser.add_argument('--tag_path', type=str, help='Path to the tag file', required=True)
    args = parser.parse_args()

    mapping_file_path = args.mapping_file_path
    traj_path = args.traj_path
    save_path = args.save_path
    refE_path = args.refE_path
    tag_path = args.tag_path
    
    write_lmdb(save_path, traj_path, mapping_file_path, refE_path, tag_path)