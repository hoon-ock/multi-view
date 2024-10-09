import pandas as pd
import numpy as np
import pickle
import ase.io
from site_analysis import SiteAnalyzer
from ocpmodels.datasets import LmdbDataset
from ase import Atoms, cell, constraints
from collections import Counter
import torch
import tqdm, glob, os

class StringGenerator():
    def __init__(self, lmdb_path, metadata_path, save_path, mode):
        self.lmdb_data = LmdbDataset({'src': lmdb_path})
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        self.save_path = save_path
        self.mode = mode
        # self.tags = pickle.load(open('/home/jovyan/ocp2023/oc20_dense_metadata/oc20dense_tags.pkl', 'rb'))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # breakpoint()

    
    def get_string_from_lmdb(self):
        result = {}
        # breakpoint()
        for i in tqdm.tqdm(range(len(self.lmdb_data))):
            pyg = self.lmdb_data[i]
            id = pyg.sid
            # if int(id) != 24:
            #     continue
            # print('id: ', id)
            if self.mode == 'eval':
                frame = pyg.fid
                save_name = str(id) + '_' + str(frame) + '.pkl'
            else:
                save_name = str(id) + '.pkl'
            #energy = pyg.y_relaxed
            energy = pyg.y
            
            atoms = self.pyg2atoms(pyg)
            ads_sym, _, miller_idx, system_id = self.get_info_from_metadata(id)
            _, _, bulk_sym = self.atoms2composition(atoms)
            # save_name = system_id + '.pkl'

            interactions = self.get_interactions(atoms)
            string = self.get_string(ads_sym, bulk_sym, miller_idx, interactions)
            
            result.update({'ads_sym': ads_sym, 
                           'bulk_sym': bulk_sym, 
                           'miller_idx': miller_idx,
                           'interactions': interactions,
                           'string': string,
                           'energy': energy})
            pickle.dump(result, open(self.save_path + '/' + save_name, 'wb'))
        
    
    @staticmethod
    def pyg2atoms(pyg):
        '''Convert a pytorch geometric data object to an ASE atoms object.'''

        atoms = Atoms(
                numbers = pyg.atomic_numbers,
                positions = pyg.pos, #pyg.pos_relaxed,
                cell = cell.Cell(pyg.cell.squeeze(0).numpy()),
                pbc = True,
                tags = pyg.tags)
        # breakpoint()

        fixed_atom_indices = torch.nonzero(pyg.fixed == 1).squeeze().tolist()
        fix_atoms = constraints.FixAtoms(indices=fixed_atom_indices)
        atoms.set_constraint(fix_atoms)
        return atoms


    def get_info_from_metadata(self, id):
        #breakpoint()
        ads_sym = self.metadata[id]['adsorbate']
        # bulk_sym = self.metadata[id]['bulk_symbols']
        bulk_mpid = self.metadata[id]['mpid']
        miller_idx = self.metadata[id]['miller_idx']
        system_id = self.metadata[id]['system_id']
        return ads_sym, bulk_mpid, miller_idx, system_id

    @staticmethod
    def atoms2composition(atoms):
        '''Convert an ASE atoms object to a chemical formula.'''
        # Initialize dictionaries to store compositions
        adsorbate_composition = Counter()
        surface_composition = Counter()
        bulk_composition = Counter()

        # Initialize strings to store chemical formulas
        adsorbate_formula = ""
        surface_formula = ""
        bulk_formula = ""

        for atom in atoms:
            if atom.tag == 1:
                surface_composition[atom.symbol] += 1
                bulk_composition[atom.symbol] += 1
            elif atom.tag == 2:
                adsorbate_composition[atom.symbol] += 1
            elif atom.tag == 0:
                bulk_composition[atom.symbol] += 1


        # Compose chemical formulas
        adsorbate_formula = ''.join([f"{element}{count}" if count > 1 else element for element, count in adsorbate_composition.items()])
        surface_formula = ''.join([f"{element}{count}" if count > 1 else element for element, count in surface_composition.items()])
        bulk_formula = ''.join([f"{element}{count}" if count > 1 else element for element, count in bulk_composition.items()])
        return adsorbate_formula, surface_formula, bulk_formula
   
    @staticmethod
    def get_string(ads_comp, surf_comp, miller_index, interactions):

        string = ads_comp +'</s>'+ surf_comp +' '+ str(miller_index) +'</s>'
        if len(interactions) == 0:
            string += '[]'
        else:
            for bond in interactions:
                string += str(bond).replace("'","")
        string = string.replace(',', '')
        string = string.replace('*', '')
        return string
    
    @staticmethod
    def get_interactions(atoms):
        '''
        interaction: [ads atom, (slab atom, bond length)... , type], [...]
        '''

        data = SiteAnalyzer(adslab=atoms)
        #bond_lengths = data.get_adsorbate_bond_lengths()['adsorbate-surface']
        
        interactions = []
        for bond in data.binding_info:
            bundle = [bond['adsorbate_element']]
            # lengths = bond_lengths[bond['adsorbate_idx']]
            # lengths = [round(l, 2) for l in lengths]        
            # bundle += [(e, l) for e, l in zip(bond['slab_atom_elements'], lengths)]
            bundle += [e for e in bond['slab_atom_elements']]
            if len(bond['slab_atom_elements']) ==1:
                type = 'atop'
            elif len(bond['slab_atom_elements']) ==2:
                type = 'bridge'
            else:
                type = 'hollow'
            bundle.append(type)
            ########## for secondary interaction ############
            for idx in bond['slab_atom_idxs']:
                second_int = data.second_binding_info[idx]
                second_int = [second_int['slab_element']] + second_int['second_interaction_element']
                bundle.append(second_int)        
            ################################################
            interactions.append(bundle)
        return interactions
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', type=str, help='Path to the lmdb file', required=True)
    parser.add_argument('--save_path', type=str, help='Path to save the strings', required=False, default='dense/')
    parser.add_argument('--metadata_path', type=str, help='Path to the metadata file', required=True)
    #parser.add_argument('--tag_path', type=str, help='Path to the tag file', required=True)
    parser.add_argument('--split', type=str, choices= ['train', 'val', 'eval', 'test'])  
    args = parser.parse_args()

    lmdb_path = args.lmdb_path
    save_path = args.save_path
    metadata_path = args.metadata_path
    #tag_path = args.tag_path
    split = args.split

    print('LMDB: ', lmdb_path)
    string_generator = StringGenerator(lmdb_path, metadata_path, save_path, mode=split)
    string_generator.get_string_from_lmdb()