import os
import csv
import numpy as np
from math import sqrt
from collections import Counter
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import periodictable
from ase.io import read
import pandas as pd
import pickle as pkl


def molecular_formula_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)

def get_covalent_radius(element_symbol):
    element = getattr(periodictable, element_symbol, None)
    if element is not None and hasattr(element, 'covalent_radius'):
        radius_angstroms = element.covalent_radius
        return radius_angstroms
    else:
        return "Element not found or covalent radius not available"

def parse_formula(formula):
    elements = Counter()
    temp = ""
    count = ""
    for char in formula:
        if char.isalpha():
            if temp:
                elements[temp] += int(count) if count else 1
            temp = char
            count = ""
        elif char.isdigit():
            count += char
    if temp:
        elements[temp] += int(count) if count else 1
    return elements

def split_ads_cat_symbol(elements):
    # Define possible elements for adsorbate
    adsorbate_elements = {'C', 'N', 'O', 'H'}

    # Count occurrences of each element
    element_counts = Counter(elements)

    # Initialize strings to store the chemical formulas
    adsorbate_composition = []
    catalyst_composition = []

    # Classify elements and build chemical formulas
    for element, count in element_counts.items():
        if element in adsorbate_elements:
            adsorbate_composition.append(f"{element}{count if count > 1 else ''}")
        else:
            catalyst_composition.append(f"{element}{count if count > 1 else ''}")

    # Join the chemical formulas
    adsorbate_formula = ''.join(adsorbate_composition)
    catalyst_formula = ''.join(catalyst_composition)
    return adsorbate_formula, catalyst_formula

def parse_cif(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extract adsorbate information from the first line
        first_line = lines[0].strip()
        #adsorbate_info = first_line.split('</s>')[0]
        adsorbate_info = first_line.split('</s>')[0].replace('data_', '')
        cat_info = first_line.split('</s>')[1]
        miller_index_part = first_line.split('</s>')[1].strip()
        miller_index = miller_index_part[miller_index_part.find('('):]
        # Initialize lists
        atom_types = []
        positions = []
        reading_atom_site = False

        for line in lines:
            line = line.strip()

            # Start reading atom site data
            if line.startswith('_atom_site_type_symbol'):
                reading_atom_site = True
                continue
            if reading_atom_site:
                if line.startswith('_atom_site_label') or line.startswith('_atom_site_symmetry_multiplicity') or \
                   line.startswith('_atom_site_occupancy') or line.startswith('_cell_length_a') or \
                   line.startswith('_cell_length_b') or line.startswith('_cell_length_c'):
                    continue

                # Process atom site data
                if line and not line.startswith('_'):
                    parts = line.split()
                    if len(parts) >= 4:
                        atom_type = parts[0]
                        try:
                            x = float(parts[-3])
                            y = float(parts[-2])
                            z = float(parts[-1])
                            atom_types.append(atom_type)
                            positions.append((x, y, z))
                        except ValueError:
                            print(f"Skipping invalid line due to ValueError: {line}")
                            continue

        if not atom_types or not positions:
            raise ValueError("Failed to parse atom types or positions.")

        atoms = list(zip(atom_types, positions))
        # ads_symbol, cat_symbol = split_ads_cat_symbol(atom_types)
        # ads_comp1 = parse_formula(adsorbate_info)
        # ads_comp2 = parse_formula(ads_symbol)
        # if ads_comp1 != ads_comp2:
        #     adsorbate_info = ads_symbol

        # cat_info = cat_symbol + ' ' + miller_index
        return atoms, adsorbate_info, cat_info

    except Exception as e:
        print(f"Error parsing CIF file {file_path}: {e}")
        return None, None, None

def get_adsorbate_elements(adsorbate_info):
    return list(filter(str.isalpha, adsorbate_info))

def calculate_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def find_binding_atom(atoms, adsorbate_indices):
    positions = [atom[1] for atom in atoms]
    min_distance = float('inf')
    binding_atom_index = None

    for i in adsorbate_indices:
        for j in range(len(positions)):
            if j not in adsorbate_indices:
                distance = calculate_distance(positions[i], positions[j])
                if distance < min_distance:
                    min_distance = distance
                    binding_atom_index = i

    return binding_atom_index

def get_neighbors(atoms, binding_atom_index, threshold):
    positions = [atom[1] for atom in atoms]
    distances = np.zeros((len(positions), len(positions)))

    for i in range(len(positions)):
        for j in range(len(positions)):
            distances[i, j] = calculate_distance(positions[i], positions[j])

    first_neighbors = set()
    second_neighbors_dict = {}

    for i, dist in enumerate(distances[binding_atom_index]):
        if i != binding_atom_index and dist < threshold and atoms[i][0] not in ['C', 'N', 'O', 'H']:
            first_neighbors.add(i)

    for fn_idx in first_neighbors:
        second_neighbors = set()
        for i, dist in enumerate(distances[fn_idx]):
            if dist < threshold:
                second_neighbors.add(i)
        second_neighbors_dict[fn_idx] = list(second_neighbors)

    return list(first_neighbors), second_neighbors_dict

def format_neighbors(atoms, binding_atom_index, first_neighbors, second_neighbors_dict):
    binding_atom_element = atoms[binding_atom_index][0]
    first_neighbors_elements = [atoms[idx][0] for idx in first_neighbors]
    
    binding_site_desc = "atop" if len(first_neighbors) == 1 else "bridge" if len(first_neighbors) == 2 else "hollow"
    
    first_neighbors_str = " ".join(first_neighbors_elements)
    
    second_neighbors_lists = []
    adsorbate_elements = {'C', 'H', 'N', 'O'}
    
    for fn_idx in first_neighbors:
        second_neighbors_elements = [atoms[idx][0] for idx in second_neighbors_dict.get(fn_idx, [])]
        if second_neighbors_elements:
            # Ensure the binding atom is at the end of each second neighbor list
            if binding_atom_element in second_neighbors_elements:
                second_neighbors_elements.remove(binding_atom_element)
            second_neighbors_elements.append(binding_atom_element)
            
            adsorbate_atoms_in_list = [atom for atom in second_neighbors_elements if atom in adsorbate_elements]
            if adsorbate_atoms_in_list:
                for atom in adsorbate_atoms_in_list:
                    second_neighbors_elements.remove(atom)
                second_neighbors_elements.extend(adsorbate_atoms_in_list)
                
            second_neighbors_lists.append(second_neighbors_elements)

    return binding_atom_element, first_neighbors_str, binding_site_desc, second_neighbors_lists


def process_cif(file_path):
    atoms, adsorbate_info, cat_info = parse_cif(file_path)
    if atoms is None:
        return None

    ase_atoms = read(file_path)

    atom_pos = ase_atoms.get_positions()
    atom_types = ase_atoms.get_chemical_symbols()
    atoms = list(zip(atom_types, atom_pos))

    if not atoms:
        return None

    adsorbate_elements = get_adsorbate_elements(adsorbate_info)
    adsorbate_indices = [i for i, atom in enumerate(atoms) if atom[0] in adsorbate_elements]
    if not adsorbate_indices:
        return None

    binding_atom_index = find_binding_atom(atoms, adsorbate_indices)
    if binding_atom_index is None:
        return None

    binding_atom_element = atoms[binding_atom_index][0]
    covalent_radius = get_covalent_radius(binding_atom_element)
    if isinstance(covalent_radius, str):  # Check if there was an error
        print(f"Error: {covalent_radius}")
        return None

    threshold = 5 * covalent_radius
    first_neighbors, second_neighbors_dict = get_neighbors(atoms, binding_atom_index, threshold)
    if not first_neighbors:
        return f"{adsorbate_info}</s>{cat_info}</s>[]"

    binding_atom_element, first_neighbors_str, binding_site_desc, second_neighbors_lists = format_neighbors(atoms, binding_atom_index, first_neighbors, second_neighbors_dict)

    second_neighbors_str = " ".join([f"[{' '.join(lst)}]" for lst in second_neighbors_lists if lst])
    
    result_string = f"[{binding_atom_element} {first_neighbors_str} {binding_site_desc} {second_neighbors_str}]"
    return f"{adsorbate_info}</s>{cat_info}</s>{result_string}"

def create_pkl_files(csv_file, pkl_file2):

    df = pd.read_csv(csv_file, header=None)
    if df.empty:
        print("The CSV file is empty or could not be read.")
        return

    data = pd.DataFrame({
        'id': range(len(df)),
        'text': df[0] 
    })

    with open(pkl_file2, 'wb') as file2:
        pkl.dump(data, file2)

def main():
    input_folder = 'CIFS_for_conversion'
    output_file = 'LLM_strings.csv'
    output_pkl = 'LLM_strings.pkl'

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for filename in os.listdir(input_folder):
            if filename.endswith('.cif'):
                file_path = os.path.join(input_folder, filename)
                output_string = process_cif(file_path)
                if output_string:
                    writer.writerow([output_string])
                else:
                    print(f"Failed to process file {filename}")

    create_pkl_files(output_file, output_pkl)
    #os.remove(output_file)

if __name__ == "__main__":
    main()