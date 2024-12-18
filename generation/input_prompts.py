import os
import re
import tarfile
import pickle
from collections import Counter
from ase.io import read
from ase import Atoms
import shutil
import pandas as pd

def count_elements(string):
    ase_atoms = Atoms(string)
    element_counts = Counter(ase_atoms.get_chemical_symbols())
    return element_counts

def process_and_filter_data(data, max_adsorbate_elements, max_adsorbate_atoms, max_catalyst_elements, max_catalyst_atoms):
    filtered_data = []
    seen_entries = set()
    
    for entry in data:
        entry = entry.replace('data_', '')
        adsorbate, rest = entry.split('</s>')
        catalyst_formula, miller_index = rest.split('(')
        miller_index = miller_index.strip(')')
        catalyst_formula = catalyst_formula.strip()
        
        adsorbate_atoms = adsorbate
        catalyst_atoms = catalyst_formula
        
        if adsorbate_atoms is None or catalyst_atoms is None:
            continue
        
        adsorbate_elements = count_elements(adsorbate_atoms)
        catalyst_elements = count_elements(catalyst_atoms)
        
        if (len(adsorbate_elements) <= max_adsorbate_elements and
            len(catalyst_elements) <= max_catalyst_elements and
            sum(adsorbate_elements.values()) <= max_adsorbate_atoms and
            sum(catalyst_elements.values()) <= max_catalyst_atoms):
            
            if entry not in seen_entries:
                seen_entries.add(entry)
                filtered_data.append(entry)
    
    return filtered_data

def extract_second_lines(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            txt_files = pickle.load(f)
    except:
        txt_files = pd.read_pickle(pkl_path)

    second_lines = []
    for filename, content in txt_files:
        lines = content.split('\n')
        if len(lines) > 1:
            second_line = lines[1]
        else:
            second_line = ""
        second_lines.append(second_line)

    return second_lines

def extract_input_prompts(pkl_path):
    # Load CIF files from the pickle file and extract second lines
    # pkl_path = 'oc20dense_train_prep_str.pkl'
    second_lines = extract_second_lines(pkl_path)

    # Filter the data based on the criteria
    max_adsorbate_elements = 3
    max_adsorbate_atoms = 5
    max_catalyst_elements = 3
    max_catalyst_atoms = 72
    filtered_data = process_and_filter_data(second_lines, max_adsorbate_elements, max_adsorbate_atoms, max_catalyst_elements, max_catalyst_atoms)

    # Create a directory to save filtered CIF files
    output_dir = 'chosen_prompts'
    os.makedirs(output_dir, exist_ok=True)

    # Save filtered CIF files
    unique_names = set()
    for i, string in enumerate(filtered_data):
        base_name = f'chosen_{i}.txt'
        unique_name = base_name
        counter = 1
        while unique_name in unique_names:
            unique_name = f'chosen_{i}_{counter}.txt'
            counter += 1
        unique_names.add(unique_name)
        
        with open(os.path.join(output_dir, unique_name), 'w') as f:
            f.write('data_' + string)

    # Create a tar.gz archive of the filtered CIF files
    with tarfile.open('input_prompts.tar.gz', 'w:gz') as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))

    # Include a line to delete the folder
    shutil.rmtree(output_dir)

    print(f"Chose {len(filtered_data)} prompt files archived as 'chosen_prompts.tar.gz'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    args = parser.parse_args()

    data_path = args.data_path
    extract_input_prompts(data_path)
