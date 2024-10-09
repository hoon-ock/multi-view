import os
import pandas as pd
import pickle
import csv
from collections import Counter
from ase.io import read
from ase.data import chemical_symbols
from ase import Atoms
import shutil

def normalize_counts(element_counts):
    normalized = []
    for elem, count in element_counts:
        if count > 1:
            normalized.append(f'{elem}{count}')
        else:
            normalized.append(elem)
    return ''.join(normalized)

def get_element_counts_from_formula(formula):
    atoms = Atoms(formula)
    def get_element_counts_from_ase(atoms):
        element_counts = Counter()
        for atom in atoms:
            element_counts[atom.symbol] += 1
        return tuple(sorted(element_counts.items()))
    return get_element_counts_from_ase(atoms)

def process_data(pkl_file, cif_folder, output_csv, cifs_for_conversion_dir):
    data = pd.read_pickle(pkl_file)
    energy_dict = {}

    os.makedirs(cifs_for_conversion_dir, exist_ok=True)

    for index, row in data.iterrows():
        text = row['text']
        target = row['target']
        parts = text.split('</s>')
        adsorbate = parts[0].strip()
        catalyst = parts[1].strip()
        configuration = parts[2].strip() if len(parts) > 2 else ''  # Extract configuration

        adsorbate_catalyst = f"{adsorbate}</s>{catalyst}"
        adsorbate_counts = get_element_counts_from_formula(adsorbate)
        catalyst_counts = get_element_counts_from_formula(catalyst.split(' ')[0])
        adsorbate_catalyst_counts = (adsorbate_counts, catalyst_counts)

        if adsorbate_catalyst_counts not in energy_dict:
            energy_dict[adsorbate_catalyst_counts] = []
        energy_dict[adsorbate_catalyst_counts].append((target, configuration))  # Store target and configuration

    cif_dict = {}

    for filename in os.listdir(cif_folder):
        if filename.endswith('.cif'):
            file_path = os.path.join(cif_folder, filename)

            with open(file_path, 'r') as file:
                first_line = file.readline().strip()
                if first_line.startswith('data_'):
                    adsorbate_catalyst = first_line.replace('data_', '').strip()
                    adsorbate, catalyst_with_miller = adsorbate_catalyst.split('</s>')
                    catalyst_parts = catalyst_with_miller.split(' ')
                    catalyst = catalyst_parts[0]
                    miller_index = ' '.join(catalyst_parts[1:])

                    adsorbate_counts = get_element_counts_from_formula(adsorbate)
                    catalyst_counts = get_element_counts_from_formula(catalyst)
                    adsorbate_catalyst_counts = (adsorbate_counts, catalyst_counts)

                    normalized_adsorbate_catalyst = f"{normalize_counts(adsorbate_counts)}</s>{normalize_counts(catalyst_counts)} {miller_index}"

                    cif_dict[adsorbate_catalyst_counts] = (normalized_adsorbate_catalyst, file_path)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Adsorbate-Catalyst', 'Configurations', 'Energies'])

        for adsorbate_catalyst_counts, targets_and_configs in energy_dict.items():
            if adsorbate_catalyst_counts in cif_dict:
                original_adsorbate_catalyst, cif_file_path = cif_dict[adsorbate_catalyst_counts]
                
                shutil.copy(cif_file_path, os.path.join(cifs_for_conversion_dir, os.path.basename(cif_file_path)))
                
                for target, config in targets_and_configs:
                    writer.writerow([original_adsorbate_catalyst, config, target])

def create_pkl_files(csv_file, pkl_file1, pkl_file2):
    df = pd.read_csv(csv_file)

    if not {'Adsorbate-Catalyst', 'Configurations'}.issubset(df.columns):
        raise ValueError("CSV file must contain 'Adsorbate-Catalyst' and 'Configurations' columns.")

    unique_adsorbate_catalyst_df = pd.DataFrame({
        'id': range(len(df['Adsorbate-Catalyst'].unique())),
        'text': df['Adsorbate-Catalyst'].unique()
    })

    df['Adsorbate-Catalyst-Config'] = df.apply(lambda row: f"{row['Adsorbate-Catalyst']}</s>{row['Configurations']}", axis=1)
    adsorbate_catalyst_config_df = pd.DataFrame({
        'id': df.index,
        'text': df['Adsorbate-Catalyst-Config'].tolist()
    })

    with open(pkl_file1, 'wb') as file1:
        pickle.dump(unique_adsorbate_catalyst_df, file1)
    with open(pkl_file2, 'wb') as file2:
        pickle.dump(adsorbate_catalyst_config_df, file2)

def reorder_pkl_files(pkl_file1, pkl_file2, cifs_for_conversion_dir):
    # Load the existing PKL files
    with open(pkl_file1, 'rb') as f1, open(pkl_file2, 'rb') as f2:
        adsorbate_catalyst_df = pickle.load(f1)
        adsorbate_catalyst_config_df = pickle.load(f2)

    # List all .cif files in the cifs_for_conversion directory
    cifs_for_conversion_files = [f for f in os.listdir(cifs_for_conversion_dir) if f.endswith('.cif')]

    # Use ASE matching to reorder entries based on the CIF file order
    reordered_adsorbate_catalyst = []
    reordered_adsorbate_catalyst_config = []

    for cif_file in cifs_for_conversion_files:
        file_path = os.path.join(cifs_for_conversion_dir, cif_file)
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            if first_line.startswith('data_'):
                adsorbate_catalyst = first_line.replace('data_', '').strip()
                adsorbate, catalyst_with_miller = adsorbate_catalyst.split('</s>')
                catalyst_parts = catalyst_with_miller.split(' ')
                catalyst = catalyst_parts[0]
                miller_index = ' '.join(catalyst_parts[1:])

                # Get counts for matching
                adsorbate_counts = get_element_counts_from_formula(adsorbate)
                catalyst_counts = get_element_counts_from_formula(catalyst)

                # Normalize adsorbate-catalyst to compare with existing PKL file entries
                normalized_adsorbate_catalyst = f"{normalize_counts(adsorbate_counts)}</s>{normalize_counts(catalyst_counts)} {miller_index}"

                # Match the normalized counts with PKL entries for pkl_file1
                for idx, row in adsorbate_catalyst_df.iterrows():
                    if normalized_adsorbate_catalyst in row['text']:
                        reordered_adsorbate_catalyst.append(row)
                        break

                # Match for the adsorbate-catalyst-config PKL (pkl_file2)
                matching_configs = adsorbate_catalyst_config_df[
                    adsorbate_catalyst_config_df['text'].apply(lambda x: normalized_adsorbate_catalyst in x)
                ]
                
                if not matching_configs.empty:
                    reordered_adsorbate_catalyst_config.extend(matching_configs.to_dict('records'))  # Retain all matches

    # Create reordered DataFrames
    reordered_adsorbate_catalyst_df = pd.DataFrame(reordered_adsorbate_catalyst)
    reordered_adsorbate_catalyst_config_df = pd.DataFrame(reordered_adsorbate_catalyst_config)

    # Save the reordered entries back to the PKL files
    with open(pkl_file1, 'wb') as f1, open(pkl_file2, 'wb') as f2:
        pickle.dump(reordered_adsorbate_catalyst_df, f1)
        pickle.dump(reordered_adsorbate_catalyst_config_df, f2)

    print("PKL files reordered based on CIF file order, retaining all entries for pkl_file2.")

def reorder_csv_file(csv_file, cifs_for_conversion_dir, output_csv_reordered):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # List all .cif files in the cifs_for_conversion directory
    cifs_for_conversion_files = [f for f in os.listdir(cifs_for_conversion_dir) if f.endswith('.cif')]
    
    # Prepare to reorder the CSV
    reordered_rows = []
    
    for cif_file in cifs_for_conversion_files:
        file_path = os.path.join(cifs_for_conversion_dir, cif_file)
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            if first_line.startswith('data_'):
                adsorbate_catalyst = first_line.replace('data_', '').strip()
                adsorbate, catalyst_with_miller = adsorbate_catalyst.split('</s>')
                catalyst_parts = catalyst_with_miller.split(' ')
                catalyst = catalyst_parts[0]
                
                # Get counts for matching
                adsorbate_counts = get_element_counts_from_formula(adsorbate)
                catalyst_counts = get_element_counts_from_formula(catalyst)

                # Normalize adsorbate-catalyst to compare with existing DataFrame entries
                normalized_adsorbate_catalyst = f"{normalize_counts(adsorbate_counts)}</s>{normalize_counts(catalyst_counts)}"
                
                # Get all matching rows based on normalized adsorbate-catalyst
                matching_rows = df[df['Adsorbate-Catalyst'].apply(lambda x: normalized_adsorbate_catalyst in x)]
                
                if not matching_rows.empty:
                    reordered_rows.extend(matching_rows.to_dict('records'))  # Retain all matches
    
    # Create the reordered DataFrame
    reordered_df = pd.DataFrame(reordered_rows)
    
    # Save the reordered DataFrame to a new CSV file
    reordered_df.to_csv(output_csv_reordered, index=False)
    print(f"Reordered CSV file saved as {output_csv_reordered}.")


# Example usage
# process_data('data.pkl', 'cif_folder', 'output.csv', 'CIFS_for_conversion')
# create_pkl_files('input.csv', 'pkl_file1.pkl', 'pkl_file2.pkl')
# reorder_pkl_files('pkl_file1.pkl', 'pkl_file2.pkl', 'CIFS_for_conversion')
# reorder_csv_file('output.csv', 'CIFS_for_conversion', 'reordered_output.csv')

def extract_energies(pkl_file, cif_folder='valid_cifs'):
    #pkl_file = 'oc20dense_train_rev.pkl'
    #cif_folder = 'valid_cifs'
    output_csv = 'DFT_energies.csv'
    pkl_file1 = 'adsorbate_catalyst_GT.pkl'
    pkl_file2 = 'adsorbate_catalyst_config_GT.pkl'
    cifs_for_conversion_dir = 'CIFS_for_conversion'
    reordered_output_csv = 'DFT_energies_reordered.csv'

    process_data(pkl_file, cif_folder, output_csv, cifs_for_conversion_dir)

    create_pkl_files(output_csv, pkl_file1, pkl_file2)

    reorder_pkl_files(pkl_file1, pkl_file2, cifs_for_conversion_dir)

    reorder_csv_file(output_csv, cifs_for_conversion_dir, reordered_output_csv)

    cif_files = [f for f in os.listdir(cif_folder) if f.endswith('.cif')]
    print(f"Number of .cif files in valid_cifs directory: {len(cif_files)}")

    cif_files_new = [f for f in os.listdir(cifs_for_conversion_dir) if f.endswith('.cif')]
    print(f"Number of .cif files in cifs_for_conversion_dir directory: {len(cif_files_new)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the inference data file.")
    parser.add_argument("--cif_dir", type=str, default='valid_cifs', help="Path to the directory containing valid CIF files.")
    args = parser.parse_args()
    extract_energies(args.data_path, args.cif_dir)
