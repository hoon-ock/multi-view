import os
import re
import csv
import tarfile
from collections import Counter, defaultdict
from ase.io import read
from ase import Atoms

def count_elements(string):
    ase_atoms = Atoms(string)
    element_counts = Counter(ase_atoms.get_chemical_symbols())
    return element_counts

def extract_atom_counts(cif_content):
    atom_counts = Counter()
    lines = cif_content.split('\n')
    
    atom_section_started = False
    for line in lines:
        if '_atom_site_type_symbol' in line:
            atom_section_started = True
            continue
        
        if atom_section_started:
            if line.startswith('loop_') or not line.strip():
                break  # End of the atom section
            parts = line.split()
            if parts:
                atom_symbol = parts[0]
                if not atom_symbol.startswith('_atom'):
                    atom_counts[atom_symbol] += 1
    return atom_counts


def validate_counts(formula_counts, cif_counts, before_s_elements, after_s_elements, tolerance_before, tolerance_after):
    for element in before_s_elements:
        if abs(formula_counts[element] - cif_counts[element]) > tolerance_before:
            return False
    for element in after_s_elements:
        if abs(formula_counts[element] - cif_counts[element]) > tolerance_after:
            return False
    return True

def process_cif_file(cif_content):
    lines = cif_content.split('\n')
    
    if not lines:
        return None, None
    
    first_line = lines[0].strip().replace('data_', '')
    
    adsorbate, rest = first_line.split('</s>')
    catalyst_formula, _ = rest.split('(')
    catalyst_formula = catalyst_formula.strip()

    adsorbate_elements = count_elements(adsorbate)
    catalyst_elements = count_elements(catalyst_formula)
    
    before_s_elements = [el[0] for el in adsorbate_elements]
    after_s_elements = [el[0] for el in catalyst_elements]
    
    formula_counts = count_elements(adsorbate) + count_elements(catalyst_formula)
    cif_counts = extract_atom_counts(cif_content)

    is_valid = validate_counts(formula_counts, cif_counts, before_s_elements, after_s_elements, tolerance_before=0, tolerance_after=12)
    
    return first_line, is_valid

# def main(tar_gz_path, num_generations):
#     valid_cif_dir = 'valid_cifs'
#     os.makedirs(valid_cif_dir, exist_ok=True)
#     csv_file_path = 'valid_cifs.csv'

#     seen_structures = set()
#     structure_files = defaultdict(list)

#     with tarfile.open(tar_gz_path, 'r:gz') as tar:
#         for member in tar.getmembers():
#             if member.isfile() and member.name.endswith('.cif'):
#                 cif_content = tar.extractfile(member).read().decode('utf-8')
#                 first_line, is_valid = process_cif_file(cif_content)
                
#                 if first_line:
#                     base_name = member.name.split('__')[0]
#                     generation_number = int(member.name.split('__')[1].split('.')[0])
#                     structure_files[base_name].append((generation_number, member.name, cif_content, is_valid))
    
#     chosen_structures = {}
#     for base_name, versions in structure_files.items():
#         versions.sort(key=lambda x: x[0])
        
#         for _, member_name, cif_content, is_valid in versions:
#             if is_valid:
#                 chosen_structures[base_name] = (member_name, cif_content)
#                 break
#         else:
#             chosen_structures[base_name] = versions[0][1], versions[0][2]

#     with open(csv_file_path, 'w', newline='') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(['Structure'])

#         with tarfile.open('valid_cifs.tar.gz', 'w:gz') as tar_output:
#             for base_name, (member_name, cif_content) in chosen_structures.items():
#                 first_line, is_valid = process_cif_file(cif_content)
                
#                 if is_valid:
#                     csvwriter.writerow([first_line])
#                     valid_cif_path = os.path.join(valid_cif_dir, os.path.basename(member_name))

#                     with open(valid_cif_path, 'w') as f:
#                         f.write(cif_content)

#                     tar_output.add(valid_cif_path, arcname=os.path.basename(valid_cif_path))

#     #shutil.rmtree(valid_cif_dir)
#     os.remove(csv_file_path)

def main(tar_gz_path, num_generations):
    valid_cif_dir = 'valid_cifs'
    os.makedirs(valid_cif_dir, exist_ok=True)
    csv_file_path = 'valid_cifs.csv'

    # seen_structures = set()
    structure_files = defaultdict(list)
    file_count = 0

    # Extract and process CIF files from tar.gz
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.cif'):
                cif_content = tar.extractfile(member).read().decode('utf-8')
                first_line, is_valid = process_cif_file(cif_content)
                
                if first_line:
                    base_name = member.name.split('__')[0]
                    generation_number = int(member.name.split('__')[1].split('.')[0])
                    structure_files[base_name].append((generation_number, member.name, cif_content, is_valid))
    
    # Select the valid CIF file for each structure
    chosen_structures = {}
    for base_name, versions in structure_files.items():
        versions.sort(key=lambda x: x[0])
        
        for _, member_name, cif_content, is_valid in versions:
            if is_valid:
                chosen_structures[base_name] = (member_name, cif_content)
                break
        else:
            chosen_structures[base_name] = versions[0][1], versions[0][2]

    # Write the valid CIFs to a CSV and save the valid CIF files
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Structure'])

        for base_name, (member_name, cif_content) in chosen_structures.items():
            first_line, is_valid = process_cif_file(cif_content)
            
            if is_valid:
                csvwriter.writerow([first_line])
                valid_cif_path = os.path.join(valid_cif_dir, os.path.basename(member_name))

                with open(valid_cif_path, 'w') as f:
                    f.write(cif_content)
                
                file_count+=1

    # Optionally, remove the temporary CSV file after use
    os.remove(csv_file_path)
    print(f"{file_count} valid CIF files saved in the {valid_cif_dir} directory")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the inference data file.")
    parser.add_argument("--num_gens", type=int, default=3, help="Number of generations in the dataset.")
    args = parser.parse_args()
    #tar_gz_path = 'inference.tar.gz'  # Change this to the path where your tar.gz file is located
    #num_generations = 3  # Change this to the number of generations in your dataset
    main(args.data_path, args.num_gens)