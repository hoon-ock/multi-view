import os
import pandas as pd
import glob
import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--string_dir', type=str, help='Path to the string directory', required=False, default='oc20/')
parser.add_argument('--save_path', type=str, help='Path to save the strings', required=True)

args = parser.parse_args()
string_dir = args.string_dir
save_path = args.save_path

id_list, string_list, energy_list = [], [], []

file_list = glob.glob(os.path.join(string_dir, "*.pkl"))

for file in tqdm.tqdm(file_list, total=len(file_list)):
    # breakpoint()
    id = file.split("/")[-1].split(".")[0]
    data = pd.read_pickle(file)
    id_list.append(id)
    string_list.append(data["string"])
    energy_list.append(data["energy"])


df = pd.DataFrame({"id": id_list, 
                   "text": string_list, 
                   "target": energy_list})
df.to_pickle(save_path)