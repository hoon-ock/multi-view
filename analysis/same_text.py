import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, tqdm
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices= ['escn', 'scn', 'gnoc'], required=False)
parser.add_argument('--enhancement', type=str, choices= ['ca_gap', 'ca', 'gap', 'base'], required=False)
args = parser.parse_args()
model = args.model
enhancement = args.enhancement
print("====================================")
print(f"model: {model}")
print(f"enhancement: {enhancement}")
print("====================================")

def get_system_id(row, mapping):
    id_key = int(row['id'].split('_')[0])
    return mapping[id_key]['system_id'] if id_key in mapping else None

def parity_plot2(label_dup, pred_dup, label_nondup, pred_nondup, 
                 xlabel='$\Delta E$ [eV]', ylabel='$\Delta \hat{E}$ [eV]',
                 margin=False, colors=['red', 'blue'], xylim= [-7, 2], alpha=0.5):
    '''
    label_dup, pred_dup: labels and predictions for duplicate text set
    label_nondup, pred_nondup: labels and predictions for unique text set
    xlabel: x-label name
    ylabel: y-label name
    colors: list containing colors for duplicate and unique text sets
    xylim: x and y limits for the plot
    alpha: transparency level for scatter plots
    '''
    # Calculate the error metrics for duplicate and unique text sets
    mae_dup = mean_absolute_error(label_dup, pred_dup)
    rmse_dup = np.sqrt(mean_squared_error(label_dup, pred_dup))
    r2_dup = r2_score(label_dup, pred_dup)

    mae_nondup = mean_absolute_error(label_nondup, pred_nondup)
    rmse_nondup = np.sqrt(mean_squared_error(label_nondup, pred_nondup))
    r2_nondup = r2_score(label_nondup, pred_nondup)

    # Plot
    lims = xylim

    plt.figure(figsize=(6, 6))
    plt.scatter(label_dup, pred_dup, 
                color=colors[0], alpha=alpha, 
                label=f'Duplicate text set (MAE = {mae_dup:.2f} eV)')
    plt.scatter(label_nondup, pred_nondup, 
                color=colors[1], alpha=alpha, 
                label=f'Unique text set (MAE = {mae_nondup:.2f} eV)')

    plt.plot(lims, lims, '--', c='grey')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(fontsize=14, loc='lower left')

    if not margin:
        plt.axvline(lims[1], c='k', lw=0.8)
        plt.axhline(lims[1], c='k', lw=0.8)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.show()
    return (r2_dup, r2_nondup), (mae_dup, mae_nondup), (rmse_dup, rmse_nondup)
# Example usage of the function
# parity_plot2(label_dup, pred_dup, label_nondup, pred_nondup)


# ===================== Load Mapping =====================
mapping = pd.read_pickle('/home/jovyan/ocp2023/oc20_dense_metadata/oc20dense_mapping.pkl')

model_key_mapping = {'scn': 'scn-2M', 'escn': 'escn-2M', 'gnoc': 'gemnet-oc-2M'}
checkpoint_mapping = {'base':'rebase3-vanilla-catberta_1223_1537', 
                      'ca':'rebase3-da-catberta_1224_1518', 
                      'gap':'rebase-gap-catberta_1221_2208', 
                      'ca_gap':'rebase-da-gap-catberta_1226_0542'}

# ==================== Load Data ====================
# load data
data_path = "/home/jovyan/shared-scratch/jhoon/ocp2023/clip_data"
data = pd.read_pickle(data_path + f"/oc20dense_eval_{model}_relaxed.pkl")

# load prediction
pred_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/results/ml-relax-pred/{model}/{checkpoint_mapping[enhancement]}"
pred = np.load(pred_path + "/pred.npz")
ids = pred['ids']
energy = pred['energy']
# Create a dictionary mapping from ids to energy
pred_dict = dict(zip(ids, energy))

# load dft
dft = pd.read_pickle('/home/jovyan/ocp2023/AdsorbML/adsorbml/2023_neurips_challenge/ml_relaxed_dft_targets.pkl')[model_key_mapping[model]]


# ==================== Process loaded Data ====================
preds, labels = [], []
for id in tqdm.tqdm(data['id']):
    
    sid = int(id.split('_')[0])
    system_id = mapping[sid]['system_id']
    config_id = mapping[sid]['config_id']
    label = dft[system_id][config_id]

    labels.append(label)
    preds.append(pred_dict[id])

# count label value > 100
print('Number of total data: ', len(labels))
print('Number of outliers: ', np.sum(np.array(labels) > 100))
print('Number of valid data: ', len(labels) - np.sum(np.array(labels) > 100))

# Make combined dataframe
data['label'] = labels
data['pred'] = preds
# Apply the function to each row in the dataframe to create the 'sid' column
data['sid'] = data.apply(get_system_id, mapping=mapping, axis=1)
data = data[data['label']<=100]
data.drop(columns=['chg_emb'], inplace=True)

# ==================== Find duplicates ====================
# Identifying duplicates in the 'sid' column
text_duplicates = data[data.duplicated('text', keep=False)]

# Keeping only the non-duplicates
text_non_duplicates = data[~data.duplicated('text', keep=False)]
print('------------------------------------')
print('Number of duplicates: ', len(text_duplicates))
print('Number of non-duplicates: ', len(text_non_duplicates))


# ==================== Plot ====================
labels_dup = text_duplicates['target']
preds_dup = text_duplicates['pred']

labels_nondup = text_non_duplicates['target']
preds_nondup = text_non_duplicates['pred']

r2, mae, rmse = parity_plot2(labels_dup, preds_dup, labels_nondup, preds_nondup)

save_path = f"duplicate/{model}_{enhancement}" #os.path.join('parity_plot.png')
# save the plot
plt.savefig(save_path+'.png', bbox_inches='tight', facecolor='w')

# save r2, mae, rmse as text file
with open(save_path+'.txt', 'w') as f:
    f.write(f'r2: {r2}\n')
    f.write(f'mae: {mae}\n')
    f.write(f'rmse: {rmse}\n')
print(r2, mae, rmse)