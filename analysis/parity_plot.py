import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, tqdm
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score)

def pred_to_npz(pred_path, save_dir):
    '''
    pred_path: path to the prediction result
    save_dir: path to save the npz file
    '''
    pred = pd.read_pickle(pred_path)
    ids = list(pred.keys())
    energy = list(pred.values())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'pred.npz')
    np.savez(save_path, ids=ids, energy=energy)

def obtain_preds_and_labels(pred, model, mapping_file_path, dft_target_path):
    """
    Obtain predictions and corresponding labels.
    
    Parameters:
    pred (dict): Dictionary containing prediction values.
    model (str): Model type ('scn', 'escn', or 'gnoc'). Default is 'scn'.
    
    Returns:
    tuple: Two numpy arrays (predictions and labels).
    """
    # breakpoint()
    model_key_mapping = {'scn': 'scn-2M', 'escn': 'escn-2M', 'gnoc': 'gemnet-oc-2M'}
    map_name = model_key_mapping[model]
    oc20dense_mapping = pd.read_pickle(mapping_file_path)
    dft = pd.read_pickle(dft_target_path)[map_name]
    print('meta data loaded')

    preds, labels = [], []
    for key, value in tqdm.tqdm(pred.items()):
        
        sid = int(key.split('_')[0])
        system_id = oc20dense_mapping[sid]['system_id']
        config_id = oc20dense_mapping[sid]['config_id']
        label = dft[system_id][config_id]
        # exclude the outliers
        if label > 100:
            continue
        labels.append(label)
        preds.append(value)
    return np.array(preds), np.array(labels)


def parity_plot(label, pred, 
                plot_type, 
                xlabel='$\Delta E$ [eV]', 
                ylabel='$\Delta \hat{E}$ [eV]', 
                margin=False, color=None,
                xylim= [-7, 2]):
    '''
    targets_pred: x value
    targets_val: y value 
    residuals: targets_pred - targets_val 
    plot_type: hexabin / scatter
    xlabel: x-label name
    ylabel: y-label name
    delta: unit dE if True, unit E if False
    color: customize the color
    '''
    # Plot
    residuals = pred - label
    lims = xylim #[min(min(label), min(pred)) - 0.5, max(max(label), max(pred)) + 0.5]
    # lims = xylim
    if plot_type == 'hexabin':
        grid = sns.jointplot(x=label, 
                             y=pred,
                             kind='hex',
                             bins='log',
                             extent=lims+lims, 
                             color=color)
        
    elif plot_type == 'scatter':
        grid = sns.jointplot(x=label, 
                             y=pred, 
                             kind='scatter',
                             color=color)
        
    ax = grid.ax_joint
    _ = ax.set_xlim(lims)
    _ = ax.set_ylim(lims)
    _ = ax.plot(lims, lims, '--', c='grey')
    _ = ax.set_xlabel(f'{xlabel}', fontsize=16)
    _ = ax.set_ylabel(f'{ylabel}', fontsize=16)
    
    # Calculate the error metrics
    mae = mean_absolute_error(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    r2 = r2_score(label, pred)
    
    # Report
    text = ('\n' +
            '  $R^2$ = %.2f\n' % r2 +
            '  MAE = %.2f eV\n' % mae + 
            '  RMSE = %.2f eV\n' % rmse
            )
    
    _ = ax.text(x=lims[0], y=lims[1], s=text,
                horizontalalignment='left',
                verticalalignment='top', fontsize=14)
    if margin == False:
        grid.ax_marg_x.remove()
        grid.ax_marg_y.remove()
        plt.axvline(lims[1], c='k', lw=2.2)
        plt.axhline(lims[1], c='k', lw=1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return r2, mae, rmse

def get_parity_plots(pred_path, save_dir, model, mapping_file_path, dft_target_path):
    pred = pd.read_pickle(pred_path)
    print('obtain the predictions and labels')
    preds, labels = obtain_preds_and_labels(pred, model, mapping_file_path, dft_target_path)
    print('plotting the parity plot')
    r2, mae, rmse = parity_plot(labels, preds, plot_type='scatter')
    # save the plot
    save_path = os.path.join(save_dir, 'parity_plot.png')
    plt.savefig(save_path, bbox_inches='tight', facecolor='w')
    plt.close()
    return r2, mae, rmse


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, help='Path to prediction result pickle file', required=True)
    parser.add_argument('--save_dir', type=str, help='Path to save the postprocessed results', required=True)
    parser.add_argument('--mapping_file_path', type=str, help='Path to the mapping file', required=True)
    parser.add_argument('--dft_target_path', type=str, help='Path to the DFT target for ML-relaxed structures pickle file', required=True)
    parser.add_argument('--model', type=str, choices= ['eqv2', 'escn', 'scn', 'gnoc'], required=False)
    args = parser.parse_args()

    pred_path = args.pred_path
    save_dir = args.save_dir
    model = args.model
    mapping_file_path = args.mapping_file_path 
    dft_target_path = args.dft_target_path 
    # postprocessing for prediction result
    pred_to_npz(pred_path, save_dir)

    # parity plots for prediction result
    r2, mae, rmse = get_parity_plots(pred_path, save_dir, model, mapping_file_path, dft_target_path)
    print(f"R2: {r2} | MAE: {mae} eV | RMSE: {rmse} eV")