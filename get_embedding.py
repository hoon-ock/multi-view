import torch
from torch.utils.data import DataLoader
from dataset import RegressionDataset
from model.models import RegressionModel
from model.modules import TextEncoder
import numpy as np
import pandas as pd
import os, yaml, pickle
from transformers import RobertaTokenizerFast, RobertaModel
import tqdm
# Import your RegressionModel class, train_fn, and any other necessary modules

# Define a function for making predictions
def predict_fn(data_loader, model, device):
    model.eval()  # Put the model in evaluation mode.
    predictions = []

    with torch.no_grad():  # Disable gradient calculation.
        for batch in tqdm.tqdm(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch).squeeze(-1)
            predictions.extend(outputs.cpu().numpy())  # Store predictions as numpy arrays
            # breakpoint()
    return np.array(predictions)



def get_embedding(data_path, pt_ckpt_dir_path, save_path, tag, debug=False):  
    # with open(os.path.join(pt_ckpt_dir_path, "clip.yml"), "r") as f:
    #     config = yaml.safe_load(f)
    # Hyperparameters and settings   
    
   
    ############################################################################
    ckpt_name = pt_ckpt_dir_path.split('/')[-1]
    # if 'roberta-base' not in pt_ckpt_dir_path:
    
    pt_ckpt_path = os.path.join(pt_ckpt_dir_path, "checkpoint.pt")
    model_config_path = os.path.join(pt_ckpt_dir_path, "clip.yml") #config["model_config"]
    if 'roberta-base' in pt_ckpt_dir_path:
        model_config_path = "model/clip.yml"
    ############################################################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug:
        device = "cpu"
    batch = 32
    
    print("=============================================================")
    print(f"Generating embedding with {ckpt_name}")
    print("=============================================================")

    
    # ========================= DATA LOADING =================================
    # Load train and validation data 
    df_test = pd.read_pickle(data_path)
    
    if debug:
        df_test = df_test.sample(10)
        
    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    # breakpoint()
    # Initialize training dataset
    test_dataset = RegressionDataset(texts = df_test["text"].values,
                                      targets = df_test["target"].values,
                                      chg_emb = df_test["chg_emb"].values,
                                      tokenizer = tokenizer,
                                      seq_len= tokenizer.model_max_length)
    # Create training dataloader
    test_data_loader = DataLoader(test_dataset, batch_size = batch,
                                  shuffle = False, num_workers=1)

    #breakpoint()    
    # ===================== MODEL and TOKENIZER ===============================
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    # breakpoint()
    print('loading pretrained text encoder and projection layer from')
    print(ckpt_name)
    if 'roberta-base' not in pt_ckpt_dir_path:
        
    
        #model = RegressionModel(model_config).to(device)
        model = TextEncoder(model_config).to(device)
        
        if 'clip_ssl' not in pt_ckpt_dir_path:
            state_dict = torch.load(pt_ckpt_path, map_location=device)['model_state_dict']
        elif 'clip_ssl' in pt_ckpt_dir_path:
            state_dict = torch.load(pt_ckpt_path, map_location=device)

        prefix = 'text_encoder.'  # Modify this to match the actual prefix in your checkpoint
        # breakpoint()
        new_state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        # breakpoint()
        model.load_state_dict(new_state_dict, strict=True) 
        #breakpoint()
    elif 'roberta-base' in pt_ckpt_dir_path:
        # with open('model/clip.yml', "r") as f:
        #     model_config = yaml.safe_load(f)
        model_config['Path']['pretrain_ckpt'] = "roberta-base"         
        model = TextEncoder(model_config).to(device)

    # ========================= PREDICTION ====================================
    predictions = predict_fn(test_data_loader, model, device)
    #breakpoint()
    # ========================= SAVE PREDICTION ===============================
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    save_path = os.path.join(save_path, f"emb-{ckpt_name}-{tag}-strc.pkl")
    
    # df_test['pred'] = predictions
    # df_test.to_pickle(save_path) 
    
    ## save as dictionary where key is id in df_test, and values are predictions
    predictions_dict = dict(zip(df_test["id"].values, predictions))

    with open(save_path, "wb") as f:
        pickle.dump(predictions_dict, f)
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["train-combined", "train-oc20","eval","test"], default="test")
    parser.add_argument("--model", type=str, choices=["gnoc", "escn", "scn", "eqv2"], default="eqv2")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--train_type", type=str, choices=["direct_regress", "clip_regress", "clip_ssl"], default=None)
    # parser.add_argument("--data_path", type=str, default=None)    
    # parser.add_argument("--pt_ckpt_dir_path", type=str, default=None)
    # parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    split = args.split
    model = args.model
    ckpt = args.ckpt
    train_type = args.train_type
    debug = args.debug
    # Conduct prediction
    tag = f"{split}-{model}"
    if split == "train-combined":
        data_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/clip_data/oc20_oc20dense_train_relaxed.pkl"
        tag = split
    elif split == "train-oc20":
        data_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/clip_data/oc20dense_train_relaxed.pkl"
        tag = split
    elif split == "eval" or split == "test":
        data_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/clip_data/oc20dense_{split}_{model}_relaxed.pkl"
    pt_ckpt_dir_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/checkpoints/{train_type}/{ckpt}"
    save_path = "/home/jovyan/shared-scratch/jhoon/ocp2023/results/encoder_embedding" 
    
    print("=============================================================")
    print(f"Making predictions for {split} split of {model} model")
    get_embedding(data_path, pt_ckpt_dir_path, save_path, tag, debug)