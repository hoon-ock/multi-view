import torch
from torch.utils.data import DataLoader
from dataset import RegressionDataset
from model.modules import TextEncoder
import numpy as np
import pandas as pd
import os, yaml, pickle
from transformers import RobertaTokenizerFast
import tqdm

# Define a function for making predictions
def predict_fn(data_loader, model, device):
    model.eval()  # Put the model in evaluation mode.
    predictions = []

    with torch.no_grad():  # Disable gradient calculation.
        for batch in tqdm.tqdm(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch).squeeze(-1)
            predictions.extend(outputs.cpu().numpy())  # Store predictions as numpy arrays
    return np.array(predictions)


def get_embedding(data_path, pt_ckpt_dir_path, save_path, tag, debug=False):  
    # Hyperparameters and settings     
    ############################################################################
    ckpt_name = pt_ckpt_dir_path.split('/')[-1]
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
    # Initialize training dataset
    test_dataset = RegressionDataset(texts = df_test["text"].values,
                                      targets = df_test["target"].values,
                                      tokenizer = tokenizer,
                                      seq_len= tokenizer.model_max_length)
    # Create training dataloader
    test_data_loader = DataLoader(test_dataset, batch_size = batch,
                                  shuffle = False, num_workers=1)
 
    # =============================== MODEL ===============================
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    print('loading pretrained text encoder and projection layer from')
    print(ckpt_name)
    if 'roberta-base' not in pt_ckpt_dir_path:
        model = TextEncoder(model_config).to(device)
        
        if 'clip_ssl' not in pt_ckpt_dir_path:
            state_dict = torch.load(pt_ckpt_path, map_location=device)['model_state_dict']
        elif 'clip_ssl' in pt_ckpt_dir_path:
            state_dict = torch.load(pt_ckpt_path, map_location=device)

        prefix = 'text_encoder.'  # Modify this to match the actual prefix in your checkpoint
        new_state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        model.load_state_dict(new_state_dict, strict=True) 

    elif 'roberta-base' in pt_ckpt_dir_path:
        model_config['Path']['pretrain_ckpt'] = "roberta-base"         
        model = TextEncoder(model_config).to(device)

    # ========================= PREDICTION ====================================
    predictions = predict_fn(test_data_loader, model, device)

    # ========================= SAVE PREDICTION ===============================
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f"emb-{ckpt_name}-{tag}-strc.pkl")

    ## save as dictionary where key is id in df_test, and values are predictions
    predictions_dict = dict(zip(df_test["id"].values, predictions))

    with open(save_path, "wb") as f:
        pickle.dump(predictions_dict, f)
 
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(description="Script to get embeddings.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--pt_ckpt_dir_path", type=str, required=True, help="Path to the pretrained checkpoint directory.")
    parser.add_argument("--save_path", type=str, required=True, help="Path where to save the embeddings.")
    parser.add_argument("--tag", type=str, default=datetime.now().strftime("%y%m%d_%H%M%S"), help="Tag for the run, defaults to current date and time if not provided.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode if set.")
    args = parser.parse_args()
    
    # Directly use the provided paths and tag
    data_path = args.data_path
    pt_ckpt_dir_path = args.pt_ckpt_dir_path
    save_path = args.save_path
    tag = args.tag
    debug = args.debug
    
    print("=============================================================")
    print(f"Making predictions with tag: {tag}")
    get_embedding(data_path, pt_ckpt_dir_path, save_path, tag, debug)
