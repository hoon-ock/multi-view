import torch
from torch.utils.data import DataLoader
from dataset import RegressionDataset
from model.models import RegressionModel, RegressionModel2
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



def run_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug=False):  
    # Hyperparameters and settings      
    ############################################################################
    ckpt_name = pt_ckpt_dir_path.split('/')[-1]
    pt_ckpt_path = os.path.join(pt_ckpt_dir_path, "checkpoint.pt")
    model_config_path = os.path.join(pt_ckpt_dir_path, "clip.yml") 
    train_config_path = os.path.join(pt_ckpt_dir_path, "regress_train.yml") 
    ############################################################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug:
        device = "cpu"
    batch = 32
    print("=============================================================")
    print(f"Prediction made with {ckpt_name}")
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
    
    # ===================== MODEL and TOKENIZER ===============================
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)    
        
    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)
    head = train_config["head"]
    
    if head == "pooler":
        model = RegressionModel2(model_config).to(device)
    else:
        model = RegressionModel(model_config).to(device)
    

    print('loading pretrained checkpoint from')
    print(ckpt_name)
    state_dict = torch.load(pt_ckpt_path, map_location=device)['model_state_dict']
    model.load_state_dict(state_dict, strict=True) 
    # ========================= PREDICTION ====================================
    predictions = predict_fn(test_data_loader, model, device)
    
    # ========================= SAVE PREDICTION ===============================
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f"{ckpt_name}-{tag}-strc.pkl")
    
    ## save as dictionary where key is id in df_test, and values are predictions
    predictions_dict = dict(zip(df_test["id"].values, predictions))
    with open(save_path, "wb") as f:
        pickle.dump(predictions_dict, f)
 

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(description="Script to get predictions.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--pt_ckpt_dir_path", type=str, required=True, help="Path to the pretrained checkpoint directory.")
    parser.add_argument("--save_path", type=str, required=True, help="Path where to save the predictions.")
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
    print(f"Making predictions for {data_path}") #{split} split of {model} model")
    run_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug)