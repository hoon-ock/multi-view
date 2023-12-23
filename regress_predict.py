import torch
from torch.utils.data import DataLoader
from dataset import RegressionDataset
from model.models import RegressionModel, RegressionModel2
import numpy as np
import pandas as pd
import os, yaml, pickle
from transformers import RobertaTokenizerFast
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

    return np.array(predictions)



def run_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug=False):  
    # with open(os.path.join(pt_ckpt_dir_path, "clip.yml"), "r") as f:
    #     config = yaml.safe_load(f)
    # Hyperparameters and settings   
    
   
    ############################################################################
    ckpt_name = pt_ckpt_dir_path.split('/')[-1]
    pt_ckpt_path = os.path.join(pt_ckpt_dir_path, "checkpoint.pt")
    model_config_path = os.path.join(pt_ckpt_dir_path, "clip.yml") #config["model_config"]
    train_config_path = os.path.join(pt_ckpt_dir_path, "regress_train.yml") #config["train_config"]
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
    # for prediction we don't need to do charge embedding broadcasting
    if model_config['CHGConfig']['emb_tagging']:
        print("emb_tagging is set to False for prediction")
        model_config['CHGConfig']['emb_tagging'] = False
    
    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)
    head = train_config["head"]
    # breakpoint()
    if head == "pooler":
        model = RegressionModel2(model_config).to(device)
    else:
        model = RegressionModel(model_config).to(device)
    

    print('loading pretrained checkpoint from')
    print(ckpt_name)
    state_dict = torch.load(pt_ckpt_path, map_location=device)['model_state_dict']
    model.load_state_dict(state_dict, strict=True) 
    # breakpoint() 
    # ========================= PREDICTION ====================================
    predictions = predict_fn(test_data_loader, model, device)
    
    # ========================= SAVE PREDICTION ===============================
    save_path = os.path.join(save_path, f"ssl-reg-{ckpt_name}-{tag}-strc.pkl")
    
    # df_test['pred'] = predictions
    # df_test.to_pickle(save_path) 
    
    ## save as dictionary where key is id in df_test, and values are predictions
    predictions_dict = dict(zip(df_test["id"].values, predictions))
    with open(save_path, "wb") as f:
        pickle.dump(predictions_dict, f)
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["eval","test"], default="test")
    parser.add_argument("--model", type=str, choices=["gnoc", "escn", "scn", "eqv2"], default="eqv2")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--train_type", type=str, choices=["direct_regress", "clip_regress"], default="clip_regress")
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
    data_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/clip_data/oc20dense_{split}_{model}_relaxed.pkl"
    pt_ckpt_dir_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/checkpoints/{train_type}/{ckpt}"
    save_path = "/home/jovyan/shared-scratch/jhoon/ocp2023/results/predictions" 
    tag = f"{split}-{model}"
    print("=============================================================")
    print(f"Making predictions for {split} split of {model} model")
    run_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug)