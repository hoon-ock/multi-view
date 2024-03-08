import torch
from torch.utils.data import DataLoader
from dataset import RegressionDataset
from model.modules import TextEncoder_attn
import numpy as np
import pandas as pd
import os, yaml, pickle
from transformers import RobertaTokenizerFast
import tqdm

def attention_per_part(data_loader, model, tokenizer, device):
    model.eval()
    
    # Assume there are 12 heads in the model (adjust based on your model's architecture)
    num_heads = 12
    weights = torch.zeros((4, num_heads)) # For 4 sections and num_heads heads

    num_fails = 0
    with torch.no_grad():  # Disable gradient calculation.
        for batch in tqdm.tqdm(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            attentions = model(batch).squeeze(-1) # Shape: [Batch, Heads, Seq_len, Seq_len]
            
            for i in range(attentions.size(0)):  # Iterate over batch
                ids = batch['input_ids'][i]
                tokens = tokenizer.convert_ids_to_tokens(ids)

                # Process each head separately
                for head in range(num_heads):
                    head_attn = attentions[i, head, 0, :]  # Scores from <s> token for each head

                    partsAtt = []
                    tokens_copy = tokens[1:]
                    head_attn = head_attn[1:]
                    num_end_tokens = tokens_copy.count('</s>')
                    num_of_sections = 3
                    if num_end_tokens != num_of_sections:
                        print('</s> number not matching with section number')
                        num_fails += 1
                        continue
                    
                    for _ in range(num_of_sections):
                        index = tokens_copy.index('</s>') + 1
                        partsAtt.append(head_attn[:index])
                        tokens_copy = tokens_copy[index:]
                        head_attn = head_attn[index:]

                    for part in range(len(partsAtt)):
                        weights[part, head] += partsAtt[part].sum().item()

        n = len(data_loader.dataset)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weights[i, j] /= n

        print(weights)
        print('the number of failures: ', num_fails)

        results = {'weights': weights, 'num_fails': num_fails}

    return results

def run_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug=False):      
   
    ############################################################################
    if 'roberta-base' not in pt_ckpt_dir_path:
        ckpt_name = pt_ckpt_dir_path.split('/')[-1]
        pt_ckpt_path = os.path.join(pt_ckpt_dir_path, "checkpoint.pt")
        model_config_path = os.path.join(pt_ckpt_dir_path, "clip.yml") 
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

    elif 'roberta-base' in pt_ckpt_dir_path:
        ckpt_name = 'roberta-base'
        with open("model/clip.yml", "r") as f:
            model_config = yaml.safe_load(f)
        model_config['Path']['pretrain_ckpt'] = "roberta-base" 
        print('loading encoder from roberta-base')
    ############################################################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug:
        device = "cpu"
    # breakpoint()
    print("=============================================================")
    print(f"Attention from {ckpt_name}")
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
                                      chg_emb = df_test["chg_emb"].values,
                                      tokenizer = tokenizer,
                                      seq_len= tokenizer.model_max_length)
    # Create training dataloader
    test_data_loader = DataLoader(test_dataset, batch_size =1,
                                  shuffle = False, num_workers=1)

    
    # ========================== MODEL ==========================   
    model = TextEncoder_attn(model_config).to(device)  

    print('loading pretrained checkpoint from')
    print(ckpt_name)
    if ckpt_name != 'roberta-base':
        prefix = 'text_encoder.'
        if 'ssl' not in ckpt_name:
            state_dict = torch.load(pt_ckpt_path, map_location=device)['model_state_dict']
        elif 'ssl' in ckpt_name:
            state_dict = torch.load(pt_ckpt_path, map_location=device)
        new_state_dict = {key[len(prefix):]: value for key, value in state_dict.items() \
                          if key.startswith(prefix) and not key.startswith(prefix + 'chg_embedding')}
        model.load_state_dict(new_state_dict, strict=True) 

    # ========================= PREDICTION ====================================
    predictions = attention_per_part(test_data_loader, model, tokenizer, device) # section-wise attention score
    
    # ========================= SAVE PREDICTION ===============================
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f"attn-{ckpt_name}-{tag}-strc.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(predictions, f)
    

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(description="Script to run predictions.")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--pt_ckpt_dir_path", type=str, required=True, help="Path to the pretrained checkpoint directory.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the predictions.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--tag", type=str, default=datetime.now().strftime("%y%m%d_%H%M%S"), help="Tag for the run. Defaults to current date and time if not provided.")

    args = parser.parse_args()
    
    # Use the directly provided paths
    data_path = args.data_path
    pt_ckpt_dir_path = args.pt_ckpt_dir_path
    save_path = args.save_path
    debug = args.debug
    tag = args.tag
    #tag = "custom-tag-based-on-your-logic" # Modify this based on your needs
    
    print("=============================================================")
    print(f"Making predictions with provided paths")
    run_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug)
