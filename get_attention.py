import torch
from torch.utils.data import DataLoader
from dataset import RegressionDataset
from model.modules import TextEncoder_attn
import numpy as np
import pandas as pd
import os, yaml, pickle
from transformers import RobertaTokenizerFast
import tqdm
# Import your RegressionModel class, train_fn, and any other necessary modules

# Define a function for making predictions
# def predict_fn(data_loader, model, device):
#     model.eval()  # Put the model in evaluation mode.
#     predictions = []

#     with torch.no_grad():  # Disable gradient calculation.
#         for batch in tqdm.tqdm(data_loader):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(batch).squeeze(-1)
#             predictions.extend(outputs.cpu().numpy())  # Store predictions as numpy arrays

#     return np.array(predictions)

# def attention_per_part(data_loader, model, tokenizer, device):
#     # model = model.to(device)
#     model.eval()
#     weights = [0, 0, 0, 0] # size is set as the max section number (4)
#     with torch.no_grad():  # Disable gradient calculation.
#         for batch in tqdm.tqdm(data_loader):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             attentions = model(batch).squeeze(-1) # model should be TextEncoder_attn
#             multihead_avg = attentions.mean(axis=1) # Average over heads
#             attns = multihead_avg[:, 0, :]	# Scores from <s> token
#             # breakpoint()
#             ids, attns = batch['input_ids'].squeeze(0), attns.squeeze(0)	# Batch dim
#             tokens = tokenizer.convert_ids_to_tokens(ids)

#             partsAtt = []
#             num_fails = 0
  
#             tokens = tokens[1:]
#             attns = attns[1:]
#         	# count number of </s> in tokens
#             num_end_tokens = tokens.count('</s>')
#             num_of_sections = 3
#             if num_end_tokens != num_of_sections:
#                 print('</s> number not matching with section number')
#                 num_fails += 1
#                 continue
            
#             for i in range(num_of_sections):
#                 index = tokens.index('</s>') + 1
#                 partsAtt.append(attns[:index])
#                 tokens = tokens[index:]
#                 attns = attns[index:]

#             for part in range(len(partsAtt)):
#                 weights[part] += partsAtt[part].sum().item()

#         n = len(data_loader)
#         for i in range(len(weights)):
#             weights[i] /= n

#         print(weights)
#         print('the number of failures: ', num_fails)

#         results = {'weights': weights, 'num_fails': num_fails}

#     return results

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
    # with open(os.path.join(pt_ckpt_dir_path, "clip.yml"), "r") as f:
    #     config = yaml.safe_load(f)
    # Hyperparameters and settings   
    
   
    ############################################################################
    ckpt_name = pt_ckpt_dir_path.split('/')[-1]
    pt_ckpt_path = os.path.join(pt_ckpt_dir_path, "checkpoint.pt")
    model_config_path = os.path.join(pt_ckpt_dir_path, "clip.yml") #config["model_config"]
    # train_config_path = os.path.join(pt_ckpt_dir_path, "regress_train.yml") #config["train_config"]
    ############################################################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug:
        device = "cpu"
    
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
    # breakpoint()
    # Initialize training dataset
    test_dataset = RegressionDataset(texts = df_test["text"].values,
                                      targets = df_test["target"].values,
                                      chg_emb = df_test["chg_emb"].values,
                                      tokenizer = tokenizer,
                                      seq_len= tokenizer.model_max_length)
    # Create training dataloader
    test_data_loader = DataLoader(test_dataset, batch_size =1,
                                  shuffle = False, num_workers=1)

    #breakpoint()    
    # ===================== MODEL and TOKENIZER ===============================
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    model = TextEncoder_attn(model_config).to(device)  

    print('loading pretrained checkpoint from')
    print(ckpt_name)
    ####  check this part!!!!
    # breakpoint()
    model_state_dict = model.state_dict()
    # prefix = 'text_encoder.'
    # state_dict = torch.load(pt_ckpt_path, map_location=device)['model_state_dict']
    # new_state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    # # remove keys starting with chg_embedding from the new_state_dict
    # new_state_dict = {key: value for key, value in new_state_dict.items() if not key.startswith('chg_embedding')}
    prefix = 'text_encoder.'
    state_dict = torch.load(pt_ckpt_path, map_location=device)['model_state_dict']
    new_state_dict = {key[len(prefix):]: value for key, value in state_dict.items() \
                      if key.startswith(prefix) and not key.startswith(prefix + 'chg_embedding')}
    # breakpoint()
    model.load_state_dict(new_state_dict, strict=True) 
    # breakpoint() 
    # ========================= PREDICTION ====================================
    predictions = attention_per_part(test_data_loader, model, tokenizer, device)
    #predict_fn(test_data_loader, model, device)
    
    # ========================= SAVE PREDICTION ===============================
    save_path = os.path.join(save_path, f"attn-{ckpt_name}-{tag}-strc.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(predictions, f)
    # df_test['pred'] = predictions
    # df_test.to_pickle(save_path) 
    
    ## save as dictionary where key is id in df_test, and values are predictions
    # predictions_dict = dict(zip(df_test["id"].values, predictions))
    # with open(save_path, "wb") as f:
    #     pickle.dump(predictions_dict, f)
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["eval","eval_dup","eval_non_dup"], default="eval")
    parser.add_argument("--model", type=str, choices=["gnoc", "escn", "scn", "eqv2"], default="eqv2")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--train_type", type=str, choices=["direct_regress", "clip_regress"], default="clip_regress")

    args = parser.parse_args()
    split = args.split
    model = args.model
    ckpt = args.ckpt
    train_type = args.train_type
    debug = args.debug
    # Conduct prediction
    data_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/clip_data/oc20dense_{split}_{model}_relaxed.pkl"
    # data_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/clip_data/{split}_{model}.pkl"
    # /eval_{model}_dup.pkl
    pt_ckpt_dir_path = f"/home/jovyan/shared-scratch/jhoon/ocp2023/checkpoints/{train_type}/{ckpt}"
    save_path = "/home/jovyan/shared-scratch/jhoon/ocp2023/results/attentions/indv/" 
    tag = f"{split}-{model}"
    print("=============================================================")
    print(f"Making predictions for {split} split of {model} model")
    run_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug)