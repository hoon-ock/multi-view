import numpy as np
import pandas as pd
import torch, transformers, os
from torch.utils.data import DataLoader
from dataset import ClipDataset
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import roberta_base_AdamW_grouped_LLRD
import yaml, os, shutil
from model.models import CLIPModel
from transformers import RobertaTokenizerFast
from datetime import datetime


def update_projection_dim(train_config_file):
    """
    Update 'projection_dim' in the model config YAML file based on 'GNN_EMB'.

    Parameters:
        config (dict): The primary configuration dictionary with keys 
                      'gnn_emb' and 'model_config'.
    """
    # Load config file
    with open(train_config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract relevant information from the primary config.
    MODEL_CONFIG = config["model_config"]
    GNN_EMB = config['gnn_emb']
    
    # Load the MODEL_CONFIG yaml file.
    with open(MODEL_CONFIG, 'r') as file:
        try:
            model_config_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return
    
    # Update 'projection_dim' based on the value of GNN_EMB.
    if GNN_EMB == 'gnoc_emb':
        model_config_file['ProjectionConfig']['projection_dim'] = 256
    elif GNN_EMB == 'eq_emb':
        model_config_file['ProjectionConfig']['projection_dim'] = 3200
    else:
        print(f"Unknown GNN_EMB value: {GNN_EMB}. Not updating 'projection_dim'.")
        return
    
    # Save the updated MODEL_CONFIG back to file.
    with open(MODEL_CONFIG, 'w') as file:
        try:
            yaml.dump(model_config_file, file)
        except yaml.YAMLError as exc:
            print(exc)    

def train_fn(data_loader, model, optimizer, device, scheduler, run_name):
    model.train()                               # Put the model in training mode.                   
    lr_list = []
    train_losses = []
    print('training...')

    batch_iteration = 0

    for batch in tqdm(data_loader):                   # Loop over all batches.
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()                   # To zero out the gradients.
        loss = model(batch).squeeze(-1) 

        train_losses.append(loss.item())    
        loss.backward()                         # To backpropagate the error (gradients are computed).
        optimizer.step()                        # To update parameters based on current gradients.
        lr_list.append(optimizer.param_groups[0]["lr"])
        
        # if batch_iteration % 100 == 0:
        #     # Log the loss and learning rate with wandb
        #     if "debug" not in run_name:
        #         wandb.log({"train_loss": loss.item(), 
        #                    "lr": optimizer.param_groups[0]["lr"]}, 
        #                    step = batch_iteration)

        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()                        # To update learning rate.    

        batch_iteration += 1

    return np.mean(train_losses), np.mean(lr_list)



def validate_fn(data_loader, model, device, run_name):  
    model.eval()                                    # Put model in evaluation mode.
    val_losses = []

    print('validating...')

    batch_iteration = 0
    with torch.no_grad():                           # Disable gradient calculation.
        for batch in tqdm(data_loader):                   # Loop over all batches.   
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch).squeeze(-1)                     
            val_losses.append(loss.item())

            # if batch_iteration % 200 == 0:
            # # Log the loss and learning rate with wandb
            #     if "debug" not in run_name:
            #         wandb.log({"val_loss": loss.item()}, step=batch_iteration)
            # batch_iteration += 1
    return np.mean(val_losses)




def run_clip(config_file):  
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Hyperparameters and settings   
    RUN_NAME = config["run_name"]+datetime.now().strftime("_%m%d_%H%M")
    TRAIN_PATH = config["train_path"] 
    VAL_PATH = config["val_path"]
    CKPT_SAVE_DIR = os.path.join(config["ckpt_save_path"], RUN_NAME)
    PT_CKPT_PATH = config["pt_ckpt_path"] if config.get("pt_ckpt_path") else None
    MODEL_CONFIG = config["model_config"]
    DEVICE = config["device"]
    EPOCHS = config["num_epochs"]
    EARLY_STOP_THRESHOLD = config["early_stop_threshold"]  # Set the early stopping threshold    
    TRAIN_BS = config["batch_size"]  # Training batch size
    VAL_BS = TRAIN_BS            # Validation batch size
    LR = config["lr"] if config.get("lr") else 1e-6 # Learning rate
    WRMUP = config["warmup_steps"] if config.get("warmup_steps") else 0 # warmup step for scheduler
    OPTIM = config["optimizer"] if config.get("optimizer") else "AdamW" # optimizer type
    SCHD = config["scheduler"] if config.get("scheduler") else "reduceLR" # scheduler type
    GNN_EMB = config['gnn_emb']
    if "debug" in RUN_NAME:
        DEVICE = "cpu"


    print("=============================================================")
    print(f"{RUN_NAME} is launched")
    print("=============================================================")
    print(f"Epochs: {EPOCHS}")
    print(f"Early stopping threshold: {EARLY_STOP_THRESHOLD}")
    print(f"Training batch size: {TRAIN_BS}")
    print(f"Validation batch size: {VAL_BS}")
    print(f"Initial learning rate: {LR}")
    print(f"Warmup steps: {WRMUP}")
    print(f"Optimizer: {OPTIM}")
    print(f"Scheduler: {SCHD}")
    print("=============================================================")
    if "debug" not in RUN_NAME:
        wandb.init(project="clip", name=RUN_NAME) 
                   #,dir='/home/jovyan/shared-scratch/jhoon/ocp2023/log')
        
    # ========================= COPY CONFIG FILE ==============================
    if "debug" not in RUN_NAME:
        if not os.path.exists(CKPT_SAVE_DIR):
            os.makedirs(CKPT_SAVE_DIR)
        shutil.copyfile(config_file, os.path.join(CKPT_SAVE_DIR, config_file.split("/")[-1]))
        shutil.copyfile(MODEL_CONFIG, os.path.join(CKPT_SAVE_DIR, MODEL_CONFIG.split("/")[-1]))
    
    # ========================= DATA LOADING =================================
    # Load train and validation data 
    df_train = pd.read_pickle(TRAIN_PATH)
    df_val = pd.read_pickle(VAL_PATH)

    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    # Initialize training dataset
    train_dataset = ClipDataset(texts = df_train["text"].values,
                              targets = df_train["target"].values,
                              chg_emb = df_train["chg_emb"].values,
                              graph_emb = df_train[GNN_EMB].values,
                              tokenizer = tokenizer,
                              seq_len= tokenizer.model_max_length)
    # Initialize validation dataset
    val_dataset = ClipDataset(texts = df_val["text"].values,
                            targets = df_val["target"].values,
                            chg_emb = df_val["chg_emb"].values,
                            graph_emb = df_val[GNN_EMB].values,
                            tokenizer = tokenizer,
                            seq_len= tokenizer.model_max_length)
    # Create training dataloader
    train_data_loader = DataLoader(train_dataset, batch_size = TRAIN_BS,
                                   shuffle = True, num_workers = 2)
    # Create validation dataloader
    val_data_loader = DataLoader(val_dataset, batch_size = VAL_BS,
                                 shuffle = False, num_workers = 2)  
    
    # ===================== MODEL and TOKENIZER ===============================
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f)
    model = CLIPModel(model_config).to(DEVICE)
    if PT_CKPT_PATH:
        print('loading pretrained model from')
        print(PT_CKPT_PATH)
        state_dict = torch.load(PT_CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)        
        
    # if "debug" not in RUN_NAME:
    #     wandb.watch(model, log="parameters")
    # ====================== OPTIMIZER AND SCHEDULER =========================
    if config.get("optimizer") == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #originally 1e-6
    elif config.get("optimizer") == "gLLRD":
        optimizer, _ = roberta_base_AdamW_grouped_LLRD(model, LR)

    if SCHD == "reduceLR":
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    
    else:
        # Calculate the number of training steps (this is used by scheduler).
        # training steps = [number of batches] x [number of epochs].
        train_steps = int(len(df_train) / TRAIN_BS * EPOCHS)    
        # Get the learning rate scheduler    
        scheduler = transformers.get_scheduler(
                        SCHD,    # Create a schedule with a learning rate that decreases linearly 
                                 # from the initial learning rate set in the optimizer to 0.
                        optimizer = optimizer,
                        num_warmup_steps = WRMUP, #50
                        num_training_steps = train_steps)
    
    #=========================================================================
    # Training Loop - Start training the epochs
    #=========================================================================   
    best_loss = 999
    early_stopping_counter = 0       
    for epoch in range(1, EPOCHS+1):
        # Call the train function and get the training loss
        train_loss, lr = train_fn(train_data_loader, model, optimizer, DEVICE, scheduler, RUN_NAME)
        
        # Perform validation and get the validation loss
        val_loss = validate_fn(val_data_loader, model, DEVICE, RUN_NAME)
        if SCHD == 'reduceLR':
            scheduler.step(val_loss)
        loss = val_loss
        if "debug" not in RUN_NAME:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, 'lr': lr})
        # If there's improvement on the validation loss, save the model checkpoint.
        # Else do early stopping if threshold is reached.
        if loss < best_loss:            
            torch.save(model.state_dict(), os.path.join(CKPT_SAVE_DIR,f'checkpoint.pt'))
            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}, checkpoint saved.")
            best_loss = loss
            early_stopping_counter = 0
        else:
            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}")
            early_stopping_counter += 1
        if early_stopping_counter > EARLY_STOP_THRESHOLD:
            print(f"Early stopping triggered at epoch {epoch}! Best Loss: {round(best_loss,3)}\n")                
            break

    print(f"===== Training Termination =====")
    if "debug" not in RUN_NAME:        
        wandb.finish()


if __name__ == "__main__":

    # Load the config file
    # with open("clip_train.yml", "r") as f:
    #     config = yaml.safe_load(f)

    # Update 'projection_dim' in the model config YAML file based on 'GNN_EMB'.
    update_projection_dim("clip_train.yml")
    
    # Run the training loop
    run_clip("clip_train.yml") 