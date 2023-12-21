import numpy as np
import pandas as pd
import torch, transformers, os
from torch.utils.data import DataLoader
from dataset import RegressionDataset
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import roberta_base_AdamW_grouped_LLRD
import yaml, os, shutil
from model.models import RegressionModel, RegressionModel2
from transformers import RobertaTokenizerFast
from datetime import datetime
import torch.nn as nn



# def train_batch(data,
#                 model,
#                 batch_count,
#                 device,
#                 loss_fn):
#     token, target = data
#     token['input_ids'] = token['input_ids'].to(device)
#     token['attention_mask'] = token['attention_mask'].to(device)
#     token['input_ids'] = token['input_ids'].squeeze(1)
#     token['attention_mask'] = token['attention_mask'].squeeze(1)
#     target = target.to(device)
#     outputs = model(token)
#     outputs = outputs.squeeze(1)
#     loss = loss_fn(outputs, target)
#     return loss, outputs

# def load_pretrained_model(model,
#                           pretrained_model):
#     load_state = torch.load(pretrained_model) 
#     model_state = model.state_dict()
#     print(model_state)
#     # exit()
#     for name, param in load_state.items():
#         # print(name)
#         if name not in model_state:
#             print('NOT loaded:', name)
#             continue
#         else:
#             print('loaded:', name)
#             if isinstance(param, nn.parameter.Parameter):
#                 param = param.data
#         model_state[name].copy_(param)
#         print("Loaded pre-trained model with success.")
#     return model


# def load_clip_checkpoint(model, checkpoint_path, device):

#     # Load the state_dict from the checkpoint file
#     state_dict = torch.load(checkpoint_path, map_location=device)

#     # Print missing keys
#     missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
#     print(f"Missing keys: {missing_keys}")

#     # Load state_dict while ignoring missing keys
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
    
#     return model


def train_fn(data_loader, model, optimizer, device, 
             scheduler, loss_fn, log_interval, debug=False,):
    
    model.train()                               # Put the model in training mode.                   
    lr_list = []
    train_losses = []
    print('training...')

    batch_iteration = 0

    for batch in tqdm(data_loader):                   # Loop over all batches.
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = batch["target"]

        optimizer.zero_grad()                   # To zero out the gradients.
        outputs = model(batch).squeeze(-1) 
        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())    
        loss.backward()                         # To backpropagate the error (gradients are computed).
        optimizer.step()                        # To update parameters based on current gradients.
        lr_list.append(optimizer.param_groups[0]["lr"])
        
        if (batch_iteration != 0) and (batch_iteration % log_interval == 0) and (debug == False):
            wandb.log({"iter_train_loss": loss.item()}) 
                       # step = train_count + batch_iteration)

        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()                        # To update learning rate.    

        batch_iteration += 1

    return np.mean(train_losses), np.mean(lr_list)


def validate_fn(data_loader, model, device, loss_fn):  
    model.eval()                                    # Put model in evaluation mode.
    val_losses = []
    val_maes = []
    print('validating...')

    with torch.no_grad():                           # Disable gradient calculation.
        for batch in tqdm(data_loader):                   # Loop over all batches.   
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch).squeeze(-1) 
            targets = batch["target"]    
            loss = loss_fn(outputs, targets)   
            mae = torch.mean(torch.abs(targets - outputs))             
            val_losses.append(loss.item())
            val_maes.append(mae.item())

    return np.mean(val_losses), np.mean(val_maes)


def run_regression(config_file):  
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    # Hyperparameters and settings   
    RUN_NAME = config["run_name"]+datetime.now().strftime("_%m%d_%H%M")
    TRAIN_PATH = config["train_path"] 
    VAL_PATH = config["val_path"]
    CKPT_SAVE_DIR = os.path.join(config["ckpt_save_path"], RUN_NAME)

    ############################################################################
    RESUME_PATH = config["resume_path"] if config.get("resume_path") else None
    RESUME_CONFIG = config["resume_config"] if config.get("resume_config") else None

    PT_CKPT_PATH = config["pt_ckpt_path"] if config.get("pt_ckpt_path") else None
    # CATBERTA_CKPT_PATH = config["catberta_ckpt_path"] if config.get("catberta_ckpt_path") else None
    MODEL_CONFIG = config["model_config"]
    ############################################################################
    HEAD = config["head"] if config.get("head") else "regress"
    DEVICE = config["device"]
    EPOCHS = config["num_epochs"]
    EARLY_STOP_THRESHOLD = config["early_stop_threshold"]  # Set the early stopping threshold    
    TRAIN_BS = config["batch_size"]  # Training batch size
    VAL_BS = TRAIN_BS            # Validation batch size
    LR = config["lr"] if config.get("lr") else 1e-6 # Learning rate
    WRMUP = config["warmup_steps"] if config.get("warmup_steps") else 0 # warmup step for scheduler
    OPTIM = config["optimizer"] if config.get("optimizer") else "AdamW" # optimizer type
    SCHD = config["scheduler"] if config.get("scheduler") else "reduceLR" # scheduler type
    LOSS_FN = config["loss_fn"] if config.get("loss_fn") else "MSELoss" # loss function
    LOG_INTERVAL = config["log_interval"] if config.get("log_interval") else 10 # log interval
    DEBUG = config["debug"] if config.get("debug") else False
    if DEBUG:
        DEVICE = "cpu"
    if RESUME_PATH and RESUME_CONFIG:
        MODEL_CONFIG = RESUME_CONFIG


    print("=============================================================")
    print(f"{RUN_NAME} is launched")
    print("=============================================================")
    print(f"Head: {HEAD}")
    print(f"Epochs: {EPOCHS}")
    print(f"Early stopping threshold: {EARLY_STOP_THRESHOLD}")
    print(f"Training batch size: {TRAIN_BS}")
    print(f"Validation batch size: {VAL_BS}")
    print(f"Initial learning rate: {LR}")
    print(f"Warmup steps: {WRMUP}")
    print(f"Optimizer: {OPTIM}")
    print(f"Scheduler: {SCHD}")
    print(f"Loss function: {LOSS_FN}")
    print("=============================================================")
    if not DEBUG:
        wandb.init(project="clip-regress", name=RUN_NAME) 
                   #,dir='/home/jovyan/shared-scratch/jhoon/ocp2023/log')
        
    # ========================= COPY CONFIG FILE ==============================
    if not os.path.exists(CKPT_SAVE_DIR):
        os.makedirs(CKPT_SAVE_DIR)
    
    if not DEBUG:
        shutil.copyfile(config_file, os.path.join(CKPT_SAVE_DIR, config_file.split('/')[-1]))
        shutil.copyfile(MODEL_CONFIG, os.path.join(CKPT_SAVE_DIR, MODEL_CONFIG.split('/')[-1]))
    
    # ========================= DATA LOADING =================================
    # Load train and validation data 
    df_train = pd.read_pickle(TRAIN_PATH)
    df_val = pd.read_pickle(VAL_PATH)
    if DEBUG:
        df_train = df_train.sample(10)
        df_val = df_val.sample(2)

    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    # Initialize training dataset
    train_dataset = RegressionDataset(texts = df_train["text"].values,
                                      targets = df_train["target"].values,
                                      chg_emb = df_train["chg_emb"].values,
                                      tokenizer = tokenizer,
                                      seq_len= tokenizer.model_max_length)
    # Initialize validation dataset
    val_dataset = RegressionDataset(texts = df_val["text"].values,
                                    targets = df_val["target"].values,
                                    chg_emb = df_val["chg_emb"].values,
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

    if HEAD == "pooler":
        model = RegressionModel2(model_config).to(DEVICE)
    else:
        model = RegressionModel(model_config).to(DEVICE)
    
    if PT_CKPT_PATH:
        print('loading pretrained text encoder and projection layer from')
        print(PT_CKPT_PATH)
        state_dict = torch.load(PT_CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)  
   
    elif RESUME_PATH:
        print('resume training from ', RESUME_PATH)
        state_dict = torch.load(RESUME_PATH, map_location=DEVICE)['model_state_dict']
        # breakpoint()
        model.load_state_dict(state_dict, strict=False)

    # elif CATBERTA_CKPT_PATH:
    #     print('loading pretrained catberta from')
    #     print(CATBERTA_CKPT_PATH)
    #     breakpoint()
    #     state_dict = torch.load(CATBERTA_CKPT_PATH, map_location=DEVICE)


        
    # if "debug" not in RUN_NAME:
    #     wandb.watch(model, log="parameters")
    # ====================== OPTIMIZER AND SCHEDULER =========================
    if config.get("optimizer") == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #originally 1e-6
    elif config.get("optimizer") == "gLLRD":
        optimizer, _ = roberta_base_AdamW_grouped_LLRD(model, LR)

    if SCHD == "reduceLR":
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4)
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
                        num_training_steps = train_steps) #1000
    # ========================= LOSS FUNCTION ================================
    if LOSS_FN == "MSELoss":
        loss_fn = torch.nn.MSELoss()
    elif LOSS_FN == "L1Loss":
        loss_fn = torch.nn.L1Loss()

    #=========================================================================
    # Training Loop - Start training the epochs
    #=========================================================================   
    best_loss = 999
    early_stopping_counter = 0       
    for epoch in range(1, EPOCHS+1):
        # Call the train function and get the training loss
        train_loss, lr = train_fn(train_data_loader, 
                                  model, 
                                  optimizer, 
                                  DEVICE, 
                                  scheduler,
                                  loss_fn, 
                                  LOG_INTERVAL,
                                  debug=DEBUG)
        
        # Perform validation and get the validation loss
        val_loss, val_mae = validate_fn(val_data_loader, 
                                        model, 
                                        DEVICE, 
                                        loss_fn)
        if SCHD == 'reduceLR':
            scheduler.step(val_loss)
        
        loss = val_loss
        if not DEBUG:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, 'val_mae': val_mae, 'lr': lr})
        # If there's improvement on the validation loss, save the model checkpoint.
        # Else do early stopping if threshold is reached.
        if loss < best_loss:            
            # torch.save(model.state_dict(), os.path.join(CKPT_SAVE_DIR, 'checkpoint.pt'))

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_loss': best_loss},
                         os.path.join(CKPT_SAVE_DIR, 'checkpoint.pt'))

            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}, \
Val MAE = {round(val_mae,3)}, checkpoint saved.")
            best_loss = loss
            early_stopping_counter = 0
        else:
            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}, \
Val MAE = {round(val_mae,3)},")
            early_stopping_counter += 1
        if early_stopping_counter > EARLY_STOP_THRESHOLD:
            print(f"Early stopping triggered at epoch {epoch}! Best Loss: {round(best_loss,3)}\n")                
            break

    print(f"===== Training Termination =====")
    if not DEBUG:    
        wandb.finish()


if __name__ == "__main__":
  
    # Run the training loop
    run_regression("regress_train.yml") 