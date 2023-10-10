import torch
import torch.nn as nn
import torch.nn.functional as F

def roberta_base_AdamW_grouped_LLRD(model, init_lr, debug=False):
        
    opt_parameters = [] # To be passed to the optimizer (only parameters of the layers you want to update).
    debug_param_groups = []
    named_parameters = list(model.named_parameters()) 
    
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
    # breakpoint()
    for i, (name, params) in enumerate(named_parameters):  
        
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01
 
        if name.startswith("model.embeddings") or name.startswith("model.encoder"):            
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = init_lr       
            
            # For set_2, increase lr to 0.00000175
            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr
            
            # For set_3, increase lr to 0.0000035 
            lr = init_lr * 3.5 if any(p in name for p in set_3) else lr
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})  
        #breakpoint()    
        # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).                
        if name.startswith("regressor") or name.startswith("model.pooler"):               
            lr = init_lr * 3.6 
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})    
            
        debug_param_groups.append(f"{i} {name}")

    if debug: 
        for g in range(len(debug_param_groups)): print(debug_param_groups[g]) 

    return torch.optim.AdamW(opt_parameters, lr=init_lr, weight_decay=0.01), debug_param_groups




# def load_model(model, checkpoint_path, device):
#     # Load the state_dict from the checkpoint file
#     checkpoint = torch.load(checkpoint_path, map_location=device)  # Use 'cuda:0' if using GPU
    
#     # Update the model's state_dict
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     return model