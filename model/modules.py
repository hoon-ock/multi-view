import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

def normalize_tensor(input_tensor):
    """
    Normalize the input tensor across the last dimension.

    Parameters:
    - input_tensor (torch.Tensor): The tensor to normalize.

    Returns:
    torch.Tensor: The normalized tensor.
    """
    mean_tensor = input_tensor.mean(dim=-1, keepdim=True)
    std_tensor = input_tensor.std(dim=-1, keepdim=True)
    return (input_tensor - mean_tensor) / (std_tensor + 1e-7)

class TextEncoder(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.pretrain_ckpt_path = config['Path']['pretrain_ckpt']
        self.hidden_size = config['RobertaConfig']['hidden_size'] # 768 for roberta
        self.num_attention_heads = config['RobertaConfig']['num_attention_heads']                 
        self.num_hidden_layers = config['RobertaConfig']['num_hidden_layers'] ## Encoder layers
        self.vocab_size = config['RobertaConfig']['vocab_size']
        self.max_position_embeddings = config['RobertaConfig']['max_position_embeddings']
        #self.emb_tagging = config['CHGConfig']['emb_tagging'] if config['CHGConfig']['emb_tagging'] else False
        self.num_chg_dim = config['CHGConfig']['num_chg_dim']       
        self.emb_tagging = config['CHGConfig']["emb_tagging"] #config['CHG_EMB_TAG']
        #self.output_dim = config['RegressorConfig']['output_dim']
        self.chg_embedding = nn.Linear(self.num_chg_dim, self.hidden_size) ## num_chg_dims = 64

        # if self.pretrain_ckpt_path is not None:

        roberta_config = RobertaConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings, 
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers= self.num_hidden_layers,
            hidden_size=self.hidden_size,
            type_vocab_size=1,
        )
        self.transformer = RobertaModel(config=roberta_config)
        
        if self.pretrain_ckpt_path is not None:
            print('Loading pre-trained weights from', self.pretrain_ckpt_path)
            self.load_pretrained_weights()
        
        # else:
        #     self.transformer = RobertaModel.from_pretrained('roberta-base')
        
        self.token_embedding = self.transformer.embeddings 
        #self.regressor = nn.Linear(self.hidden_size, self.output_dims) 


    def load_pretrained_weights(self):
        model_dict = self.transformer.state_dict()
        state_dict = torch.load(self.pretrain_ckpt_path)
        
        matching_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.replace('model.', '') in model_dict}
        assert matching_state_dict.keys() == model_dict.keys(), f"Missing keys: {model_dict.keys() - matching_state_dict.keys()}"
        self.transformer.load_state_dict(matching_state_dict, strict=True)
        
        
    def forward(self, batch): #tokens, chg_embed, mask):
        """
        Forward pass through the model.

        Parameters:
        - batch (Dict[str, torch.Tensor]): A dictionary containing input tensors.

        Returns:
        torch.Tensor: The model's output tensor.
        """
        tokens_embed = self.token_embedding(batch["input_ids"]) # [batch_size, seq_len, hidden_size]
        if self.emb_tagging:
            chg_embed = self.chg_embedding(batch["chg_embed"]) 
            tokens_embed = normalize_tensor(tokens_embed)
            chg_embed = normalize_tensor(chg_embed)
            initial_embeddings =  tokens_embed + chg_embed.unsqueeze(1) #/10
        # initial_embeddings =  tokens_embed # + chg_embed.unsqueeze(1)/10
            # breakpoint()
        else:
            initial_embeddings = tokens_embed
            # breakpoint()
        
        outputs = self.transformer(attention_mask = batch["attention_mask"], 
                                    inputs_embeds = initial_embeddings)
        logits = outputs.last_hidden_state[:, 0, :]
        # pooler = outputs["pooler_output"]
        # output = self.regressor(pooler)
        return logits
    

class ProjectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config['RobertaConfig']['hidden_size']
        self.projection_dim = config['ProjectionConfig']['projection_dim']
        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(config['ProjectionConfig']['dropout_rate'])
        self.layer_norm = nn.LayerNorm(self.projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
class RegressionHead(nn.Module):
    '''
    config: clip.yml used for pre-training 
    '''
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['ProjectionConfig']['projection_dim']
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.input_dim, self.input_dim),
            nn.SiLU(),
            nn.Linear(self.input_dim, 1)  
        )
    def forward(self, x):
        return self.regressor(x)