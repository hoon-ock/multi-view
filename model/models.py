import torch
from torch import nn
import torch.nn.functional as F

from .modules import TextEncoder, ProjectionHead, RegressionHead


class CLIPModel(nn.Module):
    def __init__(
        self,
        config
        # temperature=CFG.temperature,
        # image_embedding=CFG.image_embedding,
        # text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        # self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(config)
        # self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(config)
        self.temperature = config['CLIPConfig']['temperature']

    def forward(self, batch):
        # Getting Graph and Text Features
        # image_features = self.image_encoder(batch["image"])
        # breakpoint()
        text_features = self.text_encoder(batch)
        # Getting Image and Text Embeddings (with same dimension)
        # image_embeddings = self.image_projection(image_features)
        
        text_embeddings = self.text_projection(text_features)
        graph_embeddings = batch['graph_embed']

        # Calculating the Loss
        logits = (text_embeddings @ graph_embeddings.T) / self.temperature
        graphs_similarity = graph_embeddings @ graph_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (graphs_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    

class RegressionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.text_encoder = TextEncoder(config)
        self.text_projection = ProjectionHead(config)
        self.regressor = RegressionHead(config)
        self._initialize_weights(self.regressor)

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, batch):
        output = self.text_encoder(batch)
        output = self.text_projection(output)
        output = self.regressor(output)
        return output