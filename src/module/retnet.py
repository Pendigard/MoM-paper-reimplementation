import torch
import torch.nn as nn
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder


class RetNetModule(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.config = RetNetConfig(
            vocab_size=config_dict["vocab_size"],
            decoder_embed_dim=config_dict["dim"],
            decoder_retention_heads=2, 
            decoder_ffn_embed_dim=config_dict["dim"] * 4,
            decoder_layers=config_dict["num_layers"],
            recurrent_chunk_size=None,
            dropout=0.0,             
            activation_fn="gelu",    
            drop_path_rate=0.0
        )

        self.embed_tokens = nn.Embedding(config_dict["vocab_size"], config_dict["dim"], padding_idx=0)

        self.model = RetNetDecoder(self.config, self.embed_tokens)

    def forward(self, input_ids):
        logits, _ = self.model(input_ids)
        return logits