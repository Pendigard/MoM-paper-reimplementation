import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

class TransformerPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        

        num_heads = 8 
        
        self.llama_config = LlamaConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["dim"],          
            intermediate_size=config["dim"] * 4, 
            num_hidden_layers=config["num_layers"],
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,       
            hidden_act="silu",                   
            max_position_embeddings=config["seq_len"],
            rms_norm_eps=1e-5,
            use_cache=False                     
        )
        
        self.model = LlamaForCausalLM(self.llama_config)

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        
        return outputs.logits, 0.0