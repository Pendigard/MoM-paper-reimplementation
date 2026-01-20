import torch
import torch.nn as nn
import time
from typing import Dict, Tuple, Optional, Callable


def linear_attn(M : torch.Tensor, M_k : torch.Tensor, M_v : torch.Tensor, indices_update : torch.Tensor) -> torch.Tensor:
    """
    @brief Met à jour la mémoire M avec la clé k et la valeur v en utilisant une attention linéaire.
    @param M: Mémoires actuelles de forme (batch_size, num_memories, hidden_dim, hidden_dim)
    @param M_k: Clés des mémoires de forme (batch_size, num_memories, hidden_dim)
    @param M_v: Valeur des mémoires forme (batch_size, num_memories,  hidden_dim)
    @param indices_update: Indices des mémoires à mettre à jour de forme (batch_size, k + 1)
    @return: Mémoire mise à jour de forme (batch_size, num_memories,  hidden_dim, hidden_dim)
    """
    hidden_dim = M.shape[2]

    M_kv = M_k.unsqueeze(-1) @ M_v.unsqueeze(-2)
    
    # (b, k, d, d)
    indices_update_exp = indices_update.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hidden_dim, hidden_dim)
    
    M_kv_to_add = torch.gather(M_kv, 1, indices_update_exp)

    M_new = M.clone()
    return M_new.scatter_add_(dim=1, index=indices_update_exp, src=M_kv_to_add)

class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, M : torch.Tensor, M_k : torch.Tensor, M_v : torch.Tensor, indices_update : torch.Tensor) -> torch.Tensor:
        return linear_attn(M, M_k, M_v, indices_update)



class MoM(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, num_memories : int, k : int, update_module: nn.Module = None, mode = "linear", *args, **kwargs):
        """
        @brief Module de mixture de mémoires (Mixture of Memories). Il s'agit d'une implémentation naïve, utilisé au début du projet.
        @param input_dim: Dimension de l'entrée x.
        @param hidden_dim: Dimension de chaque mémoire M_t.
        @param num_memories: Nombre de mémoires (Ça ne prend pas en compte la mémoire partagée).
        @param k: Hyperparamètre k pour la sélection des top-k mémoires.
        @param update_module: Module de mise à jour des mémoires.
        """
        super().__init__(*args, **kwargs)

        self.num_memories = num_memories
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.k = k
        self.mode = mode

        self.W_k = nn.Linear(input_dim, hidden_dim * (num_memories + 1))
        self.W_v = nn.Linear(input_dim, hidden_dim * (num_memories + 1)) # On inclut la mémoire partagée
        self.W_g = nn.Linear(input_dim, num_memories) # On ne calcule pas de score pour la mémoire partagée
        self.W_q = nn.Linear(input_dim, hidden_dim)

        if self.mode == "gla":
            self.W_gate = nn.Linear(input_dim, hidden_dim * (num_memories + 1))
        elif self.mode == "deltanet":
            # comme on a a et b on doit doubler la taille
            self.W_gate = nn.Linear(input_dim, hidden_dim * 2 * (num_memories + 1))

        self.update_module = update_module or LinearAttention()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X : torch.Tensor, M_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @brief passe-avant du module MoM en version naïve.
        @param M_0: État initiale des mémoires de forme (hidden_dim, hidden_dim).
        @param X: Entrée de forme (seq_len, batch_size, input_dim)
        @return: Les outputs et l'état final des mémoires. (seq_len, batch_size, hidden_dim), (batch_size, num_memories + 1, hidden_dim, hidden_dim)
        """
        batch_size = X.shape[1]
        M_t = M_0.expand(batch_size, self.num_memories + 1, self.hidden_dim, self.hidden_dim)
        outputs = []
        for x_t in X:
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            score_t = torch.softmax(self.W_g(x_t), dim=-1)

            m_scores, m_indices = torch.topk(score_t, self.k)
            m_indices = m_indices + 1 # On décale de 1 car la sélection ne se fait pas sur la mémoire partagée
            m_indices_update = torch.cat([torch.zeros(batch_size, 1, device=M_t.device, dtype=torch.long), m_indices], dim=1) # On ajoute la mémoire partagée (index 0) aux indices des mémoires à mettre à jour
            m_indices_update = m_indices_update.to(device=M_t.device, dtype=torch.long)  
            
            g_t = m_scores / m_scores.sum(dim=1, keepdim=True) # On normalise les scores

            M_k = self.W_k(x_t).reshape(batch_size, self.num_memories + 1, self.hidden_dim)
            M_v = self.W_v(x_t).reshape(batch_size, self.num_memories + 1, self.hidden_dim)

            if self.mode == "linear":
                M_t = self.update_module(M_t, M_k, M_v, m_indices_update)
            else :
                active_mask = torch.zeros(batch_size, self.num_memories + 1, device=X.device)
                active_mask.scatter_(1, m_indices_update, 1)
                mask = active_mask.view(batch_size, self.num_memories + 1, 1, 1)

                if self.mode == "gla":
                    gate_logits = self.W_gate(x_t).reshape(batch_size, self.num_memories + 1, self.hidden_dim)
                    alpha = torch.sigmoid(gate_logits).unsqueeze(-1) 
                    M_decayed = M_t * alpha
                    kv = M_k.unsqueeze(-1) @ M_v.unsqueeze(-2)
                    M_new = M_decayed + kv
                    M_t = M_new * mask + (1-mask) * M_t
                elif self.mode == "deltanet":
                    gates = self.W_gate(x_t).reshape(batch_size, self.num_memories + 1, self.hidden_dim * 2)
                    
                    alpha1, beta1 = torch.sigmoid(gates).chunk(2, dim=-1)
                    alpha = torch.sigmoid(alpha1).unsqueeze(2)
                    beta = torch.sigmoid(beta1).unsqueeze(2)
                    
                    recall = torch.matmul(M_k.unsqueeze(2), M_t)
                    V = beta * M_v.unsqueeze(2)
                    recall_weighted = recall * alpha
                    
                    bracket = V - recall_weighted
                    update = torch.matmul(M_k.unsqueeze(3), bracket)
                    M_new = (alpha * M_t) + update
                    M_t = mask * M_new + (1 - mask) * M_t



            # On récupère les états des mémoires sélectionnées
            M_to_use = M_t.gather(dim=1, index=m_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim, self.hidden_dim))
            # On pondère les mémoires par leurs scores g calculés précédemment
            M_weighted = M_to_use * g_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim, self.hidden_dim)

            M_out = M_weighted.sum(dim=1) + M_t[:,0,:,:]
            
            q_t = self.W_q(x_t)
            o_t = q_t.unsqueeze(-2) @ M_out
            outputs.append(o_t.squeeze(1))

        return torch.stack(outputs)
    