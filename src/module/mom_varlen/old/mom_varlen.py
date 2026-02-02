import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
import time
import triton
import triton.language as tl
from src.module.update_module_varlen import GLAVarlenModule, LinearAttentionVarlenModule
from src.utils.benchmark_utils import cuda_time_ms, cuda_time_and_memory
import src.module.naive_mom as naive_mom

def first_idx(tensor: torch.Tensor) -> torch.Tensor:
    """
    @brief Renvoie les indices des premiers éléments de chaque séquence dans un tenseur trié
    @param tensor : Tensor trié de taille (N,)
    @return : Tensor de taille (P,) contenant les indices des premiers éléments de chaque séquence
    """
    is_start = torch.ones(tensor.size(0) + 1, dtype=torch.bool, device=tensor.device)
    is_start[1:-1] = tensor[1:] != tensor[:-1]
    first_idx = torch.nonzero(is_start, as_tuple=True)[0]
    return first_idx

class MoM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_memories: int, k: int, update_module: nn.Module = None, *args, **kwargs):
        """
        @brief Module de mixture de mémoires (Mixture of Memories). Il s'agit d'une implémentation varlen optimisée avec triton.
        @param input_dim: Dimension de l'entrée x.
        @param hidden_dim: Dimension de chaque mémoire M_t.
        @param num_memories: Nombre de mémoires (Ça ne prend pas en compte la mémoire partagée).
        @param k: Hyperparamètre k pour la sélection des top-k mémoires.
        """
        super().__init__(*args, **kwargs)

        self.num_memories = num_memories          # locals only
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.k = k

        # note: outputs M+1 blocks (shared + locals)
        self.W_k = nn.Linear(input_dim, hidden_dim * (num_memories + 1))
        self.W_v = nn.Linear(input_dim, hidden_dim * (num_memories + 1))
        self.W_g = nn.Linear(input_dim, num_memories)
        self.W_q = nn.Linear(input_dim, hidden_dim)

        self.update_module = update_module or LinearAttentionVarlenModule()
        self.softmax = nn.Softmax(dim=-1)

    def build_varlen_pack(self, X: torch.Tensor, indices: torch.Tensor, scores: torch.Tensor):
        """
        @brief Réorganise le batch X pour le kernel varlen
        @param X : Batch d'entrée de taille (seq_len, batch_size, dim)
        @param indices : Indices des mémoires sélectionnées de taille (seq_len, batch_size, K)
        @param scores : Scores d'attention associés aux mémoires sélectionnées de taille (seq_len, batch_size, K)
        @return Dictionnaire contenant :
            - 'x_tilde' : Tensor de taille (N, D) avec N = L*B*K, les vecteurs d'entrée réorganisés
            - 't_orig' : Tensor de taille (N,) contenant les indices de la séquence d'origine pour chaque vecteur dans x_tilde
            - 'b_orig' : Tensor de taille (N,) contenant les indices de batch d'origine pour chaque vecteur dans x_tilde
            - 'm_id' : Tensor de taille (N,) contenant les indices de mémoire associés à chaque vecteur dans x_tilde
            - 'alpha' : Tensor de taille (N,) contenant les poids de chaque mémoire dans x_tilde
        """
        L, B, D = X.shape
        assert indices.shape[0] == L and indices.shape[1] == B
        K = indices.shape[2]

        device = X.device
        dtype_idx = torch.int

        # On convertit X en un vecteur de taille (N, D) avec N = L*B*K
        # On "duplique" x k fois les x pour en avoir un par mémoire sélectionnée
        X_rep = X.unsqueeze(2).expand(L, B, K, D).reshape(-1, D) # (N, D)
        # On flatten la matrice des indices de mémoire
        m_ids_rep = indices.reshape(-1).to(dtype_idx) # (N,)
        alpha = scores.reshape(-1) # (N,)

        t_rep = torch.arange(L, device=device, dtype=dtype_idx).view(L, 1, 1).expand(L, B, K).reshape(-1) # (N,)
        b_rep = torch.arange(B, device=device, dtype=dtype_idx).view(1, B, 1).expand(L, B, K).reshape(-1) # (N,)
        # t_rep et b_rep sont les indices respectifs de l'indice dans la séquence et de l'indice de batch,
        # répétés de la même manière que X_rep et m_rep

        # Maintenant on trie selon (b, m, l)
        max_memory = int(m_ids_rep.max().item()) + 1
        key = b_rep * (max_memory * L) + m_ids_rep * L + t_rep # (N,)
        # On trie d'abord par l'indice de batch, puis par l'indice de mémoire, puis par l'indice dans la séquence
        # Les indices de batch prennent les valeurs les plus élevées dans la clé, donc le tri se fait d'abord par batch
        # Ensuite par groupe, puis par position dans la séquence


        perm = torch.argsort(key, stable=True)

        x_tilde = X_rep[perm] # (N, D)
        t_orig = t_rep[perm] # (N,)
        b_orig = b_rep[perm] # (N,)
        m_id = m_ids_rep[perm] # (N,)
        alpha = alpha[perm] # (N,)

        # Construction de s pour le kernel varlen
        # Chaque p correspond à un couple (b, m)
        # On regarde donc quand est-ce que la mémoire ou le batch change dans la liste triée
        s_mem = first_idx(m_id)
        s_batch = first_idx(b_orig)
        s = torch.unique(torch.cat([s_mem, s_batch]), sorted=True)


        return {
            'x_tilde' : x_tilde,
            't_orig' : t_orig,
            'b_orig' : b_orig,
            'm_id' : m_id,
            'alpha' : alpha,
            's' : s
        }

    def forward(
        self,
        X: torch.Tensor, # (T, B, Din)
        M0: torch.Tensor, # (B, M+1, d, d)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        @brief passe-avant du module MoM en version varlen.
        @param X: Entrée de forme (seq_len, batch_size, input_dim)
        @param M0: État initiale des mémoires de forme (hidden_dim, hidden_dim).
        @param update_function: Fonction de mise à jour des mémoires avec varlen.
        @return: Les outputs de forme (seq_len, batch_size, hidden_dim)
        """

        T, B, _ = X.shape

        scores = self.W_g(X)  # (T, B, M)
        scores = self.softmax(scores)  # (T, B, M)
        m_scores, m_indices = torch.topk(scores, self.k) # (T, B, k)

        aux_loss = torch.sum(scores.mean(dim=(0,1)) ** 2)
        m_scores = m_scores / m_scores.sum(dim=-1, keepdim=True) # Normalisation des scores
        m_indices = m_indices + 1 # On décale de 1 car la sélection ne se fait pas sur la mémoire partagée
        m_indices_update = torch.cat([torch.zeros(T, B, 1, dtype=m_indices.dtype, device=X.device), m_indices], dim=2) # On ajoute la mémoire partagée (index 0) aux indices des mémoires à mettre à jour
        m_scores_update = torch.cat([torch.ones(T, B, 1, dtype=m_scores.dtype, device=X.device), m_scores], dim=2)  # On ajoute un score de 1 pour la mémoire partagée

        pack = self.build_varlen_pack(X, m_indices_update, m_scores_update)
        # N = T * B * (k + 1)
        x_tilde = pack["x_tilde"] # (N, Din)
        m_id = pack["m_id"] # (N,)
        Tt = x_tilde.shape[0]
        d = self.hidden_dim
        Mp1 = self.num_memories + 1
        q_tilde = self.W_q(x_tilde) # (N, d)
        # Passe avant sur toutes les mémoires en une seule fois
        K_all = self.W_k(x_tilde).view(Tt, Mp1, d) # (N, M+1, d)
        V_all = self.W_v(x_tilde).view(Tt, Mp1, d) # (N, M+1, d)
        # On sélectionne les clés et valeurs correspondant aux mémoires routées
        k_tilde = K_all[torch.arange(Tt, device=x_tilde.device), m_id] # (N, d)
        v_tilde = V_all[torch.arange(Tt, device=x_tilde.device), m_id] # (N, d)

        o_tilde, m_tilde = self.update_module(q_tilde, k_tilde, v_tilde, pack["s"], M0, x_tilde)  # (N, d)

        o = torch.zeros(T, B, d, device=X.device, dtype=X.dtype)

        t_orig = pack["t_orig"]
        b_orig = pack["b_orig"]
        alpha  = pack["alpha"]

        o.index_put_(
            (t_orig, b_orig),
            alpha.unsqueeze(-1) * o_tilde,
            accumulate=True
        )

        return o, None, aux_loss / T
