import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
import time
import triton
import triton.language as tl
from src.module.update_module import LinearAttentionVarlenModule
from src.module.update_module_varlen import LinearAttentionVarlenModule as lavm
# from src.utils.benchmark_utils import cuda_time_ms, cuda_time_and_memory
import src.module.naive_mom as naive_mom
import src.module.mom_varlen as mvo

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

        self.update_module = update_module or LinearAttentionVarlenModule(use_triton=True)
        self.softmax = nn.Softmax(dim=-1)

    def build_varlen_pack(self, indices: torch.Tensor, scores: torch.Tensor, T : int, B : int, device: torch.device) -> Dict[str, torch.Tensor]:
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
        assert indices.shape[0] == T and indices.shape[1] == B
        K = indices.shape[2]
        dtype_idx = torch.int

        # On flatten la matrice des indices de mémoire
        m_ids_rep = indices.reshape(-1).to(dtype_idx) # (N,)
        alpha = scores.reshape(-1) # (N,)

        t_rep = torch.arange(T, device=device, dtype=dtype_idx).view(T, 1, 1).expand(T, B, K).reshape(-1) # (N,)
        b_rep = torch.arange(B, device=device, dtype=dtype_idx).view(1, B, 1).expand(T, B, K).reshape(-1) # (N,)
        # t_rep et b_rep sont les indices respectifs de l'indice dans la séquence et de l'indice de batch,
        # répétés de la même manière que X_rep et m_rep

        # Maintenant on trie selon (b, m, l)
        max_memory = int(m_ids_rep.max().item()) + 1
        key = b_rep * (max_memory * T) + m_ids_rep * T + t_rep # (N,)
        # On trie d'abord par l'indice de batch, puis par l'indice de mémoire, puis par l'indice dans la séquence
        # Les indices de batch prennent les valeurs les plus élevées dans la clé, donc le tri se fait d'abord par batch
        # Ensuite par groupe, puis par position dans la séquence


        perm = torch.argsort(key, stable=True)

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

        pack = self.build_varlen_pack(m_indices_update, m_scores_update, T, B, X.device)

        print(torch.cuda.memory_allocated() / (1024 ** 2), "MB allocated before MoM kernel")
        k = self.W_k(X).reshape(T, B, self.num_memories + 1, self.hidden_dim)
        v = self.W_v(X).reshape(T, B, self.num_memories + 1, self.hidden_dim)
        q = self.W_q(X)
        print(torch.cuda.memory_allocated() / (1024 ** 2), "MB allocated after MoM prep")

        o = self.update_module(q, k, v, M0, pack, X)

        return o, None, aux_loss / T



if __name__ == "__main__":
    # Test rapide du module MoM varlen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, B, Din = 2, 2, 2
    d = 8
    M = 5
    k = 3

    X = torch.randn(T, B, Din).to(device)
    M0 = torch.zeros(d, d).to(device)

    mom_varlen = MoM(Din, d, M, k, update_module=LinearAttentionVarlenModule(use_triton=True)).to(device)

    naive_mom_module = naive_mom.MoMRef(Din, d, M, k, update_module=naive_mom.LinearAttention()).to(device)

    mom_varlen_old = mvo.MoM(Din, d, M, k, update_module=lavm(use_triton=True)).to(device)

    naive_mom_module.load_state_dict(mom_varlen.state_dict())

    mom_varlen_old.load_state_dict(mom_varlen.state_dict())

    o_varlen, _, aux_loss = mom_varlen(X, M0)

    print("Output varlen shape:", o_varlen.shape)

    o_naive, _, _ = naive_mom_module(X, M0)

    print("Output naive shape:", o_naive.shape)

    o_varlen_old, _, _ = mom_varlen_old(X, M0)

    print("Output varlen old shape:", o_varlen_old.shape)

    # Vérification de l'égalité des résultats
    if torch.allclose(o_varlen, o_naive, atol=1e-5):
        print("Les sorties varlen et naïve sont égales !")
    else:
        print("Les sorties varlen et naïve sont différentes.")
        print("Différence max :", (o_varlen - o_naive).abs().max().item())

    if torch.allclose(o_varlen, o_varlen_old, atol=1e-5):
        print("Les sorties varlen et varlen old sont égales !")
    else:
        print("Les sorties varlen et varlen old sont différentes.")
        print("Différence max :", (o_varlen - o_varlen_old).abs().max().item())

    if torch.allclose(o_naive, o_varlen_old, atol=1e-5):
        print("Les sorties naïve et varlen old sont égales !")
    else:
        print("Les sorties naïve et varlen old sont différentes.")
        print("Différence max :", (o_naive - o_varlen_old).abs().max().item())

