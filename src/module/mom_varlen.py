import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
import time
import triton
import triton.language as tl

@triton.jit
def linear_attn_varlen_kernel(q_ptr, k_ptr, v_ptr, o_ptr, # Sans offset
                s_ptr, M_0_ptr, # Avec offset
                d : tl.constexpr, # hidden_dim de la mémoire
                BN: tl.constexpr, # Taille du bloc de mémoire à traiter
                ):
    """
    @brief Kernel triton pour la linear attention avec séquences de longueurs variables
    @param q_ptr : Pointeur vers les query de taille (T_total, d)
    @param k_ptr : Pointeur vers les key de taille (T_total, d)
    @param v_ptr : Pointeur vers les value de taille (T_total, d)
    @param o_ptr : Pointeur vers la sortie de taille (T_total, d)
    @param s_ptr : Pointeur vers la séquence d'indices (num_seqs + 1,)
    @param M_0_ptr : Pointeur vers la mémoire initiale de taille (d, d)
    @param d : hidden_dim des mémoires
    @param BN : Taille du bloc de mémoire à traiter
    """
    pid_p = tl.program_id(0) # Id p de la séquence
    pid_j = tl.program_id(1) # Id j : quel bout de M_p on traite
    j = pid_j * BN + tl.arange(0, BN) # Début du bloc de M (BN,)
    j_mask = j < d # Masque pour ne pas dépasser la dimension de M
    start = tl.load(s_ptr + pid_p).to(tl.int32)      # Début de la séquence p
    end = tl.load(s_ptr + pid_p + 1).to(tl.int32)    # Fin de la séquence p
    # M_cols = tl.zeros((d, BN), dtype=tl.float32)
    i = tl.arange(0, d) # (d,)

    # [None, :] : ajoute une dimension à l'indice 0 (unsqueeze(0))
    # [:, None] : ajoute une dimension à l'indice 1 (unsqueeze(1))
    offs = i[:, None] * d + j[None, :]          # (d, BN)
    mask = j_mask[None, :]  # (1, BN)
    M_cols = tl.load(M_0_ptr + offs, mask=mask, other=0.0).to(tl.float32)  # (d, BN)
    
    t = start
    while t < end:
        kt = tl.load(k_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        vt = tl.load(v_ptr + t * d + j, mask=j_mask, other=0.0).to(tl.float32) # (BN,)
        # M += kt @ vt
        # Produit externe partiel
        M_cols = M_cols + kt[:, None] * vt[None, :]  # (d, BN)
        # Calcul de o_t = q_t @ M
        q_t = tl.load(q_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        o_cols = tl.sum(M_cols * q_t[:, None], axis=0)  # (BN,)
        tl.store(o_ptr + t * d + j, o_cols.to(tl.float32), mask=j_mask)
        t += 1



def linear_attn_varlen_triton(q, k, v, s, M0):
    """
    @brief Wrapper pour le kernel triton de la linear attention avec varlen
    @param q : Query de taille (T_total, d)
    @param k : Key de taille (T_total, d)
    @param v : Value de taille (T_total, d)
    @param s : Séquence d'indices de taille (num_seqs + 1,)
    @param M0 : Mémoire initiale de taille (d, d)
    @return : Sortie de taille (T_total, d)
    """
    # q,k,v: (Tt, d) CUDA contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()
    M0 = M0.contiguous()
    _, d = q.shape
    P = s.numel() - 1
    BN = 32

    o = torch.empty_like(q)

    grid = (P, triton.cdiv(d, BN))
    linear_attn_varlen_kernel[grid](
        q, k, v, o,
        s, M0,
        d=d,
        BN=BN,
        num_warps=4
    )
    return o

def linear_attn_varlen(q, k, v, s, M0):
    """
    @brief Implémentation naïve de la linear attention avec varlen. C'est une version de référence.
    @param q : Query de taille (T_total, d)
    @param k : Key de taille (T_total, d)
    @param v : Value de taille (T_total, d)
    @param s : Séquence d'indices de taille (num_seqs + 1,)
    @param M0 : Mémoire initiale de taille (d, d)
    @return : Sortie de taille (T_total, d)
    """
    _, d = q.shape
    o = torch.zeros_like(q)

    for p in range(len(s) - 1):
        start = s[p].item()
        end = s[p + 1].item()

        M = M0.clone()

        for t in range(start, end):
            kt = k[t].unsqueeze(-1)  # (d, 1)
            vt = v[t].unsqueeze(0)   # (1, d)
            M = M + (kt @ vt)          # (d, d)
            o[t] = q[t] @ M

    return o

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
    def __init__(self, input_dim: int, hidden_dim: int, num_memories: int, k: int, *args, **kwargs):
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
        self.W_g = nn.Linear(input_dim, num_memories)  # routing over locals only
        self.W_q = nn.Linear(input_dim, hidden_dim)

    @torch.no_grad()
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
        dtype_idx = torch.long

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
        varlen_update: Callable = linear_attn_varlen_triton,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        @brief passe-avant du module MoM en version varlen.
        @param X: Entrée de forme (seq_len, batch_size, input_dim)
        @param M0: État initiale des mémoires de forme (hidden_dim, hidden_dim).
        @param varlen_update: Fonction de mise à jour des mémoires avec varlen.
        @return: Les outputs de forme (seq_len, batch_size, hidden_dim)
        """

        T, B, _ = X.shape

        scores = nn.Softmax(dim=-1)(self.W_g(X)) # (T, B, M)
        m_scores, m_indices = torch.topk(scores, self.k) # (T, B, k)
        m_scores = m_scores / m_scores.sum(dim=-1, keepdim=True) # Normalisation des scores
        m_indices = m_indices + 1 # On décale de 1 car la sélection ne se fait pas sur la mémoire partagée
        m_indices_update = torch.cat([torch.zeros(T, B, 1, dtype=torch.long, device=X.device), m_indices], dim=2) # On ajoute la mémoire partagée (index 0) aux indices des mémoires à mettre à jour
        m_scores_update = torch.cat([torch.ones(T, B, 1, dtype=torch.long, device=X.device), m_scores], dim=2)  # On ajoute un score de 1 pour la mémoire partagée

        pack = self.build_varlen_pack(X, m_indices_update, m_scores_update)
        # N = T * B * (k + 1)
        x_tilde = pack["x_tilde"] # (N, Din)
        m_id = pack["m_id"] # (N,)
        Tt = x_tilde.shape[0]
        d = self.hidden_dim
        Mp1 = self.num_memories + 1
        q_tilde = self.W_q(x_tilde) # (N, d)
        K_all = self.W_k(x_tilde).view(Tt, Mp1, d) # (N, M+1, d)
        V_all = self.W_v(x_tilde).view(Tt, Mp1, d) # (N, M+1, d)
        k_tilde = K_all[torch.arange(Tt, device=x_tilde.device), m_id] # (N, d)
        v_tilde = V_all[torch.arange(Tt, device=x_tilde.device), m_id] # (N, d)

        o_tilde = varlen_update(q_tilde, k_tilde, v_tilde, pack["s"], M0)

        o = torch.zeros(T, B, d, device=X.device, dtype=X.dtype)

        t_orig = pack["t_orig"]
        b_orig = pack["b_orig"]
        alpha  = pack["alpha"]

        o.index_put_(
            (t_orig, b_orig),
            alpha.unsqueeze(-1) * o_tilde,
            accumulate=True
        )
        return o