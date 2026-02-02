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
import src.module.mom_varlen.mom_varlen as mvo
import triton
import triton.language as tl

@triton.jit
def linear_attn_varlen_kernel_mem(x_ptr, w_q_ptr, b_q_ptr, w_k_ptr, b_k_ptr, w_v_ptr, b_v_ptr, o_ptr, # Sans offset
                s_ptr, M_0_ptr, alpha_ptr, m_id_ptr, t_ptr, b_ptr,
                d : tl.constexpr, # hidden_dim de la mémoire
                BI: tl.constexpr, # Taille du bloc de mémoire à traiter
                BJ: tl.constexpr, # Taille du bloc de mémoire à traiter
                BK: tl.constexpr,

                num_memories : tl.constexpr,
                batch_size : tl.constexpr,
                seq_len : tl.constexpr
                ):
    pid_p = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_i = tl.program_id(2)

    j = pid_j * BJ + tl.arange(0, BJ)         # (BJ,)
    i = pid_i * BI + tl.arange(0, BI)         # (BI,)

    j_mask = j < d
    i_mask = i < d
    start = tl.load(s_ptr + pid_p).to(tl.int32) # Début de la séquence p
    end = tl.load(s_ptr + pid_p + 1).to(tl.int32) # Fin de la séquence p

    # [None, :] : ajoute une dimension à l'indice 0 (unsqueeze(0))
    # [:, None] : ajoute une dimension à l'indice 1 (unsqueeze(1))
    offs = i[:, None] * d + j[None, :] # (d, BN)

    # Masque pour la sauvegarde m à chaque étape, on s'assure que les colonnes ne dépassent pas d
    # Pas besoin de masque pour les lignes car i < d toujours
    offs_M = i[:, None] * d + j[None, :]      # (BI,BJ)
    M_block = tl.load(M_0_ptr + offs_M,
                    mask=i_mask[:, None] & j_mask[None, :],
                    other=0.0).to(tl.float32)
    
    t = start
    while t < end:
        t_orig = tl.load(t_ptr + t).to(tl.int32)
        b_orig = tl.load(b_ptr + t).to(tl.int32)
        m_id = tl.load(m_id_ptr + t).to(tl.int32)
        alpha = tl.load(alpha_ptr + t).to(tl.float32)
        
        block_x = t_orig * (batch_size * d) + b_orig * d

        x = tl.load(x_ptr + block_x + i, mask=i_mask, other=0.0).to(tl.float32)   # (BI,)
        kt_i = tl.zeros((BI,), dtype=tl.float32)
        q_i  = tl.zeros((BI,), dtype=tl.float32)
        vt_j = tl.zeros((BJ,), dtype=tl.float32)

        # réduction sur k
        for kk in range(0, d, BK):
            k = kk + tl.arange(0, BK)                          # (BK,)
            k_mask = k < d

            x_k = tl.load(x_ptr + block_x + k, mask=k_mask, other=0.0).to(tl.float32)  # (BK,)

            # ---- W_k: rows = (m_id*d + i), cols = k  => (BI,BK)
            Wk = tl.load(
                w_k_ptr + (m_id * d + i)[:, None] * d + k[None, :],
                mask=i_mask[:, None] & k_mask[None, :],
                other=0.0
            ).to(tl.float32)
            kt_i += tl.sum(Wk * x_k[None, :], axis=1)          # (BI,)

            # ---- W_q: rows = i, cols = k => (BI,BK)
            Wq = tl.load(
                w_q_ptr + i[:, None] * d + k[None, :],
                mask=i_mask[:, None] & k_mask[None, :],
                other=0.0
            ).to(tl.float32)
            q_i += tl.sum(Wq * x_k[None, :], axis=1)           # (BI,)

            # ---- W_v: rows = (m_id*d + j), cols = k => (BJ,BK)
            Wv = tl.load(
                w_v_ptr + (m_id * d + j)[:, None] * d + k[None, :],
                mask=j_mask[:, None] & k_mask[None, :],
                other=0.0
            ).to(tl.float32)
            vt_j += tl.sum(Wv * x_k[None, :], axis=1)          # (BJ,)
        
        kt_i += tl.load(b_k_ptr + m_id * d + i, mask=i_mask, other=0.0).to(tl.float32)  # (BI,)
        q_i  += tl.load(b_q_ptr + i,           mask=i_mask, other=0.0).to(tl.float32)    # (BI,)
        vt_j += tl.load(b_v_ptr + m_id * d + j, mask=j_mask, other=0.0).to(tl.float32)   # (BJ,)

        # M += kt @ vt
        # Produit externe partiel
        M_block = M_block + kt_i[:, None] * vt_j[None, :]  # (BI, BJ)
        # Calcul de o_t = q_t @ M
        contrib = alpha * tl.sum(M_block * q_i[:, None], axis=0)  # (BJ,)
        tl.atomic_add(o_ptr + block_x + j, contrib, mask=j_mask)
        # Atomic add permet une écriture concurrente sur o_ptr
        t += 1

def linear_attn_varlen_triton_mem(x, w_q, b_q, w_k, b_k, w_v, b_v,
                          s, M0, alpha, m_id, t_orig, b_orig):
    seq_len, batch_size, d = x.shape
    num_memories = w_k.shape[0] // d
    Tt = s[-1].item()
    P = s.numel() - 1
    BI = 32
    BJ = 16
    BK = 16

    o = torch.zeros_like(x)

    grid = (P, triton.cdiv(d, BI), triton.cdiv(d, BJ))
    linear_attn_varlen_kernel_mem[grid](
        x, w_q, b_q, w_k, b_k, w_v, b_v, o,
        s, M0, alpha, m_id, t_orig, b_orig,
        d=d,
        BI=BI,
        BJ=BJ,
        BK=BK,
        num_memories=num_memories,
        batch_size=batch_size,
        seq_len=seq_len,
        num_warps=4
    )
    
    return o

class LinearAttentionVarlenModuleMem(nn.Module):
    def __init__(self, use_triton: bool = True):
        super(LinearAttentionVarlenModuleMem, self).__init__()
        self.use_triton = use_triton

    def forward(
        self,
        x: torch.Tensor, # (T, B, d)
        w_q: torch.Tensor,
        b_q: torch.Tensor,
        w_k: torch.Tensor,
        b_k: torch.Tensor,
        w_v: torch.Tensor,
        b_v: torch.Tensor,
        M0: torch.Tensor, # (d, d)
        pack: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        @brief passe-avant de la linear attention avec varlen.
        @param x : Entrée de forme (seq_len, batch_size, hidden_dim)
        @param w_q, b_q : Poids et biais de la couche linéaire pour les queries
        @param w_k, b_k : Poids et biais de la couche linéaire pour les keys
        @param w_v, b_v : Poids et biais de la couche linéaire pour les values
        @param M0 : Mémoire initiale de forme (hidden_dim, hidden_dim).
        @param pack : Dictionnaire contenant les tenseurs nécessaires pour le traitement varlen.
        @return : Les outputs de forme (seq_len, batch_size, hidden_dim)
        """

        if self.use_triton:
            # current memory usage
            o = linear_attn_varlen_triton_mem(
                x, w_q, b_q, w_k, b_k, w_v, b_v,
                pack['s'], M0, pack['alpha'], pack['m_id'], pack['t_orig'], pack['b_orig']
            )
        else:
            raise NotImplementedError("Seule l'implémentation Triton est disponible pour le moment.")

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

        self.update_module = update_module or LinearAttentionVarlenModuleMem(use_triton=True)
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

        o = self.update_module(X, self.W_q.weight, self.W_q.bias,
                              self.W_k.weight, self.W_k.bias,
                              self.W_v.weight, self.W_v.bias,
                              M0, pack)
                              

        return o, None, aux_loss / T



if __name__ == "__main__":
    # Test rapide du module MoM varlen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, B, Din = 2, 2, 64
    d = 64
    M = 5
    k = 3

    X = torch.randn(T, B, Din).to(device)
    M0 = torch.zeros(d, d).to(device)

    mom_varlen = MoM(Din, d, M, k).to(device)

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

