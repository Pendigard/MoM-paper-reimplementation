import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
import time
import triton
import triton.language as tl

@triton.jit
def linear_attn_varlen_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                s_ptr, M_0_ptr, alpha_ptr, m_id_ptr, t_ptr, b_ptr,
                d : tl.constexpr,
                BN: tl.constexpr,

                num_memories : tl.constexpr,
                batch_size : tl.constexpr,
                seq_len : tl.constexpr
                ):
    pid_p = tl.program_id(0) # Id p de la séquence
    pid_j = tl.program_id(1) # Id j : quel bout de M_p on traite
    j = pid_j * BN + tl.arange(0, BN) # Début du bloc de M (BN,)
    j_mask = j < d # Masque pour ne pas dépasser la dimension de M
    start = tl.load(s_ptr + pid_p).to(tl.int32) # Début de la séquence p
    end = tl.load(s_ptr + pid_p + 1).to(tl.int32) # Fin de la séquence p
    # M_cols = tl.zeros((d, BN), dtype=tl.float32)
    i = tl.arange(0, d) # (d,)

    # [None, :] : ajoute une dimension à l'indice 0 (unsqueeze(0))
    # [:, None] : ajoute une dimension à l'indice 1 (unsqueeze(1))
    offs = i[:, None] * d + j[None, :] # (d, BN)

    # Masque pour la sauvegarde m à chaque étape, on s'assure que les colonnes ne dépassent pas d
    # Pas besoin de masque pour les lignes car i < d toujours
    mask = j_mask[None, :] # (1, BN)       
    M_cols = tl.load(M_0_ptr + offs, mask=mask, other=0.0).to(tl.float32)  # (d, BN)
    
    t = start
    while t < end:
        t_orig = tl.load(t_ptr + t).to(tl.int32)
        b_orig = tl.load(b_ptr + t).to(tl.int32)
        m_id = tl.load(m_id_ptr + t).to(tl.int32)
        alpha = tl.load(alpha_ptr + t).to(tl.float32)

        block_kv = t_orig * (batch_size * num_memories) + b_orig * num_memories + m_id

        block_q = t_orig * batch_size + b_orig

        kt = tl.load(k_ptr + block_kv * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        vt = tl.load(v_ptr + block_kv * d + j, mask=j_mask, other=0.0).to(tl.float32) # (BN,)
        # M += kt @ vt
        # Produit externe partiel
        M_cols = M_cols + kt[:, None] * vt[None, :]  # (d, BN)
        # tl.store(m_ptr + t * d * d + offs, M_cols.to(tl.float32), mask=mask)
        # Calcul de o_t = q_t @ M
        q_t = tl.load(q_ptr + block_q * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        contrib = alpha * tl.sum(M_cols * q_t[:, None], axis=0)  # (BN,)
        tl.atomic_add(o_ptr + block_q * d + j, contrib.to(tl.float32), mask=j_mask)
        t += 1

class LinearAttentionVarlenTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, 
                s, M0, alpha, m_id, t_orig, b_orig):
        """
        @brief Wrapper pour le kernel triton de la linear attention avec varlen
        @param q : Query de taille (T_total, d)
        @param k : Key de taille (T_total, d)
        @param v : Value de taille (T_total, d)
        @param s : Séquence d'indices de taille (num_seqs + 1,)
        @param M0 : Mémoire initiale de taille (d, d)
        @param alpha : Poids de chaque contribution de mémoire de taille (T_total,)
        @param m_id : Indice de la mémoire utilisée pour chaque contribution de taille (T_total,)
        @param t_orig : Indice temporel original de chaque contribution de taille (T_total,)
        @param b_orig : Indice de batch original de chaque contribution de taille (T_total,)
        @return : Sortie de taille (T_total, d)
        """
        # q,k,v: (Tt, d) CUDA contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        M0 = M0.contiguous()
        alpha = alpha.contiguous()
        m_id = m_id.contiguous()
        t_orig = t_orig.contiguous()
        b_orig = b_orig.contiguous()
        seq_len, batch_size, num_memories, d = k.shape
        P = s.numel() - 1
        BN = 32

        o = torch.zeros_like(q) # (T, batch_size, d)
        # m = torch.empty((Tt, d, d), device=q.device, dtype=q.dtype)

        grid = (P, triton.cdiv(d, BN))
        linear_attn_varlen_kernel[grid](
            q, k, v, o,
            s, M0, alpha, m_id, t_orig, b_orig,
            d=d,
            BN=BN,

            seq_len=seq_len,
            batch_size=batch_size,
            num_memories=num_memories,

            num_warps=4
        )

        return o

def linear_attn_varlen_triton(q, k, v, s, M0, alpha, m_id, t_orig, b_orig):
    return LinearAttentionVarlenTriton.apply(q, k, v, s, M0, alpha, m_id, t_orig, b_orig)

def linear_attn_varlen(q, k, v, s, M0, alpha, m_id, t_orig, b_orig):
    d = q.shape[-1]
    o = torch.zeros_like(q)
    M_total = []

    for p in range(len(s) - 1):
        start = s[p].item()
        end = s[p + 1].item()

        M = M0.clone()

        for t in range(start, end):
            seq_idx = t_orig[t].item()
            batch_idx = b_orig[t].item()
            mem_idx = m_id[t].item()
            alpha_t = alpha[t].item()



            kt = k[seq_idx, batch_idx, mem_idx].unsqueeze(-1)  # (d, 1)
            vt = v[seq_idx, batch_idx, mem_idx].unsqueeze(0)   # (1, d)
            M = M + (kt @ vt)          # (d, d)
            M_total.append(M.clone())
            o[seq_idx, batch_idx] += alpha_t * (M.T @ q[seq_idx, batch_idx])  # (d,)

    return o

class LinearAttentionVarlenModule(nn.Module):
    def __init__(self, use_triton: bool = False):
        super().__init__()
        self.use_triton = use_triton
        if self.use_triton:
            self.update_function = linear_attn_varlen_triton
        else:
            self.update_function = linear_attn_varlen

    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, M_0 : torch.Tensor, pack : Dict[str, torch.Tensor], *args) -> torch.Tensor:
        return self.update_function(q, k, v, pack['s'], M_0, pack['alpha'], pack['m_id'], pack['t_orig'], pack['b_orig'])
