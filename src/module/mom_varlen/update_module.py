import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
import time
import triton
import triton.language as tl

@triton.jit
def linear_attn_varlen_kernel(q_ptr, k_ptr, v_ptr, o_ptr, m_ptr, # Sans offset
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
        kt = tl.load(k_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        vt = tl.load(v_ptr + t * d + j, mask=j_mask, other=0.0).to(tl.float32) # (BN,)
        # M += kt @ vt
        # Produit externe partiel
        M_cols = M_cols + kt[:, None] * vt[None, :]  # (d, BN)
        tl.store(m_ptr + t * d * d + offs, M_cols.to(tl.float32), mask=mask)
        # Calcul de o_t = q_t @ M
        q_t = tl.load(q_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        o_cols = tl.sum(M_cols * q_t[:, None], axis=0)  # (BN,)
        tl.store(o_ptr + t * d + j, o_cols.to(tl.float32), mask=j_mask)
        t += 1

@triton.jit
def linear_attn_varlen_backward_kernel(q_ptr, k_ptr, v_ptr, m_ptr, grad_o_ptr,
                s_ptr,
                grad_q_ptr, grad_k_ptr, grad_v_ptr,
                d : tl.constexpr,
                BN: tl.constexpr,
                ):
    pid_p = tl.program_id(0)
    pid_j = tl.program_id(1)
    j = pid_j * BN + tl.arange(0, BN)
    j_mask = j < d
    start = tl.load(s_ptr + pid_p).to(tl.int32)
    end = tl.load(s_ptr + pid_p + 1).to(tl.int32)
    i = tl.arange(0, d)
    offs = i[:, None] * d + j[None, :]
    mask = j_mask[None, :]
    t = end - 1
    q_T_delta_acc = tl.zeros((d, BN), dtype=tl.float32)
    delta_T_q_acc = tl.zeros((BN, d), dtype=tl.float32)
    while t >= start:
        kt = tl.load(k_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        vt = tl.load(v_ptr + t * d + j, mask=j_mask, other=0.0).to(tl.float32) # (BN,)
        qt = tl.load(q_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        m = tl.load(m_ptr + t * d * d + offs, mask=mask, other=0.0).to(tl.float32) # (d, BN)
        delta_o = tl.load(grad_o_ptr + t * d + j, mask=j_mask, other=0.0).to(tl.float32) # (BN,)

        q_T_delta_acc += qt[:, None] * delta_o[None, :] # (d, BN)
        delta_T_q_acc += delta_o[:, None] * qt[None, :] # (BN, d)

        grad_v_t = tl.sum(kt[None, :] * delta_T_q_acc, axis=1)  # (BN,)
        grad_k_t = tl.sum(vt[None, :] * q_T_delta_acc, axis=1)  # (d,)
        grad_q_t = tl.sum(delta_o[None, :] * m, axis=1)  # (d,)

        tl.store(grad_q_ptr + t * d + i, grad_q_t.to(tl.float32), mask=True)
        tl.store(grad_k_ptr + t * d + i, grad_k_t.to(tl.float32), mask=True)
        tl.store(grad_v_ptr + t * d + j, grad_v_t.to(tl.float32), mask=j_mask)

        t -= 1

class LinearAttentionVarlenTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, M0):
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
        Tt, d = q.shape
        P = s.numel() - 1
        BN = 32

        o = torch.empty_like(q)
        m = torch.empty((Tt, d, d), device=q.device, dtype=q.dtype)

        grid = (P, triton.cdiv(d, BN))
        linear_attn_varlen_kernel[grid](
            q, k, v, o, m,
            s, M0,
            d=d,
            BN=BN,
            num_warps=4
        )

        ctx.s = s
        # ctx.M0 = M0
        ctx.save_for_backward(q, k, v, m)
        return o, m

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, m = ctx.saved_tensors
        s = ctx.s
        _, d = q.shape
        P = s.numel() - 1
        BN = 32

        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)

        grid = (P, triton.cdiv(d, BN))
        linear_attn_varlen_backward_kernel[grid](
            q, k, v, m, grad_o,
            s,
            grad_q, grad_k, grad_v,
            d=d,
            BN=BN,
            num_warps=4
        )
        # On retourne les gradients dans l'ordre des entrées de la fonction forward
        # None pour s et M0 car se ne sont pas des paramètres apprenables
        return grad_q, grad_k, grad_v, None, None 

def linear_attn_varlen_triton(q, k, v, s, M0):
    return LinearAttentionVarlenTriton.apply(q, k, v, s, M0)

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
    M_total = []

    for p in range(len(s) - 1):
        start = s[p].item()
        end = s[p + 1].item()

        M = M0.clone()

        for t in range(start, end):
            kt = k[t].unsqueeze(-1)  # (d, 1)
            vt = v[t].unsqueeze(0)   # (1, d)
            M = M + (kt @ vt)          # (d, d)
            M_total.append(M.clone())
            o[t] = q[t] @ M  # (d,)

    return o, torch.stack(M_total)

class LinearAttentionVarlenModule(nn.Module):
    def __init__(self, use_triton: bool = False):
        super().__init__()
        self.use_triton = use_triton
        if self.use_triton:
            self.update_function = linear_attn_varlen_triton
        else:
            self.update_function = linear_attn_varlen

    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, s : torch.Tensor, M_0 : torch.Tensor, *args) -> torch.Tensor:
        return self.update_function(q, k, v, s, M_0)


@triton.jit
def gla_varlen_kernel(q_ptr, k_ptr, v_ptr, o_ptr, m_ptr, a_ptr,
                s_ptr, M_0_ptr, 
                d : tl.constexpr,
                BN: tl.constexpr,
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
        kt = tl.load(k_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        vt = tl.load(v_ptr + t * d + j, mask=j_mask, other=0.0).to(tl.float32) # (BN,)
        at = tl.load(a_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        # M = (a_t @ one_col) * M + (kt @ vt)
        M_cols = at[:, None] * M_cols + (kt[:, None] * vt[None, :])          # (d, BN)
        tl.store(m_ptr + t * d * d + offs, M_cols.to(tl.float32), mask=mask)
        # Calcul de o_t = q_t @ M
        q_t = tl.load(q_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
        o_cols = tl.sum(M_cols * q_t[:, None], axis=0)  # (BN,)
        tl.store(o_ptr + t * d + j, o_cols.to(tl.float32), mask=j_mask)
        t += 1

# @triton.jit
# def gla_varlen_backward_kernel(q_ptr, k_ptr, v_ptr, m_ptr, grad_o_ptr,
#                 s_ptr, M_0_ptr, cumprod_ptr,
#                 grad_q_ptr, grad_k_ptr, grad_v_ptr, grad_a_ptr,
#                 d : tl.constexpr,
#                 BN: tl.constexpr,
#                 ):
#     pid_p = tl.program_id(0)
#     pid_j = tl.program_id(1)
#     j = pid_j * BN + tl.arange(0, BN)
#     j_mask = j < d
#     start = tl.load(s_ptr + pid_p).to(tl.int32)
#     end = tl.load(s_ptr + pid_p + 1).to(tl.int32)
#     i = tl.arange(0, d)
#     offs = i[:, None] * d + j[None, :]
#     mask = j_mask[None, :]
#     t = end - 1
#     q_t_acc = tl.zeros((d,), dtype=tl.float32)
#     delta_o_acc = tl.zeros((BN,), dtype=tl.float32)
#     while t >= start:
#         kt = tl.load(k_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
#         vt = tl.load(v_ptr + t * d + j, mask=j_mask, other=0.0).to(tl.float32) # (BN,)
#         qt = tl.load(q_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
#         m = tl.load(m_ptr + t * d * d + offs, mask=mask, other=0.0).to(tl.float32) # (d, BN)
#         delta_o = tl.load(grad_o_ptr + t * d + j, mask=j_mask, other=0.0).to(tl.float32) # (BN,)
#         at = tl.load(cumprod_ptr + t * d + i, mask=True, other=0.0).to(tl.float32) # (d,)
#         if t > start:
#             m_prev = tl.load(m_ptr + (t - 1) * d * d + offs, mask=mask, other=0.0).to(tl.float32) # (d, BN)
#         else:
#             m_prev = tl.load(M_0_ptr + offs, mask=mask, other=0.0).to(tl.float32) # (d, BN)

#         q_t_acc += qt
#         delta_o_acc += delta_o
#         # q_T_delta_acc += qt[:, None] * delta_o[None, :] # (d, BN)
#         # delta_T_q_acc += delta_o[:, None] * qt[None, :] # (BN, d)
#         # q_T_delta_acc = q_T_delta_acc * at[:, None]  # (d, BN)
#         # delta_T_q_acc = delta_T_q_acc * at[None, :]  # (BN, d)

#         grad_v_t = tl.sum(kt[None, :] * delta_T_q_acc, axis=1)  # (BN,)
#         grad_k_t = tl.sum(vt[None, :] * q_T_delta_acc, axis=1)  # (d,)
#         grad_q_t = tl.sum(delta_o[None, :] * m, axis=1)  # (d,)
#         grad_a_t = tl.sum(m_prev * q_T_delta_acc, axis=1)  # (d, BN)

        
#         tl.store(grad_q_ptr + t * d + i, grad_q_t.to(tl.float32), mask=True)
#         tl.store(grad_k_ptr + t * d + i, grad_k_t.to(tl.float32), mask=True)
#         tl.store(grad_v_ptr + t * d + j, grad_v_t.to(tl.float32), mask=j_mask)
#         tl.store(grad_a_ptr + t * d + i, grad_a_t.to(tl.float32), mask=True)

#         t -= 1


class GLAVarlenTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, M0, a):
        """
        @brief Wrapper pour le kernel triton de la GLA avec varlen
        @param q : Query de taille (T_total, d)
        @param k : Key de taille (T_total, d)
        @param v : Value de taille (T_total, d)
        @param s : Séquence d'indices de taille (num_seqs + 1,)
        @param M0 : Mémoire initiale de taille (d, d)
        @param a : Poids d'attention de taille (T_total, d)
        @return : Sortie de taille (T_total, d)
        """
        # q,k,v: (Tt, d) CUDA contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        M0 = M0.contiguous()
        a = a.contiguous()
    
        Tt, d = q.shape
        P = s.numel() - 1
        BN = 32

        o = torch.empty_like(q)
        m = torch.empty((Tt, d, d), device=q.device, dtype=q.dtype)

        grid = (P, triton.cdiv(d, BN))
        gla_varlen_kernel[grid](
            q, k, v, o, m, a,
            s, M0,
            d=d,
            BN=BN,
            num_warps=4
        )

        ctx.s = s
        ctx.M0 = M0
        ctx.save_for_backward(q, k, v, m, a)
        return o, m

    @staticmethod
    def backward(ctx, grad_o):
        # q, k, v, m, a = ctx.saved_tensors
        # s = ctx.s
        # M0 = ctx.M0
        # _, d = q.shape
        # P = s.numel() - 1
        # BN = 32

        # grad_q = torch.empty_like(q)
        # grad_k = torch.empty_like(k)
        # grad_v = torch.empty_like(v)
        # grad_a = torch.empty_like(a)

        # cumprod = torch.cat([a.new_ones((1, d)), a[:-1]]).cumprod(0)

        # grid = (P, triton.cdiv(d, BN))
        # gla_varlen_backward_kernel[grid](
        #     q, k, v, m, grad_o,
        #     s, M0, cumprod,
        #     grad_q, grad_k, grad_v, grad_a,
        #     d=d,
        #     BN=BN,
        #     num_warps=4
        # )

        return None, None, None, None, None, None

def GLA_varlen_triton(q, k, v, s, M0, a):
    return GLAVarlenTriton.apply(q, k, v, s, M0, a)


def GLA_varlen(q, k, v, s, M0, a):
    """
    @brief Implémentation naïve de la GLA avec varlen. C'est une version de référence.
    @param q : Query de taille (T_total, d)
    @param k : Key de taille (T_total, d)
    @param v : Value de taille (T_total, d)
    @param s : Séquence d'indices de taille (num_seqs + 1,)
    @param M0 : Mémoire initiale de taille (d, d)
    @param a : Poids d'attention de taille (T_total, d)
    @return : Sortie de taille (T_total, d)
    """
    _, d = q.shape
    o = torch.zeros_like(q)
    one_col = torch.ones((1, d), device=q.device, dtype=q.dtype)
    M_total = []

    for p in range(len(s) - 1):
        start = s[p].item()
        end = s[p + 1].item()

        M = M0.clone()

        for t in range(start, end):
            kt = k[t].unsqueeze(-1)  # (d, 1)
            vt = v[t].unsqueeze(0)   # (1, d)
            at = a[t].unsqueeze(-1)  # (d, 1)
            M = at * M + (kt @ vt) # (d, d)
            M_total.append(M.clone())
            o[t] = q[t] @ M  # (d,)

    return o, torch.stack(M_total)

class GLAVarlenModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, use_triton: bool = False):
        super().__init__()
        self.use_triton = use_triton
        self.W_a = nn.Linear(input_dim, hidden_dim)
        if self.use_triton:
            self.update_function = GLA_varlen_triton
        else:
            self.update_function = GLA_varlen

    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, s : torch.Tensor, M_0 : torch.Tensor, x_tilde : torch.Tensor) -> torch.Tensor:
        a = self.W_a(x_tilde)  # (N, d)
        a = torch.sigmoid(a)  # (N, d)
        return self.update_function(q, k, v, s, M_0, a)