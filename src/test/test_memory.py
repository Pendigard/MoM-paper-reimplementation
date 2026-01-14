import torch
from src import mom_varlen


batch_size = 32
input_dim = 10
hidden_dim = 10
module = mom_varlen.Memory(input_dim, hidden_dim)
X = torch.randn(32, input_dim)
M_t = torch.zeros(32, hidden_dim, hidden_dim)
output = module(M_t, X)
assert output.shape == (batch_size, hidden_dim, hidden_dim)
