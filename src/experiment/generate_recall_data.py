import torch
from torch.utils.data import Dataset, DataLoader

class RecallDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_examples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_examples = num_examples
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):

        half_vocab = self.vocab_size // 2
        
        num_pairs = (self.seq_len - 1) // 2
        
        keys = torch.randint(0, half_vocab, (num_pairs,))
        values = torch.randint(half_vocab, self.vocab_size, (num_pairs,))
        
        sequence = torch.empty(num_pairs * 2, dtype=torch.long)
        sequence[0::2] = keys
        sequence[1::2] = values
        
        query_idx = torch.randint(0, num_pairs, (1,)).item()
        query_key = keys[query_idx]   
        target_val = values[query_idx] 
        
        input_ids = torch.cat([sequence, torch.tensor([query_key])])
        
        input_ids = input_ids[:self.seq_len]

        labels = torch.full_like(input_ids, -100)
        labels[-1] = target_val
        
        return input_ids, labels

def generate_recall_data(vocab_size=100, seq_len=128, num_examples=1000, batch_size=32):
    if seq_len % 2 == 0:
        seq_len -= 1
        
    dataset = RecallDataset(vocab_size, seq_len, num_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader