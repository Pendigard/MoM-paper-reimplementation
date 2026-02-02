import itertools
import logging
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import List
from conllu import parse_incr
from src.module.mom_llm import MoMLLM
import src.module.naive_mom as nm

logging.basicConfig(level=logging.INFO)

DATA_PATH = "data/"


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
                self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))

    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


def test(model, data_test, loss_fn, aux_loss_weight=0.01):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    correct_total = 0
    token_total = 0
    with torch.no_grad():
        for batch_x, batch_y in data_test:
            batch_x = batch_x.to(device).long().permute(1, 0)  # (B, T)
            batch_y = batch_y.to(device).long().permute(1, 0)  # (B, T)


            output, aux_loss = model(batch_x)
            out = output.reshape(-1, output.size(-1))
            target = batch_y.reshape(-1)
            pred = torch.argmax(output, dim=2)

            mask = batch_y != PAD_IX

            correct = (pred == batch_y) & mask

            correct_total += correct.sum().item()
            token_total += mask.sum().item()
            loss = loss_fn(out, target) + aux_loss_weight * aux_loss
            total_loss += loss.item()
    total_loss /= len(data_test)
    return total_loss, correct_total / token_total if token_total > 0 else 0.0

def train(model, data_train, data_valid, loss_fn, optimizer, max_epochs=100, writer=None, verbose=10, patience=10, device='cpu', aux_loss_weight=0.01):
    init_patience = patience
    pbar_epochs = tqdm(total=max_epochs, desc="Training (epochs)", position=0)
    best_valid_loss = float('inf')
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        pbar_batches = tqdm(data_train, desc=f"Epoch {epoch+1}/{max_epochs}", position=1, leave=False)
        for batch_x, batch_y in pbar_batches:
            batch_x = batch_x.to(device).long().permute(1, 0)  # (B, T)
            batch_y = batch_y.to(device).permute(1, 0)  # (B, T)
            optimizer.zero_grad()
            output, aux_loss = model(batch_x)
            out = output.reshape(-1, output.size(-1))
            target = batch_y.reshape(-1)
            loss = loss_fn(out, target) + aux_loss_weight * aux_loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar_batches.set_postfix(batch_loss=f"{loss.item():.4f}")

        valid_loss, valid_accuracy = test(model, data_valid, loss_fn, aux_loss_weight)

        epoch_loss /= len(data_train)
        if writer:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)


        if verbose and (epoch + 1) % verbose == 0:
            tqdm.write(f"Epoch {epoch+1}/{max_epochs} | mean_loss = {epoch_loss:.4f} | valid_loss = {valid_loss:.4f} | valid_accuracy = {valid_accuracy:.2%}")

        pbar_epochs.set_postfix(mean_loss=f"{epoch_loss:.4f}")
        pbar_epochs.update(1)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict()
            patience = init_patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    pbar_epochs.close()
    return model


def get_mem_df(model: nn.Module, data_loader: DataLoader, num_samples: int = 1000):
    model.eval()
    device = next(model.parameters()).device

    memories = []
    with torch.no_grad():
        samples_processed = 0
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device).long()
            B = batch_x.shape[1]
            T = batch_x.shape[0]

            batch_x = batch_x.transpose(0,1)  # (B, T)
            batch_y = batch_y.transpose(0,1)  # (B, T)

            scores, indices = model.get_scores(batch_x)
            hidden_states = model.get_hidden_states(batch_x)
            for b in range(B):
                for t in range(T):
                    if samples_processed >= num_samples:
                        return memories
                    if batch_x[b, t].item() == PAD_IX:
                        continue
                    mem_entry = {
                        "token_id": batch_x[b, t].item(),
                        "time_step": t,
                        "label_id": id2tag[batch_y[b, t].item()]
                    }
                    for layer in range(model.num_layers):
                        _, max_idx = scores[layer, b, t].max(0)
                        selected_memory = indices[layer, b, t, max_idx]
                        mem_entry.update({
                        f"score_{layer}": scores[layer, b, t].cpu().numpy(),
                        f"memory_index_{layer}": indices[layer, b, t].cpu().numpy(),
                        f"selected_memory_{layer}": selected_memory.item(),
                        f"hidden_state_{layer}": hidden_states[layer, b, t].cpu().numpy(),
                        })   
                    memories.append(mem_entry)
                    samples_processed += 1
    return memories

if __name__ == "__main__":
    logging.info("Loading datasets...")
    words = Vocabulary(True)
    tags = Vocabulary(False)


    data_file = open(DATA_PATH+"fr_gsd-ud-train.conllu",encoding="utf-8")
    train_data = TaggingDataset(parse_incr(data_file), words, tags, True)

    data_file = open(DATA_PATH+"fr_gsd-ud-dev.conllu",encoding='utf-8')
    dev_data = TaggingDataset(parse_incr(data_file), words, tags, True)

    data_file = open(DATA_PATH+"fr_gsd-ud-test.conllu",encoding="utf-8")
    test_data = TaggingDataset(parse_incr(data_file), words, tags, False)


    logging.info("Vocabulary size: %d", len(words))

    id2tag = {idx: tag for tag, idx in tags.word2id.items()}


    BATCH_SIZE=100

    train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)

    print("Data loaded.")

    PAD_IX = words.PAD

    model = MoMLLM(
        vocab_size=len(words),
        hidden_dim=64,
        num_memories=4,
        k=2,
        num_layers=2,
        mom_implem=nm.MoM,
        layer_norm=nn.LayerNorm,
        update_module=nm.GLAAttention,
        update_module_args=(64, 64, 4),
        output_size=len(tags)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IX)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(log_dir="runs/pos_tagging_experiment")
    train_flag = True
    if train_flag:
        logging.info("Starting training...")
        model = train(
            model,
            train_loader,
            dev_loader,
            loss_fn,
            optimizer,
            max_epochs=50,
            writer=writer,
            verbose=5,
            patience=5,
            device=device,
            aux_loss_weight=0.01
        )
    else:
        logging.info("Loading pre-trained model...")
        model.load_state_dict(torch.load("pos_tagging_mom_model.pth", map_location=device))
        logging.info("Model loaded.")

    logging.info("Evaluating on test set...")
    test_loss, test_accuracy = test(model, test_loader, loss_fn)
    logging.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2%}")
    writer.close()


    logging.info("Saving model...")
    torch.save(model.state_dict(), "pos_tagging_mom_model.pth")
    logging.info("Model saved to pos_tagging_mom_model.pth")

    logging.info("Extracting memory data...")
    mem_data = get_mem_df(model, train_loader, num_samples=10000)
    mem_df = pd.DataFrame(mem_data)
    mem_df.to_pickle("pos_tagging_memory_data.pkl")
    logging.info("Memory data saved to pos_tagging_memory_data.pkl")

