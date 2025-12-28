import itertools
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from conllu import parse_incr
from src.module.mom_pipeline import MoMPipeline
from src.module.mom import MoM, LinearAttention
import pickle

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



#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)


def test(model, data_test, loss_fn):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in data_test:
            batch_x = batch_x.to(device).long()

            output = model(batch_x)
            out = output.permute(0, 2, 1)
            loss = loss_fn(out, batch_y)
            total_loss += loss.item()
    total_loss /= len(data_test)
    return total_loss

def train(model, data_train, data_valid, loss_fn, optimizer, max_epochs=100, writer=None, verbose=10, patience=10):
    device = next(model.parameters()).device
    init_patience = patience
    pbar_epochs = tqdm(total=max_epochs, desc="Training (epochs)", position=0)
    best_valid_loss = float('inf')
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        pbar_batches = tqdm(data_train, desc=f"Epoch {epoch+1}/{max_epochs}", position=1, leave=False)
        for batch_x, batch_y in pbar_batches:
            batch_x = batch_x.to(device).long()
            optimizer.zero_grad()
            output = model(batch_x)
            out = output.permute(0, 2, 1)
            loss = loss_fn(out, batch_y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar_batches.set_postfix(batch_loss=f"{loss.item():.4f}")

        
        valid_loss = test(model, data_valid, loss_fn)

        epoch_loss /= len(data_train)
        if writer:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)


        if verbose and (epoch + 1) % verbose == 0:
            tqdm.write(f"Epoch {epoch+1}/{max_epochs} | mean_loss = {epoch_loss:.4f} | valid_loss = {valid_loss:.4f}")

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

def compute_score(model, data_test):
    device = next(model.parameters()).device
    model.eval()
    accuracy = 0.0
    with torch.no_grad():
        for batch_x, batch_y in data_test:
            batch_x = batch_x.to(device).long()

            output = model(batch_x)
            out = output.permute(0, 2, 1)
            pred = torch.argmax(out, dim=1)
            correct = (pred == batch_y).float()
            accuracy = correct.sum() / correct.numel()
    return accuracy

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


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)

print("Data loaded.")

PAD_IX = words.PAD

PoSTagger = MoMPipeline(
    input_dim=len(words),
    embedding_dim=32,
    hidden_dim=16,
    output_dim=len(tags),
    num_memories=5,
    k=2,
    update_module=LinearAttention()
    )


PoSTagger = train(
    model=PoSTagger,
    data_train=train_loader,
    data_valid=dev_loader,
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=PAD_IX),
    optimizer=optim.Adam(PoSTagger.parameters(), lr=0.001),
    max_epochs=100,
    writer=SummaryWriter(log_dir="./logs/tagging"),
    verbose=1,
    patience=5
)

test_loss = test(PoSTagger, test_loader, torch.nn.CrossEntropyLoss(ignore_index=PAD_IX))
test_accuracy = compute_score(PoSTagger, test_loader)
logging.info(f"Test loss: {test_loss:.4f}")
logging.info(f"Test accuracy: {test_accuracy:.4%}")
