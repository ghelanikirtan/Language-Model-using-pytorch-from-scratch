from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter
import os

class Vocab:
    def __init__(self, vocab, itos):
        self.vocab = vocab
        self.itos = itos
    
    def __getitem__(self, key):
        return self.vocab.get(key, self.vocab["<unk>"])
    
    def __len__(self):
        return len(self.vocab)
    
    def get_itos(self):
        return self.itos

class TextDataset(Dataset):
    
    def __init__(self, texts, vocab, max_len=64):
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].split()
        tokens = [self.vocab["<sos>"]] + [self.vocab[token] for token in text[:self.max_len-2]] + [self.vocab["<eos>"]]
        if len(tokens) < self.max_len:
            tokens += [self.vocab["<pad>"]] * (self.max_len - len(tokens))
        
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])
    
def build_vocabulary(texts, max_size=30000):
    # Word counts
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    # select top size
    specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
    vocab_words = specials + [word for word, _ in counter.most_common(max_size - len(specials))]
    # Create vocab dictionary
    vocab = {word: idx for idx, word in enumerate(vocab_words)}
    vocab["<unk>"] = vocab.get("<unk>", 0)  # Ensure <unk> is index 0
    vocab["<pad>"] = vocab.get("<pad>", 1)
    vocab["<sos>"] = vocab.get("<sos>", 2)
    vocab["<eos>"] = vocab.get("<eos>", 3)
    
    # Create index-to-string mapping
    itos = {idx: word for word, idx in vocab.items()}
    
    return Vocab(vocab, itos)

def get_data_loader(dataset_name, split, config):
    dataset = load_dataset("wikitext", dataset_name)[split]["text"]
    dataset = [text for text in dataset if text.strip() != ""]  # Remove empty strings
    vocab = build_vocabulary(dataset, config["model"]["vocab_size"])
    torch.save(vocab, config["data"]["vocab_file"])
    data = TextDataset(dataset, vocab, config["data"]["max_seq_len"])
    batch_size = config["training"]["batch_size"] if split == "train" else config["validation"]["batch_size"]
    return DataLoader(data, batch_size=batch_size, shuffle=(split == "train"))