import torch
from datasets import load_dataset
from dataset import build_vocabulary, get_data_loader
from utils import load_config

def preprocess(config_path):
    config = load_config(config_path)
    dataset = load_dataset("wikitext", config["data"]["dataset_name"])["train"]["text"]
    dataset = [text for text in dataset if text.strip() != ""]
    vocab = build_vocabulary(dataset, config["model"]["vocab_size"])
    torch.save(vocab, config["data"]["vocab_file"])
    print(f"Vocabulary saved to {config['data']['vocab_file']}")

if __name__ == "__main__":
    preprocess("config.yaml")