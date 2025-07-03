import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.serialization
from tqdm import tqdm
#  
from constants import DEVICE
from utils import load_config
from network import LanguageModel
from dataset import get_data_loader, Vocab



torch.serialization.add_safe_globals([Vocab])

def validate_model(model:LanguageModel, 
             config:dict):
    
    
    model.eval()
    
    vocab = torch.load(config['data']['vocab_file'])
    val_loader = get_data_loader(config["data"]["dataset_name"], "validation", config)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"], label_smoothing=config["training"]["label_smoothing"])
    
    
    total_loss = 0 
    pbar = tqdm(total = len(val_loader))
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output, _ = model(src)
            output = output.view(-1, len(vocab))
            loss = criterion(output, tgt)
            total_loss += loss

            postfix = OrderedDict([
                ('val_loss', f"{loss.item():.4f}")
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
            
            
    perplexity = math.exp(total_loss / len(val_loader))
    return perplexity
        
if __name__ == "__main__":
    config = load_config("config.yaml")
    vocab = torch.load(config["data"]["vocab_file"])
    checkpoint = torch.load("checkpoints/lm_checkpoint_epoch_19.pt")
    model = LanguageModel(**config["model"]).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    perplexity = validate_model(model, config)
    print(f"Validation Perplexity: {perplexity:.4f}")