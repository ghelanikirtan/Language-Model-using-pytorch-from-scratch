import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torch.serialization
from tqdm import tqdm
from collections import OrderedDict
import logging
import os
from constants import DEVICE, DEVICE_STR
from utils import load_config
from network import LanguageModel
from dataset import get_data_loader, Vocab

torch.serialization.add_safe_globals([Vocab])

def setup_logging(log_file):
    """Set up logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class CustomScheduler:
    """Learning rate scheduler with warmup and inverse square root decay."""
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        """Update learning rate based on step number."""
        self.step_num += 1
        lr = self.d_model ** -0.5 * min(self.step_num ** -0.5, 
                                       self.step_num * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

def train_model(config):
    """Train the language model with the given configuration."""
    
    log_file = config.get("training", {}).get("log_file", "training.log")
    setup_logging(log_file)
    logging.info("Starting training process...")
    
    # Load vocabulary
    vocab_path = config["data"]["vocab_file"]
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file {vocab_path} not found. Run preprocess_data.py first.")
    vocab = torch.load(vocab_path)
    logging.info(f"Loaded vocabulary with {len(vocab)} tokens.")
    
    # Initialize model, criterion, optimizer, and scheduler
    model = LanguageModel(**config["model"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab["<pad>"],
        label_smoothing=float(config["training"]["label_smoothing"])
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
        eps=float(config["training"]["epsilon"])
    )
    scheduler = CustomScheduler(
        optimizer,
        config["model"]["d_model"],
        config["training"]["warmup_steps"]
    )
    scaler = GradScaler(device=DEVICE_STR)
    
    # Load data
    try:
        train_loader = get_data_loader(config["data"]["dataset_name"], "train", config)
        logging.info(f"Loaded training dataset with {len(train_loader)} batches.")
    except Exception as e:
        logging.error(f"Failed to load training dataset: {e}")
        raise
    
    # Ensure checkpoint directory exists
    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.train()
    for epoch in range(config["training"]["epochs"]):
        total_loss = 0
        pbar = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{config['training']['epochs']}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE_STR):
                output, _ = model(src)
                output = output.view(-1, len(vocab))
                tgt = tgt.view(-1)
                loss = criterion(output, tgt)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
            lr = scheduler.step()
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(OrderedDict(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
                lr=f"{lr:.6f}"
            ))
            pbar.update(1)
        
        avg_loss = total_loss / len(train_loader)
        pbar.close()
        logging.info(f"Epoch {epoch+1}/{config['training']['epochs']} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"lm_checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "avg_loss": avg_loss
        }, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    logging.info("Training completed.")
    return model

if __name__ == "__main__":
    config = load_config("config.yaml")
    train_model(config)