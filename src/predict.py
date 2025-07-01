import torch
import torch.serialization
# 
from constants import DEVICE
from utils import load_config
from network import LanguageModel
from dataset import Vocab

# Allowlist Vocab class for safe loading
torch.serialization.add_safe_globals([Vocab])

def generate_text(model, vocab, prompt, config, device):
    model.eval()
    tokens = [vocab["<sos>"]] + [vocab[token] for token in prompt.split()]
    input_ids = torch.tensor([tokens]).to(device)
    
    for _ in range(config["prediction"]["max_gen_len"]):
        output, _ = model(input_ids)
        logits = output[:, -1, :] / config["prediction"]["temperature"]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == vocab["<eos>"]:
            break
    
    return " ".join([vocab.get_itos()[idx] for idx in input_ids[0].cpu().numpy()])

if __name__ == "__main__":
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = torch.load(config["data"]["vocab_file"])
    model = LanguageModel(**config["model"]).to(device)
    # Load the latest checkpoint (adjust the epoch number as needed)
    checkpoint = torch.load("lm_checkpoint_epoch_1.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    prompt = "The quick brown fox"
    generated_text = generate_text(model, vocab, prompt, config, device)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")