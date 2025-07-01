import torch
import torch.serialization
from network import LanguageModel
from dataset import Vocab
from utils import load_config
from constants import DEVICE

# Allowlist Vocab class for safe loading
torch.serialization.add_safe_globals([Vocab])

def generate_text(model, vocab, prompt, config, device):
    model.eval()
    tokens = [vocab["<sos>"]] + [vocab[token] for token in prompt.split()]
    input_ids = torch.tensor([tokens]).to(device)
    
    for _ in range(config["prediction"]["max_gen_len"]):
        with torch.no_grad():
            output, _ = model(input_ids)
        logits = output[:, -1, :] / config["prediction"]["temperature"]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == vocab["<eos>"]:
            break
    
    return " ".join([vocab.get_itos()[idx] for idx in input_ids[0].cpu().numpy()])

def main():
    # Load configuration and vocabulary
    config = load_config("config.yaml")
    vocab = torch.load(config["data"]["vocab_file"])
    
    # Initialize model and load checkpoint
    model = LanguageModel(**config["model"]).to(DEVICE)
    checkpoint = torch.load("checkpoints/lm_checkpoint_epoch_10.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.load_state_dict(checkpoint)
    
    print("Language Model Ready! Enter a prompt to generate text (or 'quit' to exit).")
    print(f"Example prompt: 'The quick brown fox'")
    
    while True:
        prompt = input("Enter prompt: ").strip()
        if prompt.lower() == "quit":
            print("Exiting...")
            break
        if not prompt:
            print("Please enter a non-empty prompt.")
            continue
        
        try:
            generated_text = generate_text(model, vocab, prompt, config, DEVICE)
            print(f"Generated: {generated_text}")
        except Exception as e:
            print(f"Error generating text: {e}")

if __name__ == "__main__":
    main()