import torch
import torch.serialization
from network import LanguageModel
from dataset import Vocab
from utils import load_config
from constants import DEVICE

torch.serialization.add_safe_globals([Vocab])

def generate_text(model, vocab, prompt, config, device):
    model.eval()
    tokens = [vocab["<sos>"]] + [vocab[token.lower()] for token in prompt.split()]
    input_ids = torch.tensor([tokens]).to(device)
    
    for _ in range(config["prediction"]["max_gen_len"]):
        with torch.no_grad():
            output, _ = model(input_ids)
        logits = output[:, -1, :] / config["prediction"]["temperature"]
        top_k = config["prediction"].get("top_k", 40)
        indices = torch.topk(logits, k=top_k, dim=-1).indices
        probs = torch.softmax(logits[:, indices.squeeze()], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = indices.gather(-1, next_token)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == vocab["<eos>"]:
            break
    
    special_tokens = {"<sos>", "<eos>", "<unk>", "<pad>"}
    output_tokens = [vocab.get_itos()[idx] for idx in input_ids[0].cpu().numpy()]
    filtered_tokens = [token for token in output_tokens if token not in special_tokens]
    return " ".join(filtered_tokens)

def main():
    config = load_config("config.yaml")
    try:
        vocab = torch.load(config["data"]["vocab_file"])
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return
    
    model = LanguageModel(**config["model"]).to(DEVICE)
    try:
        checkpoint = torch.load("checkpoints/lm_checkpoint_epoch_19.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
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