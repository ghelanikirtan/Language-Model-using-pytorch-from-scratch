import os 
import yaml
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(PROJECT_ROOT, 'src')


DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_STR)
