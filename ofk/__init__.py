import torch
from pathlib import Path

pyd_files = list(Path(__file__).parent.glob("_C*.pyd"))
torch.ops.load_library(str(pyd_files[0]))
