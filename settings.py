import os
from pathlib import Path

import torch

project_dir = Path(os.path.dirname(os.path.abspath(__file__)))

has_cuda = torch.cuda.is_available()
