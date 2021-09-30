import os
from pathlib import Path

import torch

project_dir = Path(os.path.dirname(os.path.abspath(__file__)))

has_cuda = torch.cuda.is_available()

selected_column = []
label_column_name = "Close"

output_dir = project_dir.joinpath("result")
output_dir.mkdir(parents=True, exist_ok=True)
