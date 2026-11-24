"""Provision the course-approved immutable base; never run during grading."""
from pathlib import Path
from huggingface_hub import snapshot_download

root=Path(__file__).resolve().parent.parent
snapshot_download('Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',
                  local_dir=root/'models/Qwen3-0.6B',
                  allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],
                  max_workers=2)
print('Approved model provisioned. Exclude models/ from submission ZIP.')
