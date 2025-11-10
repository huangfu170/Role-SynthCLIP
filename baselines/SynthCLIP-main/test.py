from models import CLIP_VITB16
import torch

model=CLIP_VITB16()

checkpoint_path = "/mnt/cpfs-data/scripts/train/clip_train/baselines/SynthCLIP/checkpoint_best.pt"
checkpoint = torch.load(checkpoint_path, map_location="gpu")
load_status = model.load_state_dict(checkpoint["state_dict"])
