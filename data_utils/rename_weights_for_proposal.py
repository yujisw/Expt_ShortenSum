import torch
from collections import OrderedDict

"""
Rename names for some weights.
- encoder.layer0~10 -> encoder.encoder.layer0~10
- encoder.layer11 -> encoder.extractor
"""

weights = torch.load("data/bart.large/model.pt")
weights["model"] = OrderedDict(("encoder."+k if k.startswith("encoder.") else k, v) for k, v in weights["model"].items())
weights["model"] = OrderedDict((k.replace("encoder.encoder.layers.11", "encoder.extractor") if k.startswith("encoder.encoder.layers.11.") else k, v) for k, v in weights["model"].items())
torch.save(weights, "data/bart.extractor.in.encoder.large/model.pt")
