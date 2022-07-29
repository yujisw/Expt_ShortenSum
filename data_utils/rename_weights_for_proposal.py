import torch
from collections import OrderedDict

old_keys = [
    'encoder.layers.11.self_attn.in_proj_weight',
    'encoder.layers.11.self_attn.in_proj_bias',
    'encoder.layers.11.self_attn.out_proj.weight',
    'encoder.layers.11.self_attn.out_proj.bias',
    'encoder.layers.11.fc1.weight',
    'encoder.layers.11.fc1.bias',
    'encoder.layers.11.fc2.weight',
    'encoder.layers.11.fc2.bias',
    'encoder.layers.11.layer_norms.0.weight',
    'encoder.layers.11.layer_norms.0.bias',
    'encoder.layers.11.layer_norms.1.weight',
    'encoder.layers.11.layer_norms.1.bias',
]
weights = torch.load("../data/bart.large/model.pt")
weights["model"] = OrderedDict((k.replace("layers.11", "extractor") if k in old_keys else k, v) for k, v in weights["model"].items())
weights["model"] = OrderedDict((k.replace("encoder.", "encoder.encoder.") if "encoder." in k else k, v) for k, v in weights["model"].items())
weights["model"] = OrderedDict((k.replace("extractor.", "encoder.extractor.") if "extractor." in k else k, v) for k, v in weights["model"].items())
torch.save(weights, "../data/bart.extractor.in.encoder.large/model.pt")
