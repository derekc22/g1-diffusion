import torch
import yaml

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def dump_config(log_path, config):
    with open(log_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_torch_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)
