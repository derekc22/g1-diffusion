import yaml

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
    

def dump_config(log_path, config):
    with open(log_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)