import yaml
from easydict import EasyDict


def load_config(config_path):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = EasyDict(cfg)

    return cfg