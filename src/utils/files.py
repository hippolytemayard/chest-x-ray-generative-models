import yaml
from omegaconf import OmegaConf


def load_omegaconf_from_yaml(path: str):
    yaml_file = open(path, "r")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    config = OmegaConf.create(config)

    return config
