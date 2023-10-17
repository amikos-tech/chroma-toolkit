import toml
from pathlib import Path

def get_config_file():
    config = None
    Path.home().joinpath(".chroma-toolkit").mkdir(exist_ok=True)
    Path.home().joinpath(".chroma-toolkit").joinpath("config.toml").touch(exist_ok=True)
    config_file = Path.home().joinpath(".chroma-toolkit").joinpath("config.toml").absolute()
    with open(config_file, "r") as f:
        config = toml.load(f)
    return config, config_file

def write_config_file(config, config_file):
    with open(config_file, "w") as f:
        toml.dump(config, f)

def get_default_env(config):
    for env in config.keys():
        if "default" in config[env] and config[env]["default"]:
            return config[env]
    return None