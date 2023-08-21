import io
import yaml


def read_config_file(config_file):
    with open(config_file, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            return config_dict
        except yaml.YAMLError as exc:
            print(exc)
            return
