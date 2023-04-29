import yaml


def load_yaml(filename: str) -> dict:
    """Load yaml."""
    with open(filename, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


cfg = load_yaml("./pm/config.yaml")

DATA_DIR = cfg["data_dir"]
SUMMARY_DIR = cfg["summary_dir"]
FLOWDATA = cfg["flowdata"]
