import box
import yaml
from dotenv import load_dotenv

_ = load_dotenv()

with open("./pm/config.yaml", "r") as f:
    CFG = box.Box(yaml.safe_load(f))

SYMBOLS = list(CFG["ADD_DOWNLOAD"].values()) + list(CFG["EQ_DICT"].values())
