from dotenv import load_dotenv
from loguru import logger  # noqa: F401
from omegaconf import OmegaConf

_ = load_dotenv()

CFG = OmegaConf.load("./pm/config.yaml")
