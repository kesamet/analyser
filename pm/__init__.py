from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf

_ = load_dotenv()

CFG = OmegaConf.load("./pm/config.yaml")
