"""
Script to download data.
"""

import argparse
import datetime

from loguru import logger
from tqdm import tqdm

from analyser.data import download_yfinance
from symbols import SYMBOLS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", default="2015-01-01", type=str)
    parser.add_argument("-d", "--dest", default="data", type=str)
    args = parser.parse_args()

    start_date = args.start_date
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    logger.info(f"Downloading to {args.dest}/")
    logger.info(f"Period: {start_date} to {end_date}\n")

    symbols = list(SYMBOLS.values())
    for symbol in tqdm(symbols, desc="yfinance"):
        download_yfinance(symbol, start_date, end_date, dirname=args.dest)
