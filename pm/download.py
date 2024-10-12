"""
Script to download data.
"""

import argparse
from datetime import date

from tqdm import tqdm

from analyser.data import download_yfinance
from pm import CFG, logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", default="2015-01-01", type=str)
    parser.add_argument("-d", "--dest", type=str)
    args = parser.parse_args()

    start_date = args.start_date
    end_date = date.today().strftime("%Y-%m-%d")
    dest = args.dest or CFG.DATA_DIR
    logger.info(f"Downloading to {dest}")
    logger.info(f"Period: {start_date} to {end_date}\n")

    symbols = list(CFG.ADDITIONS.values()) + list(CFG.SYMBOLS.values())
    for symbol in tqdm(symbols, desc="yfinance"):
        download_yfinance(symbol, start_date, end_date, dirname=dest)

    # for name, symbol in tqdm(CFG.NASDAQDATA.items(), desc="nasdaqdata"):
    #     df = download_nasdaqdata(symbol)
    #     df.columns = [name]
    #     df.to_csv(f"{CFG.SUMMARY_DIR}/pe_data.csv")
