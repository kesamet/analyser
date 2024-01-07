"""
Script to download data.
"""
import argparse
import datetime

from tqdm import tqdm

from analyser.data import download_yahoofinance
from symbols import SYMBOLS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", default="2015-01-01", type=str)
    parser.add_argument("-d", "--dest", default="data", type=str)
    args = parser.parse_args()

    start_date = args.start_date
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    print(f"\nDownloading to {args.dest}/")
    print(f"Period: {start_date} to {end_date}\n")

    symbols = list(SYMBOLS.values())
    for symbol in tqdm(symbols, desc="yahoofinance"):
        download_yahoofinance(symbol, start_date, end_date, dirname=args.dest)
