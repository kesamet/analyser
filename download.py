"""
Script to download data.
"""
import argparse
import datetime

from analyser.data import download_data
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
    for i, symbol in enumerate(symbols):
        print(f"{i:2d} of {len(symbols) - 1}: {symbol}")
        download_data(symbol, start_date, end_date, dirname=args.dest)
