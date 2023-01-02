"""
Script to download data.
"""
import argparse
import datetime

from analyser.utils_charts import download_data

try:
    from pm.config import SYMBOLS, DIRNAME

    DEST = DIRNAME
except ModuleNotFoundError:
    SYMBOLS = ["ACWI", "URTH"]
    DEST = "samples"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", default="2015-01-01", type=str)
    parser.add_argument("-d", "--dest", type=str)
    args = parser.parse_args()

    if args.dest is not None:
        dest = args.dest
    else:
        dest = DEST

    start_date = args.start_date
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    print(f"\nDownloading to {dest}/")
    print(f"Period: {start_date} to {end_date}\n")

    for i, symbol in enumerate(SYMBOLS):
        print(f"{i:2d} of {len(SYMBOLS) - 1}: {symbol}")
        download_data(symbol, start_date, end_date, dirname=dest)
