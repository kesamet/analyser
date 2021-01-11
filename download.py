"""
Script to download data.
"""
import argparse
import datetime

from analyser.utils_charts import download_data
try:
    from config import SYMBOLS
    DEST = None
except ModuleNotFoundError:
    SYMBOLS = ['ACWI', 'URTH']
    DEST = "samples"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", default="2015-01-01", type=str)
    parser.add_argument("-d", "--dest", default="data", type=str)
    args = parser.parse_args()

    start_date = args.start_date
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    print(f"\nDownloading to {args.dest}/")
    print(f"Period: {start_date} to {end_date}\n")

    for symbol in SYMBOLS:
        print(symbol)
        download_data(symbol, start_date, end_date, dirname=DEST or args.dest)
