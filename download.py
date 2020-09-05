"""
Script to download data.
"""
import argparse
import datetime

from analyser.symbols_dicts import keys_dict, etf_dict, reits_dict
from analyser.utils_charts import download_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", default="2015-01-01", type=str)
    parser.add_argument("-d", "--dest", default="data", type=str)
    args = parser.parse_args()

    start_date = args.start_date
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    print(f"\nDownloading to {args.dest}/")
    print(f"Period: {start_date} to {end_date}\n")

    all_symbols = list(keys_dict.keys()) + list(etf_dict.keys()) + list(reits_dict.keys())

    for symbol in all_symbols:
        print(symbol)
        download_data(symbol, start_date, end_date, dirname=args.dest)
