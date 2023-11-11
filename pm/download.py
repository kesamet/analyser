"""
Script to download data.
"""
import os
import argparse
from datetime import date

import nasdaqdatalink

from analyser.data import download_data
from pm import CFG
from pm.symbols import SYMBOLS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", default="2015-01-01", type=str)
    parser.add_argument("-d", "--dest", type=str)
    args = parser.parse_args()

    start_date = args.start_date
    end_date = date.today().strftime("%Y-%m-%d")
    dest = args.dest or CFG.DATA_DIR
    print(f"\nDownloading to {dest}/")
    print(f"Period: {start_date} to {end_date}\n")

    for i, symbol in enumerate(SYMBOLS):
        print(f"{i:2d} of {len(SYMBOLS)}: {symbol}")
        download_data(symbol, start_date, end_date, dirname=dest)

    print(f"{i + 1:2d} of {len(SYMBOLS)}: MULTPL/SHILLER_PE_RATIO_MONTH")
    df = nasdaqdatalink.get(
        "MULTPL/SHILLER_PE_RATIO_MONTH", authtoken=os.getenv("QUANDL_API_KEY")
    )
    df.columns = ["CAPE"]
    df.to_csv(f"{CFG.SUMMARY_DIR}/pe_data.csv")
